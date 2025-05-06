import argparse
import os
from stable_baselines3 import SAC,PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn

from RLA.rla_argparser import arg_parser_postprocess
from RLA import exp_manager, logger

from combine import CrazyflieEnv


def mujoco_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default='CrazyFile')
    parser.add_argument('--policy_type', type=str, default='MlpPolicy')
    parser.add_argument('--seed', type=int, default=18)
    parser.add_argument('--total_timesteps', type=int, default=100000)
    parser.add_argument('--render_mode', type=str, default="human")
    parser.add_argument('--eval', action='store_true', help='Only run evaluation')
    parser.add_argument('--nr',action='store_true',help='enable RLAssisant logger')
    return arg_parser_postprocess(parser).parse_args()

def make_env():
    return CrazyflieEnv(target_pos=[0, 0, 3])
def make_env_human():
    return CrazyflieEnv(target_pos=[0, 0, 3],render_mode="human")


args = mujoco_arg_parser()

# ===== 自定义策略网络（Sigmoid 输出层） =====
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 替换 action_net 为 Sigmoid 输出
        self.action_net = nn.Sequential(
            nn.Linear(64, 4),  # 假设隐藏层是64维，输出4个电机推力
            nn.Sigmoid()       # 强制输出在 [0,1]
        )

# ==== RLA config ====
if not args.nr:
    task_name = 'quadrotor-RLA-PPO'
    exp_manager.set_hyper_param(**vars(args))
    exp_manager.add_record_param(["info", "seed", 'env'])
    exp_manager.configure(task_name, rla_config='./rla_config.yaml', data_root='./')
    exp_manager.log_files_gen()
    exp_manager.print_args()


if args.eval:
    # ========== 推理模式 ==========
    vec_env = DummyVecEnv([make_env_human])
    vec_env = VecNormalize.load("ppo_vec_normalize.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load("ppo_quadrotor.pt", env=vec_env)
    obs = vec_env.reset()
    for _ in range(20000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = vec_env.step(action)
        vec_env.render()
        if terminated:
            obs = vec_env.reset()
    vec_env.close()

else:
    # ========== 训练模式 ==========
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    vec_env.envs[0]._max_episode_steps = 5000
    
    # model = SAC(
    #     "MlpPolicy",
    #     vec_env,
    #     learning_rate=1e-3,        # 增大学习率
    #     buffer_size=200000,        # 更大的回放缓冲区
    #     batch_size=256,            # 更大的批次
    #     tau=0.01,                  # 更快的目标网络更新
    #     train_freq=(1, "episode"), # 每episode更新一次
    #     gradient_steps=32,         # 更多梯度步
    #     ent_coef='auto',           # 自动调整熵系数
    #     target_entropy='auto',     # 确保自动熵调整生效
    #     verbose=1,
    #     device='cuda'
    # )
    model = PPO(
        CustomActorCriticPolicy,  # 替换为自定义策略，原来是args.policy_type
        vec_env,
        verbose=1,
        seed=args.seed,
        device='cpu' 
    )

    # [RLA] SB3 logger兼容
    if not args.nr:
        logger.record = logger.record_tabular
        logger.dump = logger.dump_tabular
        model._logger = logger
        model._custom_logger = True

    # 开始训练
    print("动作空间:", vec_env.action_space)  # TODO:检查为什么不一样
    print(model.policy)
    if args.nr:
        exit(0)
    model.learn(total_timesteps=args.total_timesteps)
    model.save("ppo_quadrotor.pt")
    backup_model_path = os.path.join(exp_manager.log_dir,"ppo_quadrotor.pt")
    model.save(backup_model_path)
    vec_env.save("ppo_vec_normalize.pkl")
    vec_env.close()
