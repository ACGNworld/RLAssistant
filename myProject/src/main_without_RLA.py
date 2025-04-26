import argparse
import os
from stable_baselines3 import SAC,PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from RLA.rla_argparser import arg_parser_postprocess

from combine import CrazyflieEnv


def mujoco_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default='CrazyFile')
    parser.add_argument('--policy_type', type=str, default='MlpPolicy')
    parser.add_argument('--seed', type=int, default=18)
    parser.add_argument('--total_timesteps', type=int, default=300000)
    parser.add_argument('--render_mode', type=str, default="human")
    parser.add_argument('--eval', action='store_true', help='Only run evaluation')
    return arg_parser_postprocess(parser).parse_args()

def make_env():
    return CrazyflieEnv(target_pos=[0, 0, 3])

args = mujoco_arg_parser()

if args.eval:
    # ========== 推理模式 ==========
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load("vec_normalize.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load("sac_quadrotor.pt", env=vec_env)
    obs = vec_env.reset()
    for _ in range(20000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = vec_env.step(action)
        vec_env.render()
        if terminated or truncated:
            obs = vec_env.reset()
    vec_env.close()

else:
    # ========== 训练模式 ==========
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    vec_env.envs[0]._max_episode_steps = 2000
    
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
    model = PPO(args.policy_type, vec_env, verbose=1, seed=args.seed, device='cpu', tensorboard_log="./ppo_tensorboard/")#TODO:没存下来

    # 开始训练
    model.learn(total_timesteps=args.total_timesteps)
    model.save("ppo_quadrotor.pt")
    vec_env.save("ppo_vec_normalize.pkl")
    vec_env.close()
