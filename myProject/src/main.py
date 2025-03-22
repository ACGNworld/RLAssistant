import gymnasium as gym

from stable_baselines3 import PPO,SAC
import argparse
from RLA.rla_argparser import arg_parser_postprocess
from RLA import exp_manager
from RLA import logger
from combine import CrazyflieEnv

def mujoco_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('--policy_type',  type=str, default='MlpPolicy')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--total_timesteps', type=int, default=30000)
    parser.add_argument('--render_mode', type=str, default="human")
    # [RLA] add RLA parameters
    parser = arg_parser_postprocess(parser)
    return parser


args = mujoco_arg_parser().parse_args()
# [RLA] config RLA.
task_name = 'quadrotor-RLA-PPO'
exp_manager.set_hyper_param(**vars(args))
exp_manager.add_record_param(["info", "seed", 'env'])
exp_manager.configure(task_name, rla_config='./rla_config.yaml', data_root='./')
exp_manager.log_files_gen()
exp_manager.print_args()


# env = gym.make(args.env)
env = CrazyflieEnv(target_pos=[0,0,3])

# model = SAC(args.policy_type, env, verbose=1, seed=args.seed)
model = PPO(args.policy_type, env, verbose=1, seed=args.seed)

# [RLA] mask the function name of logger to be consistent with the one in sb3.
logger.record = logger.record_tabular
logger.dump = logger.dump_tabular
model._logger = logger
model._custom_logger = True

model.learn(total_timesteps=args.total_timesteps)
model.save("ppo_quadrotor.pt")
env.close()

env = CrazyflieEnv(target_pos=[0,0,3],render_mode="human")
obs, _ = env.reset()
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)  # Pass only `obs`, not the tuple
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()  # Reset and unpack only `obs`

env.close()