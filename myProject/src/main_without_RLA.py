import gymnasium as gym

from ppo import PPO
import argparse
from RLA.rla_argparser import arg_parser_postprocess
from combine import CrazyflieEnv

def mujoco_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('--policy_type',  type=str, default='MlpPolicy')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--total_timesteps', type=int, default=100000)
    parser.add_argument('--render_mode', type=str, default="human")
    # [RLA] add RLA parameters
    parser = arg_parser_postprocess(parser)
    return parser


args = mujoco_arg_parser().parse_args()



# env = gym.make(args.env)
env = CrazyflieEnv()

model = PPO(args.policy_type, env, verbose=1, seed=args.seed)
# [RLA] mask the function name of logger to be consistent with the one in sb3.
model._custom_logger = True

model.learn(total_timesteps=args.total_timesteps)
model.save("ppo_quadrotor.pt")
obs, _ = env.reset()
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)  # Pass only `obs`, not the tuple
    obs, reward, terminated, truncated, info = env.step(action)
    env.render(render_mode=args.render_mode)
    if terminated or truncated:
        obs, _ = env.reset()  # Reset and unpack only `obs`

env.close()