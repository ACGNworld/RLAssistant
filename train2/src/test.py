from combine import *
from stable_baselines3 import PPO

env = QuadHoverEnv(xml_file=".\\crazyfile\\scene.xml")
model = PPO.load("quad_hover_ppo")

obs, _ = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, _, terminated, _, _ = env.step(action)
    env.render()
    if terminated:
        obs, _ = env.reset()