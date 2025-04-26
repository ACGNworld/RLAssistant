from stable_baselines3 import SAC,PPO
from typing import Dict, Optional, Union
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 150,
    "elevation": -30,
    "lookat": [0, 0, 2]
}

def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz')  # 返回弧度
    return np.sin(euler)    # 保持姿态连续性

def position_reward(current_pos, target_pos):
    xy_error = np.linalg.norm(current_pos[:2] - target_pos[:2])
    z_error = abs(current_pos[2] - target_pos[2])
    reward_xy = np.exp(-2.0 * xy_error)
    reward_z = np.exp(-3.0 * z_error)
    if xy_error < 0.1 and z_error < 0.05:
        reward_xy += 1.0
    return reward_xy + reward_z

def hover_stability_reward(euler, lin_vel, ang_vel):
    return -0.05 * (
        np.sum(np.square(euler)) +  # 欧拉角平方
        np.linalg.norm(lin_vel) +
        0.5 * np.linalg.norm(ang_vel)
    )

class CrazyflieEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 250,
    }

    def __init__(
        self,
        xml_file: str = "./crazyfile/scene.xml",
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        target_pos: Optional[np.ndarray] = None,
        reset_noise_scale: float = 0.01,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, xml_file, frame_skip, default_camera_config, reset_noise_scale, **kwargs
        )

        self.target_pos = target_pos if target_pos is not None else np.array([0, 0, 3])
        self._reset_noise_scale = reset_noise_scale

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=self.observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))
        self.i = 0

    def step(self, action):
        scaled_action = (action + 1) / 2
        scaled_action = np.clip(scaled_action, 0, 1)
        self.do_simulation(scaled_action, self.frame_skip)

        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        lin_vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        euler = quaternion2euler(quat)

        # 奖励计算
        r_pos = position_reward(pos, self.target_pos)#TODO:越飞越高
        r_hover = hover_stability_reward(euler, lin_vel, ang_vel)
        reward = r_pos + r_hover + 0.01  # 生存奖励

        # 终止条件（单位为弧度）
        terminated = bool(
            pos[2] < 0.05 or
            pos[2] > (self.target_pos[2] + 2) or
            np.any(np.abs(pos[:2]) > 5.0) or
            np.any(np.abs(euler[:2]) > np.sin(45))
        )
        if terminated:
            reward -= 10.0

        info = {
            "position_error": np.linalg.norm(pos[:2] - self.target_pos[:2]),
            "height_error": abs(pos[2] - self.target_pos[2]),
            "target_position": self.target_pos,
            "current_position": pos,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, info

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + noise * np.random.randn(self.model.nq)
        qvel = self.init_qvel + noise * np.random.randn(self.model.nv)
        # qpos[3:7] = [0, 0, 0, 1]  # 初始为水平姿态
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        pos = self.data.qpos[:3]
        euler = quaternion2euler(self.data.qpos[3:7])
        vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        return np.concatenate([pos, euler, vel, ang_vel])

    def viewer_setup(self):
        self.viewer.cam.distance = 0.8
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -30

# # 注册环境
# MujocoEnv.register(
#     id="Crazyflie-v1",
#     entry_point="crazyflie_env:CrazyflieEnv",
#     max_episode_steps=1000,
#     kwargs={"xml_file": "./crazyfile/assets/scene.xml"}
# )

if __name__ == "__main__":
    env = CrazyflieEnv(render_mode="human",target_pos=[0,0,3])

    # model = SAC.load("sac_quadrotor.pt", env)
    model = PPO.load("ppo_quadrotor.pt", env,device='cpu')
    env = CrazyflieEnv(target_pos=[0,0,3],render_mode="human")
    obs, _ = env.reset()
    for i in range(20000):
        action, _ = model.predict(obs, deterministic=True)  # Pass only `obs`, not the tuple
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()  # Reset and unpack only `obs`

    env.close()