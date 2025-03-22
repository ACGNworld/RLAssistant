__credits__ = ["fjj"]

from stable_baselines3 import PPO,SAC
from typing import Dict, Optional, Union
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 0.5,
    "azimuth": 150,
    "elevation": -30,
    "lookat": [0, 0, 0.3]
}

class CrazyflieEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 250,
    }

    def __init__(
        self,
        xml_file: str = ".\crazyfile\scene.xml",
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        target_pos: Optional[np.ndarray] = None,
        reset_noise_scale: float = 0.01,
        **kwargs,
    ):
        # 环境初始化
        utils.EzPickle.__init__(
            self, xml_file, frame_skip, default_camera_config, reset_noise_scale, **kwargs
        )
        
        # 目标点设置（x, y, z）
        self.target_pos = target_pos if target_pos is not None else np.array([0, 0, 3])
        
        # 观测空间：位置(3) + 四元数(4) + 线速度(3) + 角速度(3)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # 动作空间：四个电机的推力（0-1）
        self.action_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=self.observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # 设置元数据
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.i = int(0)

    def step(self, action):
        # 应用电机推力（带噪声）
        motor_noise = 0.01 * np.random.randn(4)
        # self.data.ctrl[:] = np.clip(action, 0, 1)

        # 执行物理仿真
        self.do_simulation(action, self.frame_skip)
        
        # 获取当前状态
        current_pos = self.data.qpos[:3]
        current_height = current_pos[2]
        
        # 计算奖励
        pos_error = np.linalg.norm(current_pos[:2] - self.target_pos[:2])
        height_error = abs(current_height - self.target_pos[2])
        crash_panelty = 0
        if current_height < 0.05: crash_panelty = -100 
        reward = - 0.2 * pos_error - 0.5 * height_error + crash_panelty  # 加权惩罚
        #少一个航向，少一个稳定
        
        # 终止条件
        terminated = bool(
            current_height < 0.05  # 坠毁检测
            or current_height > (self.target_pos[2] + 2)  # 高度超出范围
            or np.any(np.abs(current_pos[:2])) > 5.0  # 飞出边界
        )
        # truncated = terminated
        
        # 附加信息
        info = {
            "position_error": pos_error,
            "height_error": height_error,
            "target_position": self.target_pos,
            "current_position": current_pos
        }
        
        # 处理渲染
        if self.render_mode == "human":
            self.render()
        
        self.i += 1
        if self.i > 100:
            print(action)  #仅调试
            print(info)
            self.i = 0

        return self._get_obs(), reward, terminated, False, info

    def reset_model(self):
        # 重置初始状态（带噪声）
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        """组合观测向量：
        - 位置 (3)
        - 四元数 (4)
        - 线速度 (3)
        - 角速度 (3)
        """
        return np.concatenate([
            self.data.qpos[:3],      # x, y, z 位置
            self.data.qpos[3:7],     # 四元数姿态
            self.data.qvel[:3],      # 线速度
            self.data.qvel[3:6]      # 角速度
        ])

    def viewer_setup(self):
        """自定义视角设置"""
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

    # env.close()
    # model = SAC("MlpPolicy", env, verbose=1)
    model = PPO.load("ppo_quadrotor.pt", env)
    env = CrazyflieEnv(target_pos=[0,0,3],render_mode="human")
    obs, _ = env.reset()
    for i in range(20000):
        action, _ = model.predict(obs, deterministic=True)  # Pass only `obs`, not the tuple
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()  # Reset and unpack only `obs`

    env.close()