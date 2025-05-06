from stable_baselines3 import SAC,PPO
from typing import Dict, Optional, Union
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

# 电机参数
gravity = 9.8066        # 重力加速度 单位m/s^2
mass = 0.033            # 飞行器质量 单位kg
arm_length = 0.065/2.0  # 电机力臂长度 单位m
Ct = 3.25e-4            # 电机推力系数 (N/krpm^2)
Cd = 7.9379e-6          # 电机反扭系数 (Nm/krpm^2)
max_speed = 22          # 电机最大转速(krpm)
max_thrust = Ct*(max_speed**2)    # 单个电机最大推力 单位N
max_torque = Cd*(max_speed**2)   # 单个电机最大扭矩 单位Nm

# 混控器类
class Mixer:
    def __init__(self,Ct=3.25e-4,Cd=7.9379e-6,L=0.065/2.0,max_speed=22,max_thrust=0.1573,max_torque=3.842e-03):
        self.Ct  = Ct   # 电机推力系数 (N/krpm^2) 注意结果单位为力(N)
        self.Cd  = Cd  # 电机反扭系数 (Nm/krpm^2) 注意结果单位为扭矩(Nm)
        self.L   = L   # 电机力臂长度 单位m
        self.max_speed   = max_speed # 电机最大转速(krpm)
        self.max_thrust  = max_thrust # 单个电机最大推力 单位N (电机最大转速22krpm)
        self.max_torque  = max_torque # 单个电机最大扭矩 单位Nm (电机最大转速22krpm)
        # 动力分配正向矩阵
        self.mat = np.array([
            [self.Ct, self.Ct, self.Ct, self.Ct],                                   # F total
            [self.Ct*self.L, -self.Ct*self.L, -self.Ct*self.L, self.Ct*self.L],     # Mx + - - +
            [-self.Ct*self.L, -self.Ct*self.L, self.Ct*self.L, self.Ct*self.L],     # My - - + +
            [-self.Cd, self.Cd, -self.Cd, self.Cd]                                  # Mz - + - +
        ])
        # 动力分配逆向矩阵
        self.inv_mat = np.linalg.inv(self.mat)

    # 动力分配
    # thrust: 机体总推力 单位N
    # mx, my, mz: 三轴扭矩 单位Nm
    def calculate(self, thrust, mx, my, mz):
        Mx, My = mx, my  # Copy
        Mz = 0 # 首先进行X Y轴分配
        control_input = np.array([thrust, Mx, My, Mz])
        motor_speed_squ = self.inv_mat @ control_input
        # X Y Z三轴动力分配的顺序决定最终取舍的不同
        # 一般情况下 首先对X Y轴动力进行分配 余量用于分配Z轴
        max_value = np.max(motor_speed_squ)
        min_value = np.min(motor_speed_squ)
        ref_value = np.sum(motor_speed_squ) / 4.0  # 参考转速(不施加扭矩时的转速平方)
        # print(f"ref_value:{ref_value}")
        max_trim_scale = 1.0
        min_trim_scale = 1.0
        if max_value > self.max_speed **2: # 存在电机动力饱和 计算缩放因子进行缩放
            # print(f"Max Overflow")
            max_trim_scale = (self.max_speed ** 2 - ref_value)/(max_value - ref_value)
        if min_value < 0: # 存在电机动力负饱和 计算缩放因子进行缩放
            # print(f"Min Overflow")
            min_trim_scale = (ref_value)/(ref_value - min_value)
        scale = min(max_trim_scale, min_trim_scale)
        # print(f"Trim Scale:{scale}")
        # 对X Y扭矩施加缩放因子
        Mx = Mx * scale  
        My = My * scale
        # 重新计算电机转速平方
        control_input = np.array([thrust, Mx, My, Mz])
        motor_speed_squ = self.inv_mat @ control_input
        # print(f"motor_speed_squ:{motor_speed_squ}")
        # print(f"Original Torque: Mx:{Mx/scale:.6f} My:{My/scale:.6f} Trimed Torque: Mx:{Mx:.6f} My:{My:.6f}")
        if scale < 1.0: # 存在Trim 不进行Z轴扭矩分配 直接返回
            # 这里需要强行进行一下绝对值
            motor_speed_squ = np.abs(motor_speed_squ)
            return np.sqrt(motor_speed_squ)  # 返回电机转速
        else: # 仍然有余量 可以进行Z轴扭矩分配
            Mz = mz
            control_input_withz = np.array([thrust, Mx, My, Mz])  # 添加Z轴扭矩重新计算
            motor_speed_squ_withz = self.inv_mat @ control_input_withz
            # 判断是否饱和
            max_value = np.max(motor_speed_squ_withz)
            min_value = np.min(motor_speed_squ_withz)
            max_index = np.argmax(motor_speed_squ_withz)
            min_index = np.argmin(motor_speed_squ_withz)
            max_trim_scale_z = 1.0
            min_trim_scale_z = 1.0
            if max_value > self.max_speed **2: # 存在电机动力饱和 计算缩放因子进行缩放
                # print(f"Z Max Overflow")
                max_trim_scale_z = (self.max_speed ** 2 - motor_speed_squ[max_index])/(max_value - motor_speed_squ[max_index])
            if min_value < 0: # 存在电机动力负饱和 计算缩放因子进行缩放
                # print(f"Z Min Overflow")
                min_trim_scale_z = (motor_speed_squ[min_index])/(motor_speed_squ[min_index] - min_value)
            scale_z = min(max_trim_scale_z, min_trim_scale_z)
            # 对Z轴扭矩施加缩放因子
            Mz = Mz * scale_z
            # 重新计算电机转速平方
            control_input_withz = np.array([thrust, Mx, My, Mz])
            motor_speed_squ_withz = self.inv_mat @ control_input_withz
            # print(f"motor_speed_squ:{motor_speed_squ_withz}")
            # print(f"Original Torque: Mx:{Mx/scale:.6f} My:{My/scale:.6f} Mz:{Mz/scale_z:.6f} Trimed Torque: Mx:{Mx:.6f} My:{My:.6f} Mz:{Mz:.6f}")
            motor_speed_squ = np.abs(motor_speed_squ)
            return np.sqrt(motor_speed_squ_withz)  # 返回电机转速

# 电机输入转换函数
def calc_motor_input(krpm):
    if krpm > max_speed:
        krpm = max_speed
    elif krpm < 0:
        krpm = 0
    _force = Ct * krpm**2
    _input = _force / max_thrust
    return np.clip(_input, 0, 1)

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 150,
    "elevation": -30,
    "lookat": [0, 0, 2]
}

def quaternion2euler(quaternion):
    r = R.from_quat(np.roll(quaternion, -1))
    return r.as_euler('xyz')

def position_reward(current_pos, target_pos):
    xy_error = np.linalg.norm(current_pos[:2] - target_pos[:2])
    z_error = abs(current_pos[2] - target_pos[2])
    # print("z_error:",z_error) #debug
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
        
        # 初始化混控器
        self.mixer = Mixer(Ct, Cd, arm_length, max_speed, max_thrust, max_torque)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=self.observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        # 动作空间: [总推力(0-1), 滚转扭矩(-1到1), 俯仰扭矩(-1到1), 偏航扭矩(-1到1)]
        self.action_space = Box(
            low=np.array([0, -1, -1, -1]), 
            high=np.array([1, 1, 1, 1]), 
            dtype=np.float32
        )

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

    def step(self, action):
        # 解包动作并转换为物理量
        thrust = action[0] * 4 * max_thrust  # 总推力(0-4*max_thrust)
        mx = action[1] * max_torque          # 滚转扭矩
        my = action[2] * max_torque          # 俯仰扭矩
        mz = action[3] * max_torque          # 偏航扭矩
        
        # 使用混控器计算电机转速(krpm)
        motor_speeds = self.mixer.calculate(thrust, mx, my, mz)
        
        # 转换为Mujoco控制信号(0-1)
        motor_commands = np.array([
            calc_motor_input(motor_speeds[0]),
            calc_motor_input(motor_speeds[1]),
            calc_motor_input(motor_speeds[2]),
            calc_motor_input(motor_speeds[3])
        ])
        
        # 执行仿真
        self.do_simulation(motor_commands, self.frame_skip)

        # 获取状态
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        lin_vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        euler = quaternion2euler(quat)
        
        # 计算奖励
        r_pos = position_reward(pos, self.target_pos)
        r_hover = hover_stability_reward(euler, lin_vel, ang_vel)
        reward = r_pos + r_hover

        # 终止条件
        terminated = bool(
            pos[2] < 0.05 or
            pos[2] > (self.target_pos[2] + 3) or
            np.any(np.abs(pos[:2]) > 5.0) or
            np.any(np.abs(euler[:2]) > np.radians(45))  # 转换为弧度比较
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

if __name__ == "__main__":
    env = CrazyflieEnv(render_mode="human", target_pos=[0,0,3])
    model = PPO.load("ppo_quadrotor.pt", env, device='cpu')
    obs, _ = env.reset()
    for i in range(20000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()  # Reset and unpack only `obs`

    env.close()