from .sensor import Sensor

from isaacgym.torch_utils import *
import torch
from b1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift

class JointDynamicsSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        joint_dynamics = torch.cat((self.env.Kp_factors[:, 0:1],
                                    self.env.Kd_factors[:, 0:1],
                                    self.env.motor_strengths[:, 0:1],
                                    ), dim=1)
        # print(joint_dynamics)
        return joint_dynamics
    
    def get_noise_vec(self):
        return torch.zeros(3, device=self.env.device)