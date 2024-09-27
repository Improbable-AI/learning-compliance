from .sensor import Sensor

from isaacgym.torch_utils import *
import torch
from b1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift

class FrictionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.env.cfg.normalization.friction_range)
        return (self.env.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale
    
    def get_noise_vec(self):
        return torch.zeros(1, device=self.env.device)
    
    def get_dim(self):
        return 1