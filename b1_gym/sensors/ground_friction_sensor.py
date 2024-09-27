from .sensor import Sensor

from isaacgym.torch_utils import *
import torch
from b1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift

class GroundFrictionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        ground_friction_coeffs = self.env.get_ground_frictions(range(self.env.num_envs))
        ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
                self.env.cfg.normalization.ground_friction_range)
        return (ground_friction_coeffs.unsqueeze(1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale
    
    def get_noise_vec(self):
        return torch.zeros(1, device=self.env.device)