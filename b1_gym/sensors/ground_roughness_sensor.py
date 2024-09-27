from .sensor import Sensor

from isaacgym.torch_utils import *
import torch
from b1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift

class GroundRoughnessSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        ground_roughness = self.env.get_ground_roughness(range(self.env.num_envs))
        roughness_scale, roughness_shift = get_scale_shift(self.env.cfg.normalization.roughness_range)
        return (ground_roughness.unsqueeze(1) - roughness_shift) * roughness_scale
    
    def get_noise_vec(self):
        return torch.zeros(1, device=self.env.device)