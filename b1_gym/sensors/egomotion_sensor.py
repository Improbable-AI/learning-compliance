from .sensor import Sensor

from isaacgym.torch_utils import *
import torch
from b1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift

class EgomotionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        motion_scale, motion_shift = get_scale_shift(self.env.cfg.normalization.motion)
        global_body_motion = self.env.base_pos - self.env.prev_base_pos
        global_body_motion = quat_apply_yaw(quat_from_angle_axis(-1 * self.env.heading_offsets.unsqueeze(1), torch.Tensor([0, 0, 1]).to(self.env.device))[:, 0, :], global_body_motion)
        reset_env_ids = self.env.reset_buf.nonzero(as_tuple=False).flatten()
        global_body_motion[reset_env_ids] = 0.0
        # resolve teleportation by clipping for now
        global_body_motion[global_body_motion > 0.5] = 0.0
        global_body_motion[global_body_motion < -0.5] = 0.0
        
        return (global_body_motion - motion_shift) * motion_scale
    
    def get_noise_vec(self):
        return torch.zeros(3, device=self.env.device)