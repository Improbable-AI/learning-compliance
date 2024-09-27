from .sensor import Sensor
from b1_gym.utils.math_utils import quat_apply_yaw
from isaacgym.torch_utils import *

class EeGripperPositionSensor(Sensor): 
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids = None):
        return self.env.get_measured_ee_pos_spherical()
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(3, device=self.env.device)
    
    def get_dim(self):
        return 3