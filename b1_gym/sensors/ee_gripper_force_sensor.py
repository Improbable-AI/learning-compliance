from .sensor import Sensor
from b1_gym.utils.math_utils import quat_apply_yaw
from isaacgym.torch_utils import *

class EeGripperForceSensor(Sensor): 
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids = None):
        
        # world to base frame (only yaw)
        # force_yaw_base = quat_apply_yaw(quat_conjugate(self.env.base_quat[:, :4]), 
        #                                 self.env.forces[:, self.env.gripper_stator_index, :3]).view(self.env.num_envs, 3)
        # return force_yaw_base
        forces_global = self.env.forces[:, self.env.gripper_stator_index, 0:3]
        base_quat_world = self.env.base_quat.view(self.env.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        base_quat_world_indep = quat_from_euler_xyz(0 * base_rpy_world[:, 0], 0 * base_rpy_world[:, 1], base_rpy_world[:, 2])
        forces_local = quat_rotate_inverse(base_quat_world_indep, forces_global)
        
        return forces_local.view(self.env.num_envs, 3)
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(3, device=self.env.device)
    
    def get_dim(self):
        return 3