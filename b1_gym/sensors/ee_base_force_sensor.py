from .sensor import Sensor

class EeBaseForceSensor(Sensor): # along xyz-axis
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids = None):
        return self.env.forces[:, self.env.robot_base_index, :3]
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(3, device=self.env.device)
    
    def get_dim(self):
        return 1