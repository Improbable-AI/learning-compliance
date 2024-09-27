from .sensor import Sensor

class LastActionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset
        self.delay = delay

    def get_observation(self, env_ids = None):
        if self.delay == 0:
            # if self.env.num_actuated_dof > 18:
            #     self.env.actions[:, 17:19] = 0.0 # joint6 and gripper actions to 0
            return self.env.actions
        elif self.delay == 1:
            # if self.env.num_actuated_dof > 18:
            #     self.env.last_actions[:, 17:19] = 0.0 # joint6 and gripper actions to 0
            return self.env.last_actions
        else:
            raise NotImplementedError("Action delay of {} not implemented".format(self.delay))
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(self.env.num_actions, device=self.env.device)
    
    def get_dim(self):
        return self.env.num_actions