from .sensor import Sensor

class JointPositionTargetSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        

        # print("joint pos delta: ", (self.env.dof_pos[0, :self.env.num_actuated_dof] - \
        #         self.env.default_dof_pos[0, :self.env.num_actuated_dof]))
        
        
        obs = self.env.joint_pos_target

        if self.env.cfg.commands.control_only_z1:
            obs[:, :12] = 0.0 # legs obs to 0
        # if self.env.num_actuated_dof > 18:
        #     obs[:, 17:19] = 0.0 # joint6 and gripper obs to 0

        return obs
    
    def get_noise_vec(self):
        import torch
        noise_vec = torch.zeros(self.env.num_actuated_dof, device=self.env.device)
        return noise_vec
    
    def get_dim(self):
        return self.env.num_actuated_dof