from .sensor import Sensor

class RCSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):

        # Make sure we observe what we want 
        if self.env.cfg.commands.gait_phase_cmd_range[0] == self.env.cfg.commands.gait_phase_cmd_range[1]:
            self.env.commands[:, 5] = self.env.cfg.commands.gait_phase_cmd_range[0]
        if self.env.cfg.commands.gait_offset_cmd_range[0] == self.env.cfg.commands.gait_offset_cmd_range[1]:
            self.env.commands[:, 6] = self.env.cfg.commands.gait_offset_cmd_range[0]
        if self.env.cfg.commands.gait_bound_cmd_range[0] == self.env.cfg.commands.gait_bound_cmd_range[1]:
            self.env.commands[:, 7] = self.env.cfg.commands.gait_bound_cmd_range[0]

        # print(self.env.commands * self.env.commands_scale)

        # mask out the position commands for the force envs
        force_control_envs = self.env.force_or_position_control == 1
        obs_commands = self.env.commands * self.env.commands_scale
        # must be the last 3 of the privileged obs
        obs_commands[force_control_envs, 15:18] = 0

        return obs_commands
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(self.env.cfg.commands.num_commands, device=self.env.device)
    
    def get_dim(self):
        return self.env.cfg.commands.num_commands