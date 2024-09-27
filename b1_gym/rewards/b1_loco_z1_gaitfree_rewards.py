import torch
import numpy as np
from b1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
import math

TRANSFORM_BASE_ARM_X = 0.2
TRANSFORM_BASE_ARM_Z = 0.1585
DEFAULT_BASE_HEIGHT = 0.6 # 0.78

INDEX_EE_POS_RADIUS_CMD = 15
INDEX_EE_POS_PITCH_CMD = 16
INDEX_EE_POS_YAW_CMD = 17
INDEX_EE_TIMING_CMD = 18
INDEX_EE_ROLL_CMD = 19
INDEX_EE_PITCH_CMD = 20
INDEX_EE_YAW_CMD = 21

INDEX_EE_FORCE_X = 12
INDEX_EE_FORCE_Y = 13
INDEX_EE_FORCE_Z = 14

class B1LocoZ1GaitfreeRewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    ###########################
    ########## ARM ############
    ###########################

    def _reward_manip_pos_tracking(self):
        '''
        Reward for manipulation tracking (EE positon)
        '''
        # Commands in spherical coordinates in the arm base frame 
        radius_cmd = self.env.commands[:, INDEX_EE_POS_RADIUS_CMD].view(self.env.num_envs, 1) 
        pitch_cmd = self.env.commands[:, INDEX_EE_POS_PITCH_CMD].view(self.env.num_envs, 1) 
        yaw_cmd = self.env.commands[:, INDEX_EE_POS_YAW_CMD].view(self.env.num_envs, 1) 

        # Spherical to cartesian coordinates in the arm base frame 
        x_cmd_arm = radius_cmd*torch.cos(pitch_cmd)*torch.cos(yaw_cmd)
        y_cmd_arm = radius_cmd*torch.cos(pitch_cmd)*torch.sin(yaw_cmd)
        z_cmd_arm = - radius_cmd*torch.sin(pitch_cmd)

        # Cartesian coordinates in the base frame
        x_cmd_base = x_cmd_arm.add_(TRANSFORM_BASE_ARM_X)
        y_cmd_base = y_cmd_arm
        z_cmd_base = z_cmd_arm.add_(TRANSFORM_BASE_ARM_Z)
        ee_position_cmd_base = torch.cat((x_cmd_base, y_cmd_base, z_cmd_base), dim=1)

        # Commands in world frame
        base_quat_world = self.env.base_quat.view(self.env.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        # Make the commands roll and pitch independent 
        base_rpy_world[:, 0] = 0.0
        base_rpy_world[:, 1] = 0.0
        base_quat_world_indep = quat_from_euler_xyz(base_rpy_world[:, 0], base_rpy_world[:, 1], base_rpy_world[:, 2]).view(self.env.num_envs,4)

        # Make the commands independent from base height 
        x_base_pos_world = self.env.base_pos[:, 0].view(self.env.num_envs, 1) 
        y_base_pos_world = self.env.base_pos[:, 1].view(self.env.num_envs, 1) 
        z_base_pos_world = torch.ones_like(self.env.base_pos[:, 2].view(self.env.num_envs, 1))*DEFAULT_BASE_HEIGHT
        base_position_world = torch.cat((x_base_pos_world, y_base_pos_world, z_base_pos_world), dim=1)

        # Command in cartesian coordinates in world frame 
        ee_position_cmd_world = quat_rotate_inverse(quat_conjugate(base_quat_world_indep), ee_position_cmd_base) + base_position_world


        # Get current ee position in world frame 
        ee_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperStator")
        ee_pos_world = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,ee_idx,0:3].view(self.env.num_envs,3)

        # print("p", ee_pos_world, ee_position_cmd_world)
        # ee_position_error = torch.sum(torch.abs(ee_position_cmd_world - ee_pos_world), dim=1)
        # ee_position_error = torch.norm(ee_position_cmd_world - ee_pos_world, dim=1)
        ee_position_error = torch.sum(torch.square(ee_position_cmd_world - ee_pos_world), dim=1)

        ee_position_coeff = 15.0

        pos_rew =  torch.exp(-ee_position_coeff*ee_position_error)
        pos_rew = pos_rew * (1 - self.env.force_or_position_control)
        # position_or_freed = torch.logical_or(self.env.force_or_position_control == 0,
        #                                        self.env.freed_envs == 1)
        # pos_rew = pos_rew * position_or_freed.float()

        # print("p", ee_pos_world, ee_position_cmd_world)

        # print("eeposiitno error: ",torch.exp(-ee_position_coeff*ee_position_error)) 
        return pos_rew
    
    def _reward_manip_combo_tracking(self):
        '''
        Reward for manipulation tracking (EE positon)
        '''
        # Commands in spherical coordinates in the arm base frame 
        radius_cmd = self.env.commands[:, INDEX_EE_POS_RADIUS_CMD].view(self.env.num_envs, 1) 
        pitch_cmd = self.env.commands[:, INDEX_EE_POS_PITCH_CMD].view(self.env.num_envs, 1) 
        yaw_cmd = self.env.commands[:, INDEX_EE_POS_YAW_CMD].view(self.env.num_envs, 1) 

        # Spherical to cartesian coordinates in the arm base frame 
        x_cmd_arm = radius_cmd*torch.cos(pitch_cmd)*torch.cos(yaw_cmd)
        y_cmd_arm = radius_cmd*torch.cos(pitch_cmd)*torch.sin(yaw_cmd)
        z_cmd_arm = - radius_cmd*torch.sin(pitch_cmd)

        # Cartesian coordinates in the base frame
        x_cmd_base = x_cmd_arm.add_(TRANSFORM_BASE_ARM_X)
        y_cmd_base = y_cmd_arm
        z_cmd_base = z_cmd_arm.add_(TRANSFORM_BASE_ARM_Z)
        ee_position_cmd_base = torch.cat((x_cmd_base, y_cmd_base, z_cmd_base), dim=1)

        # Commands in world frame
        base_quat_world = self.env.base_quat.view(self.env.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        # Make the commands roll and pitch independent 
        base_rpy_world[:, 0] = 0.0
        base_rpy_world[:, 1] = 0.0
        base_quat_world_indep = quat_from_euler_xyz(base_rpy_world[:, 0], base_rpy_world[:, 1], base_rpy_world[:, 2]).view(self.env.num_envs,4)

        # Make the commands independent from base height 
        x_base_pos_world = self.env.base_pos[:, 0].view(self.env.num_envs, 1) 
        y_base_pos_world = self.env.base_pos[:, 1].view(self.env.num_envs, 1) 
        z_base_pos_world = torch.ones_like(self.env.base_pos[:, 2].view(self.env.num_envs, 1))*DEFAULT_BASE_HEIGHT
        base_position_world = torch.cat((x_base_pos_world, y_base_pos_world, z_base_pos_world), dim=1)

        # Command in cartesian coordinates in world frame 
        ee_position_cmd_world = quat_rotate_inverse(quat_conjugate(base_quat_world_indep), ee_position_cmd_base) + base_position_world


        # Get current ee position in world frame 
        ee_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperStator")
        ee_pos_world = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,ee_idx,0:3].view(self.env.num_envs,3)

        # print("p", ee_pos_world, ee_position_cmd_world)
        # ee_position_error = torch.sum(torch.abs(ee_position_cmd_world - ee_pos_world), dim=1)
        ee_position_error = torch.norm(ee_position_cmd_world - ee_pos_world, dim=1)
        # ee_position_error = torch.sum(torch.square(ee_position_cmd_world - ee_pos_world), dim=1)

        ee_position_coeff = self.env.cfg.rewards.manip_pos_tracking_coef

        ee_rpy_yrf = self.env.get_measured_ee_rpy_yrf()

        ee_ori_cmd = self.env.commands[:, INDEX_EE_ROLL_CMD:INDEX_EE_YAW_CMD+1].clone()



        roll_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:,0]), 
                                        2*np.pi - torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:,0]))
        pitch_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 1] - ee_ori_cmd[:, 1]), 
                                        2*np.pi - torch.abs(ee_rpy_yrf[:, 1] - ee_ori_cmd[:, 1]))
        yaw_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 2] - ee_ori_cmd[:, 2]), 
                                        2*np.pi - torch.abs(ee_rpy_yrf[:, 2] - ee_ori_cmd[:, 2]))

        assert not (torch.any(torch.logical_or(roll_error < 0, roll_error > np.pi)))

        tracking_coef_manip_ori = self.env.cfg.rewards.manip_ori_tracking_coef

        ee_ori_tracking_error = roll_error + pitch_error + yaw_error
        # ee_ori_tracking_error = roll_error**2 + pitch_error**2 + yaw_error**2

        return torch.exp(-ee_position_coeff*ee_position_error - ee_ori_tracking_error * tracking_coef_manip_ori) 
    
    def _reward_manip_ori_tracking(self):
        ee_rpy_yrf = self.env.get_measured_ee_rpy_yrf()
        ee_ori_cmd = self.env.commands[:, INDEX_EE_ROLL_CMD:INDEX_EE_YAW_CMD+1].clone()

        # convert the pitch and yaw into a unit vector
        ee_current_vec = torch.stack([torch.cos(ee_rpy_yrf[:, 1]) * torch.cos(ee_rpy_yrf[:, 2]),
                                    torch.cos(ee_rpy_yrf[:, 1]) * torch.sin(ee_rpy_yrf[:, 2]),
                                    torch.sin(ee_rpy_yrf[:, 1])], dim=1)
        ee_cmd_vec = torch.stack([torch.cos(ee_ori_cmd[:, 1]) * torch.cos(ee_ori_cmd[:, 2]),
                                    torch.cos(ee_ori_cmd[:, 1]) * torch.sin(ee_ori_cmd[:, 2]),
                                    torch.sin(ee_ori_cmd[:, 1])], dim=1)

        # calculate the angle between the two vectors
        dot_product = torch.sum(ee_current_vec * ee_cmd_vec, dim=1).unsqueeze(1)
        dot_product = dot_product.clamp(-1, 1)
        angle_error = torch.acos(dot_product)

        # compute the quaternion error
        ee_current_quat = quat_from_euler_xyz(ee_rpy_yrf[:, 0], ee_rpy_yrf[:, 1], ee_rpy_yrf[:, 2])
        ee_cmd_quat = quat_from_euler_xyz(ee_ori_cmd[:, 0], ee_ori_cmd[:, 1], ee_ori_cmd[:, 2])
        ee_quat_error = quat_mul(quat_conjugate(ee_current_quat), ee_cmd_quat)
        
        norm = torch.norm(ee_quat_error, dim=1).unsqueeze(1)
        ee_quat_error_normalized = ee_quat_error / norm

        # Compute the error as the norm of the vector part
        vector_part = ee_quat_error_normalized[:, :3]  # Exclude the scalar part w
        error_value = torch.norm(vector_part, dim=1)

        # print(ee_quat_error, error_value)

        # calculate the roll error
        roll_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:,0]),
                                        2*np.pi - torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:,0]))
          
        assert not (torch.any(torch.logical_or(roll_error < 0, roll_error > np.pi)))
        tracking_coef_manip_ori = 5.0
        # tracking_coef_manip_ori = 3.0
        # ee_ori_tracking_error = roll_error**2 + pitch_error**2 + yaw_error**2
        # ee_ori_tracking_error = roll_error**2 + angle_error**2
        ee_ori_tracking_error = error_value
        # print(ee_roll_eff, ee_rpy_yrf[:, 0], roll_error)
        if self.env.cfg.rewards.maintain_ori_force_envs:
            return torch.exp(-ee_ori_tracking_error * tracking_coef_manip_ori)
        else:
            return torch.exp(-ee_ori_tracking_error * tracking_coef_manip_ori) * (1 - self.env.force_or_position_control)
    
    def _reward_manip_ori_tracking_yaw_only(self):
        ee_rpy_yrf = self.env.get_measured_ee_rpy_yrf()
        ee_ori_cmd = self.env.commands[:, INDEX_EE_ROLL_CMD:INDEX_EE_YAW_CMD+1].clone()
        
        # compute the quaternion error
        ee_current_quat = quat_from_euler_xyz(0. * ee_rpy_yrf[:, 0], 0. * ee_rpy_yrf[:, 1], ee_rpy_yrf[:, 2])
        ee_cmd_quat = quat_from_euler_xyz(0. * ee_rpy_yrf[:, 0], 0. * ee_rpy_yrf[:, 1], ee_ori_cmd[:, 2])
        ee_quat_error = quat_mul(quat_conjugate(ee_current_quat), ee_cmd_quat)
        
        norm = torch.norm(ee_quat_error, dim=1).unsqueeze(1)
        ee_quat_error_normalized = ee_quat_error / norm

        # Compute the error as the norm of the vector part
        vector_part = ee_quat_error_normalized[:, :3]  # Exclude the scalar part w
        error_value = torch.norm(vector_part, dim=1)

        tracking_coef_manip_ori = 5.0
        # tracking_coef_manip_ori = 3.0
        # ee_ori_tracking_error = roll_error**2 + pitch_error**2 + yaw_error**2
        # ee_ori_tracking_error = roll_error**2 + angle_error**2
        ee_ori_tracking_error = error_value
        # print(ee_roll_eff, ee_rpy_yrf[:, 0], roll_error)
        if self.env.cfg.rewards.maintain_ori_force_envs:
            return torch.exp(-ee_ori_tracking_error * tracking_coef_manip_ori)
        else:
            return torch.exp(-ee_ori_tracking_error * tracking_coef_manip_ori) * (1 - self.env.force_or_position_control)
    
    
    def _reward_torque_limits_arm(self):
        # penalize torques too close to the limit
        return torch.sum(torch.square(
            (torch.abs(self.env.torques[:,12:19]) - self.env.torque_limits[12:19] * self.env.cfg.rewards.soft_torque_limit_arm).clip(min=0.)), dim=1)

    def _reward_dof_vel_arm(self):
        # Penalize dof velocities
        # k_qd = -6e-4
        return torch.sum(torch.square(self.env.dof_vel[:,12:18]), dim=1)

    def _reward_dof_acc_arm(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel[:,12:18] - self.env.dof_vel[:,12:18]) / self.env.dt), dim=1)

    def _reward_action_rate_arm(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions[:,12:18] - self.env.actions[:,12:18]), dim=1)

    def _reward_action_smoothness_1_arm(self):
        # Penalize changes in actions
        # k_s1 =-2.5
        diff = torch.square(self.env.joint_pos_target[:, 12:18] - self.env.last_joint_pos_target[:, 12:18])
        diff = diff * (self.env.last_actions[:,12:18] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2_arm(self):
        # Penalize changes in actions
        # k_s2 = -1.2
        diff = torch.square(self.env.joint_pos_target[:, 12:18] - 2 * self.env.last_joint_pos_target[:, 12:18] + self.env.last_last_joint_pos_target[:, 12:18])
        diff = diff * (self.env.last_actions[:, 12:18] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, 12:18] != 0)  # ignore second step
        return torch.sum(diff, dim=1)
    
    def _reward_base_height(self):
        base_height = torch.mean(self.env.root_states[:, 2].unsqueeze(1), dim=1)
        return torch.square(base_height - self.env.cfg.rewards.base_height_target)

    def _reward_dof_pos_limits_arm(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos[:, 12:18] - self.env.dof_pos_limits[12:18, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos[:, 12:18] - self.env.dof_pos_limits[12:18, 1]).clip(min=0.)
        out_of_limits = -(self.env.joint_pos_target[:, 12:18] - self.env.dof_pos_limits[12:18, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.joint_pos_target[:, 12:18] - self.env.dof_pos_limits[12:18, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)


    ###########################
    ########## LEG ############
    ###########################

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])
    
    def _reward_survival(self):
        # Penalize torques
        return torch.ones(self.env.num_envs, device=self.env.device)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        self.env.lin_vel_tracking_error_buf[:] = lin_vel_error
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma_v_x)
    
    def _reward_tracking_ang_vel_yaw(self):
        # Tracking of angular velocity commands (yaw axis)
        ang_vel_error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        self.env.ang_vel_tracking_error_buf[:] = ang_vel_error
        return torch.exp(-ang_vel_error/ self.env.cfg.rewards.tracking_sigma_v_yaw)
    

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
                                     dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_swing = (self.env.clock_inputs < self.env.cfg.rewards.swing_ratio).float()
        
        reward = 0
        for i in range(4):
            reward += (desired_swing[:, i]) * (
                        (foot_forces[:, i] < 1.0).float())
                        # torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4
    
    

    def _reward_tracking_contacts_shaped_vel(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        # foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = (self.env.clock_inputs > (1-self.env.cfg.rewards.stance_ratio)).float()
        reward = 0
        for i in range(4):
            reward += (desired_contact[:, i]) * (
                        (foot_forces[:, i] > 1.0).float())
                        # torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
                        # torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        # print("reward: ", reward, " foot_velocities: ", foot_velocities, " desired_contact: ", desired_contact, " foot indices: ", self.env.feet_indices, " gait indices: ", self.env.gait_indices)
        return reward / 4
    
    def _reward_feet_clearance_cmd(self):
        foot_heights = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)

        desired_swing = (self.env.clock_inputs < self.env.cfg.rewards.swing_ratio).float()

        swing_progress = torch.clamp(-1 * self.env.clock_inputs, 0, 1)
        
        foot_target_height = self.env.cfg.rewards.footswing_height * desired_swing + 0.02

        # return torch.sum((foot_heights - foot_target_height)**2 * desired_swing, dim=-1)
        return torch.sum((foot_heights - foot_target_height)**2 * swing_progress, dim=-1)
    
    def _reward_torque_limits_leg(self):
        # penalize torques too close to the limit
        return torch.sum(torch.square(
            (torch.abs(self.env.torques[:,:12]) - self.env.torque_limits[:12] * self.env.cfg.rewards.soft_torque_limit_leg).clip(min=0.)), dim=1)

    def _reward_torques(self):
        # penalize torques too close to the limit
        return torch.sum(torch.square(self.env.torques[:,:12]), dim=1)
    
    def _reward_torques_arm(self):
        # penalize torques too close to the limit
        return torch.sum(torch.square(self.env.torques[:,12:]), dim=1)

    def _reward_dof_pos_limits_leg(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos[:, :12] - self.env.dof_pos_limits[:12, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos[:, :12] - self.env.dof_pos_limits[:12, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_leg(self):
        # Penalize dof velocities
        # k_qd = -6e-4
        return torch.sum(torch.square(self.env.dof_vel[:,:12]), dim=1)

    def _reward_dof_acc_leg(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel[:,:12] - self.env.dof_vel[:,:12]) / self.env.dt), dim=1)

    def _reward_action_rate_leg(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions[:,:12] - self.env.actions[:,:12]), dim=1)

    def _reward_action_smoothness_1_leg(self):
        # Penalize changes in actions
        # k_s1 =-2.5
        diff = torch.square(self.env.joint_pos_target[:, :12] - self.env.last_joint_pos_target[:, :12])
        diff = diff * (self.env.last_actions[:,:12] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2_leg(self):
        # Penalize changes in actions
        # k_s2 = -1.2
        diff = torch.square(self.env.joint_pos_target[:, :12] - 2 * self.env.last_joint_pos_target[:, :12] + self.env.last_last_joint_pos_target[:, :12])
        diff = diff * (self.env.last_actions[:, :12] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :12] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        # print('self.env.penalised_contact_indices: ', self.env.penalised_contact_indices)
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)


    ########################
    # force tracking rewards
    ########################
    
    def _reward_ee_force_x(self):
        
        xy_forces_global = self.env.forces[:, self.env.gripper_stator_index, 0:3]
        base_quat_world = self.env.base_quat.view(self.env.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        base_quat_world_indep = quat_from_euler_xyz(0 * base_rpy_world[:, 0], 0 * base_rpy_world[:, 1], base_rpy_world[:, 2])
        xy_forces_local = quat_rotate_inverse(base_quat_world_indep, xy_forces_global)
        
        force_magn_meas = (xy_forces_local[:, 0]).view(self.env.num_envs, 1)
        force_magn_cmd = (self.env.commands[:, INDEX_EE_FORCE_X]).view(self.env.num_envs, 1)
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.env.num_envs)

        force_magn_coeff = self.env.cfg.rewards.sigma_force_z
        return torch.exp(-force_magn_coeff*force_magn_error) * self.env.force_or_position_control
    
    def _reward_ee_force_y(self):
        
        xy_forces_global = self.env.forces[:, self.env.gripper_stator_index, 0:3]
        base_quat_world = self.env.base_quat.view(self.env.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        base_quat_world_indep = quat_from_euler_xyz(0 * base_rpy_world[:, 0], 0 * base_rpy_world[:, 1], base_rpy_world[:, 2])
        xy_forces_local = quat_rotate_inverse(base_quat_world_indep, xy_forces_global)

        force_magn_meas = (xy_forces_local[:, 1]).view(self.env.num_envs, 1)
        force_magn_cmd = (self.env.commands[:, INDEX_EE_FORCE_Y]).view(self.env.num_envs, 1)
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.env.num_envs)

        force_magn_coeff = self.env.cfg.rewards.sigma_force_z
        return torch.exp(-force_magn_coeff*force_magn_error) * self.env.force_or_position_control
    
    def _reward_ee_force_z(self):

        force_magn_meas = (self.env.forces[:, self.env.gripper_stator_index, 2]).view(self.env.num_envs, 1)
        force_magn_cmd = (self.env.commands[:, INDEX_EE_FORCE_Z]).view(self.env.num_envs, 1)
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.env.num_envs)

        force_magn_coeff = self.env.cfg.rewards.sigma_force_z
        return torch.exp(-force_magn_coeff*force_magn_error) * self.env.force_or_position_control

    def _reward_ee_force_magnitude_x_pen(self):
        
        force_magn_meas = torch.abs(self.env.forces[:, self.env.gripper_stator_index, 0]).view(self.env.num_envs, 1)
        force_magn_cmd = 0.0 
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.env.num_envs)

        return force_magn_error  * self.env.force_or_position_control
    
    def _reward_ee_force_magnitude_y_pen(self):
       
        force_magn_meas = torch.abs(self.env.forces[:, self.env.gripper_stator_index, 1]).view(self.env.num_envs, 1)
        force_magn_cmd = 0.0 
        force_magn_error = torch.abs(force_magn_meas - force_magn_cmd).view(self.env.num_envs)

        return force_magn_error  * self.env.force_or_position_control

    def _reward_dof_pos(self):
        # Penalize dof positions
        # k_q = -0.75
        return torch.sum(torch.square(self.env.dof_pos[:, :12] - self.env.default_dof_pos[:, :12]), dim=1)

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # # nominal positions: [FR, FL, RR, RL]
        # if self.env.cfg.commands.num_commands >= 13:
        #     desired_stance_width = self.env.commands[:, 12:13]
        #     desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        # else:
        # desired_stance_width = 0.55
        desired_stance_width = self.env.cfg.rewards.stance_width
        desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        # if self.env.cfg.commands.num_commands >= 14:
        #     desired_stance_length = self.env.commands[:, 13:14]
        #     desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        # else:
        # desired_stance_length = 0.85
        desired_stance_length = self.env.cfg.rewards.stance_length
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.env.commands[:, 4]
        x_vel_des = self.env.commands[:, 0:1]
        yaw_vel_des = self.env.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward