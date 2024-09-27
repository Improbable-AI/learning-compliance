# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
import numpy as np
from typing import Dict, List

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch

from b1_gym import MINI_GYM_ROOT_DIR
from b1_gym.envs.base.base_task import BaseTask
from b1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift, plus_2pi_wrap_to_pi
from b1_gym.utils.terrain import Terrain, perlin
from .legged_robot_config import Cfg

TRANSFORM_BASE_ARM_X = 0.2
TRANSFORM_BASE_ARM_Z = 0.1585
DEFAULT_BASE_HEIGHT = 0.6
INDEX_EE_POS_RADIUS_CMD = 15
INDEX_EE_POS_PITCH_CMD = 16
INDEX_EE_POS_YAW_CMD = 17
INDEX_EE_POS_TIMING_CMD = 18

INDEX_EE_ROLL_CMD = 19
INDEX_EE_PITCH_CMD = 20
INDEX_EE_YAW_CMD = 21

INDEX_EE_FORCE_X = 12
INDEX_EE_FORCE_Y = 13
INDEX_EE_FORCE_Z = 14

INDEX_FORCE_OR_POSITION_INDICATOR = 22

class LeggedRobot(BaseTask):
    def __init__(self, cfg: Cfg, sim_params, physics_engine, sim_device, headless,
                 initial_dynamics_dict=None, terrain_props=None, custom_heightmap=None):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """

        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.initial_dynamics_dict = initial_dynamics_dict
        self.terrain_props = terrain_props
        self.custom_heightmap = custom_heightmap
        self.first_sim_time_step = True
        self._parse_cfg(self.cfg)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()

        self._prepare_reward_function()
        self.init_done = True
        self.record_now = False
        self.collecting_evaluation = False
        self.num_still_evaluating = 0
        self.forces_deactivated = False

    def pre_physics_step(self):
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.render_gui()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame

        self.pre_physics_step()
        
        # self.read_dof_vel = self.dof_vel[:, 12:19]
        for o in range(self.cfg.control.decimation):

            if self.cfg.commands.control_only_z1:
                self.actions[:, :12] = torch.zeros((self.num_envs, 12), dtype=torch.float32, device=self.device)

            if self.cfg.env.num_actions >= 18:
                self.actions[:, 18] = -0.1 # set joint6 and gripper to 0  

            self._compute_torques(self.actions)
            if self.cfg.asset.default_dof_drive_mode == 0: # mixed
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.joint_pos_target))
                apply_torques = self.torques.clone()
                # print(apply_torques.shape)
                apply_torques[:, 12:] = 0
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(apply_torques))
            elif self.cfg.asset.default_dof_drive_mode == 1: # position
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.joint_pos_target))
            elif self.cfg.asset.default_dof_drive_mode == 3: # torque
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            else:
                raise Exception("Control mode missing!")

            if self.first_sim_time_step:
                if torch.any(self.episode_length_buf > 1):
                    self.envs_nb = torch.arange(self.num_envs, device=self.device)
                    # Get gripper cartesian coordinates in world frame
                    self.ee_init_pos_world = self.rigid_body_state[:, self.gripper_stator_index, 0:3].view(self.num_envs, 3).clone() # Define the reference to which the target positions offset will be defined
                    self.first_sim_time_step = False
            else: 
                env_ids = self.envs_nb[self.episode_length_buf == 1] 
                if env_ids.nelement() > 0 :
                    # Get gripper cartesian coordinates in world frame
                    self.ee_init_pos_world[env_ids] = self.rigid_body_state[env_ids, self.gripper_stator_index, 0:3].view(len(env_ids), 3).clone()

            # push gripper 
            # print("DONT PUSH")
            self._push_gripper(torch.arange(self.num_envs, device=self.device), self.cfg)      
            # push robot base 
            self._push_robot_base(torch.arange(self.num_envs, device=self.device), self.cfg) 

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.GLOBAL_SPACE)
            self.gym.simulate(self.sim)
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        self.extras["int"] = (2100 - self.compute_energy()) * 0.0000003 #0.0001
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        if not self.headless:
            if self.cfg.viewer.follow_robot:
                forward_vec = quat_apply_yaw(self.base_quat, torch.tensor([2., 0., -1.0], dtype=torch.float32, device=self.device))
                self.set_camera(self.base_pos[0, :] - forward_vec, self.base_pos[0, :])

        self.episode_length_buf += 1
        # print("ep_len", self.episode_length_buf, "self.rewards.soft_torque_limit ", self.cfg.rewards.soft_torque_limit)
        # self.cfg.rewards.soft_torque_limit = self.cfg.rewards.soft_torque_start - self.cfg.rewards.soft_torque_target/self.cfg.env.episode_length_s*self.episode_length_buf*self.dt
        self.common_step_counter += 1
        
        self._update_path_distance()

        # print(f" base position world = {self.rigid_body_state[0, 0, 0:3]}")
        # print(f" FL foot position world = {self.rigid_body_state[0, 4, 0:3]}")       
        # print(f" link1 position world = {self.rigid_body_state[0, 17, 0:3]}")

        # link1_pos_base = quat_rotate_inverse(self.base_quat, self.rigid_body_state[:, 17, 0:3] - self.rigid_body_state[:, 0, 0:3])
        # print(f" link1 position body = {link1_pos_base[0,:]}")

        # print(f" link1 position body = {self.rigid_body_state[0, 17, 0:3]}")

        # prepare quantities
        self.base_pos[:] = self.root_states[self.robot_actor_idxs, 0:3]
        self.base_quat[:] = self.root_states[self.robot_actor_idxs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        
        self.gripper_position = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_stator_index, 0:3]
        self.gripper_velocity = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_stator_index, 7:10]
        
        if self.cfg.env.add_balls:
            self.randomize_ball_state()

            self.object_pos_world_frame[:] = self.root_states[self.object_actor_idxs, 0:3] 
            robot_object_vec = self.asset.get_local_pos()
            true_object_local_pos = quat_rotate_inverse(self.base_quat, robot_object_vec)
            true_object_local_pos[:, 2] = 0.0*torch.ones(self.num_envs, dtype=torch.float,
                                       device=self.device, requires_grad=False)

            # simulate observation delay
            self.object_local_pos = self.simulate_ball_pos_delay(true_object_local_pos, self.object_local_pos)
            self.object_lin_vel = self.asset.get_lin_vel()
            self.object_ang_vel = self.asset.get_ang_vel()
        # print("object linear velocity: ", self.object_lin_vel)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()


        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:, :] = self.actions[:, :self.num_actuated_dof]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[self.robot_actor_idxs, 7:13]
        
        # Compute ee position commands for the next time step
        if self.cfg.commands.num_commands > 17:# and not self.cfg.commands.force_control:
            self.compute_intermediate_ee_pos_command(env_ids)

        self._render_headless()

    def is_ee_cmd_feasible(self, radius_cmd, pitch_cmd):
        
        commands_feasible = True
        # Spherical to cartesian coordinates in the arm base frame 
        z_cmd_arm = - radius_cmd*torch.sin(pitch_cmd)

        # Cartesian coordinates in the world frame
        z_cmd_world = z_cmd_arm.add_(TRANSFORM_BASE_ARM_Z + DEFAULT_BASE_HEIGHT)
        # self.z_cmd_world = z_cmd_world

        env_ids = torch.arange(radius_cmd.shape[0], device=self.device)
        env_ids_resample = env_ids[z_cmd_world < 0.05] # 0 = no force applied 
        if env_ids_resample.nelement() > 0 :
            # print("resampling command for env_ids ", z_cmd_world)
            commands_feasible = False

        # print(" z_cmd_world = ", z_cmd_world, " for :r = ", radius_cmd, ', p = ', pitch_cmd)
        return commands_feasible, env_ids_resample

    def get_measured_ee_pos_spherical(self) -> torch.Tensor:
        '''
        Get the current ee position in the arm frame in spherical coordinates 

        Returns:
            - radius, pitch, yaw (size: (num_envs, 3))
        '''
        # Get gripper cartesian coordinates in world frame
        ee_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[0], "gripperStator")
        ee_position_world = self.rigid_body_state[:, ee_idx, 0:3].view(self.num_envs, 3) # env.rigid_body_state.shape = (num_envs, num_rigid_bodies, 13) = (1, 24, 13)
        
        # Make the commands roll and pitch independent
        base_quat_world = self.base_quat.view(self.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        # base_quat_world_indep = quat_from_euler_xyz(base_rpy_world[:, 0], base_rpy_world[:, 1], base_rpy_world[:, 2])
        base_quat_world_indep = quat_from_euler_xyz(0 * base_rpy_world[:, 0], 0 * base_rpy_world[:, 1], base_rpy_world[:, 2])

        # Make the commands independent from base height 
        x_base_pos_world = self.base_pos[:, 0].view(self.num_envs, 1) 
        y_base_pos_world = self.base_pos[:, 1].view(self.num_envs, 1) 
        z_base_pos_world = torch.ones_like(self.base_pos[:, 2].view(self.num_envs, 1))*DEFAULT_BASE_HEIGHT
        base_position_world = torch.cat((x_base_pos_world, y_base_pos_world, z_base_pos_world), dim=1)

        # Measured ee position in the base frame
        ee_position_base = quat_rotate_inverse(base_quat_world_indep, ee_position_world - base_position_world).view(self.num_envs,3)

        # Measured ee position in the arm frame in cartesian coordinates 
        ee_position_arm = torch.zeros_like(ee_position_base)
        ee_position_arm[:,0] = ee_position_base[:,0].add_(-TRANSFORM_BASE_ARM_X)
        ee_position_arm[:,1] = ee_position_base[:,1]
        ee_position_arm[:,2] = ee_position_base[:,2].add_(-TRANSFORM_BASE_ARM_Z)

        # Spherical to cartesian coordinates in the arm base frame 
        radius = torch.norm(ee_position_arm, dim=1).view(self.num_envs,1)
        pitch = -torch.asin(ee_position_arm[:,2].view(self.num_envs,1)/radius).view(self.num_envs,1)
        yaw = torch.atan2(ee_position_arm[:,1].view(self.num_envs,1), ee_position_arm[:,0].view(self.num_envs,1)).view(self.num_envs,1)
        ee_pos_sphe_arm = torch.cat((radius, pitch, yaw), dim=1).view(self.num_envs,3)

        # compute error
        radius_cmd = self.commands[:, INDEX_EE_POS_RADIUS_CMD].view(self.num_envs, 1) 
        pitch_cmd = self.commands[:, INDEX_EE_POS_PITCH_CMD].view(self.num_envs, 1) 
        yaw_cmd = self.commands[:, INDEX_EE_POS_YAW_CMD].view(self.num_envs, 1) 

        # Spherical to cartesian coordinates in the arm base frame 
        x_cmd_arm = radius_cmd*torch.cos(pitch_cmd)*torch.cos(yaw_cmd)
        y_cmd_arm = radius_cmd*torch.cos(pitch_cmd)*torch.sin(yaw_cmd)
        z_cmd_arm = - radius_cmd*torch.sin(pitch_cmd)

        # Cartesian coordinates in the base frame
        x_cmd_base = x_cmd_arm.add_(TRANSFORM_BASE_ARM_X)
        y_cmd_base = y_cmd_arm
        z_cmd_base = z_cmd_arm.add_(TRANSFORM_BASE_ARM_Z)
        ee_position_cmd_base = torch.cat((x_cmd_base, y_cmd_base, z_cmd_base), dim=1)

        ee_position_cmd_world = quat_rotate_inverse(quat_conjugate(base_quat_world_indep), ee_position_cmd_base) + base_position_world

        # self.gripper_pos_tracking_error_buf = torch.abs(ee_position_cmd_world - ee_position_world).sum(dim=1)
        self.gripper_pos_tracking_error_buf = torch.norm(ee_position_cmd_world - ee_position_world, dim=1)
        # self.gripper_pos_tracking_error_buf = torch.sum(torch.square(ee_position_cmd_world - ee_position_world), dim=1)

        return ee_pos_sphe_arm
    
    def get_measured_ee_rpy_yrf(self) -> torch.Tensor:
        ee_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[0], "gripperStator")
        ee_quat = self.rigid_body_state[:, ee_idx, 3:7].view(self.num_envs, 4)

        base_quat_world = self.base_quat.view(self.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        quat_yrf = quat_from_euler_xyz(torch.zeros_like(base_rpy_world[:, 0], dtype=torch.float, device=self.device), 
                                    torch.zeros_like(base_rpy_world[:, 1], dtype=torch.float, device=self.device), 
                                    base_rpy_world[:, 2])

        ee_quat_yrf = quat_mul(quat_conjugate(quat_yrf), ee_quat)
        ee_rpy_yrf = torch.stack(get_euler_xyz(ee_quat_yrf), dim=1)

        ee_rpy_yrf = plus_2pi_wrap_to_pi(ee_rpy_yrf)

        # get tracking error
        ee_ori_cmd = self.commands[:, INDEX_EE_ROLL_CMD:INDEX_EE_YAW_CMD+1].clone()

        roll_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:,0]), 
                                        2*np.pi - torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:,0]))
        pitch_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 1] - ee_ori_cmd[:, 1]), 
                                        2*np.pi - torch.abs(ee_rpy_yrf[:, 1] - ee_ori_cmd[:, 1]))
        yaw_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 2] - ee_ori_cmd[:, 2]), 
                                        2*np.pi - torch.abs(ee_rpy_yrf[:, 2] - ee_ori_cmd[:, 2]))

        # self.gripper_ori_tracking_error_buf = torch.stack((roll_error, pitch_error, yaw_error), dim=1).sum(dim=1)
        self.gripper_ori_tracking_error_buf = torch.norm(torch.stack((roll_error, pitch_error, yaw_error), dim=1), dim=1)

        return ee_rpy_yrf
    

    def set_gripper_teleop_value(self, value:float):
        self.teleop_gripper_value = value*torch.ones_like(self.teleop_gripper_value)

    def set_joint6_teleop_value(self, value:float):
        self.teleop_joint6_value = value*torch.ones_like(self.teleop_joint6_value)

    def set_trajectory_time(self, value:float):
        self.trajectory_time = value*torch.ones_like(self.trajectory_time)

    def set_initial_ee_pos(self):
        self.initial_ee_pos = self.get_measured_ee_pos_spherical()

    def set_initial_ee_rpy(self):
        self.initial_ee_rpy = self.get_measured_ee_rpy_yrf()

    def set_target_joint_angles(self, target_joints: List[float]):
        self.target_joint_values = target_joints

    def set_pd_gains(self):
        self.p_gains[17] = self.p_gains[17]*6/4
        self.d_gains[17] = self.d_gains[17]*6/4


    def compute_intermediate_ee_pos_command(self, env_ids):
        '''
        The ee position commands in spherical coordinates (radius, pitch, yaw)

        Args:
            env_ids (list[int]): List of environment ids which have been reset

        self.reset_buf starts with 1's -> every env is reset at the very beginning -> ee_target_pos_cmd is defined 
        '''

        # update ee pos meas
        ee_pos_meas = self.get_measured_ee_pos_spherical()
        ee_rpy_meas = self.get_measured_ee_rpy_yrf()

        # Init current and target ee positions only once for all envs
        if self.init_training:
            self.init_training = False

            # Define the new long term target ee position command 
            self.ee_target_pos_cmd = self.commands[:, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)].view(self.num_envs, 3).clone() # radius, pitch, yaw
            self.ee_target_rpy_cmd = self.commands[:, INDEX_EE_ROLL_CMD:(INDEX_EE_YAW_CMD+1)].view(self.num_envs, 3).clone()
            
            # Define the first ee position
            self.first_ee_pos = ee_pos_meas
            self.initial_ee_pos[:] = self.first_ee_pos[:]
            self.first_ee_rpy = ee_rpy_meas
            self.initial_ee_rpy[:] = self.first_ee_rpy[:]

            # print("first_ee_pos: ", self.first_ee_pos)
            # print("first_ee_rpy: ", self.first_ee_rpy)

            # input((self.first_ee_pos, self.first_ee_rpy))
        
        # When some envs have been reset, a new target ee position command is resampled in reset_idx, need to update first ee pos and target ee pos  
        if len(env_ids) > 0:
            # print("--reset")
            # Reset trajectory time 
            self.trajectory_time[env_ids] = 0.0

            # Define the new long term target ee position command 
            self.ee_target_pos_cmd[env_ids] = self.commands[env_ids, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)].view(len(env_ids), 3) # radius, pitch, yaw
            self.ee_target_rpy_cmd[env_ids] = self.commands[env_ids, INDEX_EE_ROLL_CMD:(INDEX_EE_YAW_CMD+1)].view(len(env_ids), 3)
            
            # # Define the first ee position
            self.initial_ee_pos[env_ids] = ee_pos_meas[env_ids, :]#self.first_ee_pos[env_ids, :]
            self.initial_ee_rpy[env_ids] = ee_rpy_meas[env_ids, :]#self.first_ee_rpy[env_ids, :]

            self._resample_force_or_position_control(env_ids)

        # Interpolate intermediary ee position commands 
        T_traj = self.commands[:, INDEX_EE_POS_TIMING_CMD] # size num_envs

        # Make sure that the interpolated target ee position saturates after T_traj
        
        env_ids_inter = (self.trajectory_time.view(self.num_envs) < T_traj).nonzero(as_tuple=False).flatten()

        self.commands[:, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)] = self.ee_target_pos_cmd.view(self.num_envs,3)
        self.commands[:, INDEX_EE_ROLL_CMD:(INDEX_EE_YAW_CMD+1)] = self.ee_target_rpy_cmd.view(self.num_envs,3)
        self.commands[:, INDEX_FORCE_OR_POSITION_INDICATOR] = self.force_or_position_control.view(self.num_envs)

        if self.cfg.commands.interpolate_ee_cmds:
            if len(env_ids_inter):
                new_command = self.trajectory_time.view(self.num_envs,1)/T_traj.view(self.num_envs,1)*self.ee_target_pos_cmd.view(self.num_envs,3) + (1 - self.trajectory_time.view(self.num_envs,1)/T_traj.view(self.num_envs,1))*self.initial_ee_pos.view(self.num_envs,3)
                self.commands[env_ids_inter, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)] = new_command[env_ids_inter, :]
                #print("Next cmd:   ", self.commands[env_ids_inter, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)])
                drpy = plus_2pi_wrap_to_pi(self.ee_target_rpy_cmd - self.initial_ee_rpy)
                assert not torch.any(torch.abs(drpy) > np.pi)
                new_rpy_command = plus_2pi_wrap_to_pi(self.initial_ee_rpy + self.trajectory_time.view(self.num_envs,1)/T_traj.view(self.num_envs,1)*drpy)
                new_rpy_command[:, 0] = torch.clamp(new_rpy_command[:, 0], self.cfg.commands.limit_end_effector_roll[0], self.cfg.commands.limit_end_effector_roll[1])
                new_rpy_command[:, 1] = torch.clamp(new_rpy_command[:, 1], self.cfg.commands.limit_end_effector_pitch[0], self.cfg.commands.limit_end_effector_pitch[1])
                new_rpy_command[:, 2] = torch.clamp(new_rpy_command[:, 2], self.cfg.commands.limit_end_effector_yaw[0], self.cfg.commands.limit_end_effector_yaw[1])
                self.commands[env_ids_inter, INDEX_EE_ROLL_CMD:(INDEX_EE_YAW_CMD+1)] = new_rpy_command[env_ids_inter, :]
        
                # print("inter ee pos", self.commands[env_ids_inter, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)])
                # print("inter ee rpy", self.commands[env_ids_inter, INDEX_EE_ROLL_CMD:(INDEX_EE_YAW_CMD+1)])
                # print("current ee pos", ee_pos_meas)
                # print("current ee rpy", ee_rpy_meas)
                # print("goal ee pos", self.ee_target_pos_cmd[env_ids_inter, :])
                # print("goal ee rpy", self.ee_target_rpy_cmd[env_ids_inter, :])

        # Resample commands 2 seconds after reaching the target 
        env_ids_resample = (self.trajectory_time.view(self.num_envs) > (T_traj + self.cfg.commands.settle_time)).nonzero(as_tuple=False).flatten()
        if self.cfg.commands.interpolate_ee_cmds:
            if len(env_ids_resample):
                # print("T_traj + self.cfg.commands.settle_time : ", T_traj + self.cfg.commands.settle_time)
                # Reset trajectory time 
                self.trajectory_time[env_ids_resample] = 0.0

                # Define the new long term target ee position command 
                new_radius_cmd = torch_rand_float(self.cfg.commands.ee_sphe_radius[0], self.cfg.commands.ee_sphe_radius[1], (len(env_ids_resample), 1), device=self.device).view(len(env_ids_resample))
                new_pitch_cmd = torch_rand_float(self.cfg.commands.ee_sphe_pitch[0], self.cfg.commands.ee_sphe_pitch[1], (len(env_ids_resample), 1), device=self.device).view(len(env_ids_resample))
                new_yaw_cmd = torch_rand_float(self.cfg.commands.ee_sphe_yaw[0], self.cfg.commands.ee_sphe_yaw[1], (len(env_ids_resample), 1), device=self.device).view(len(env_ids_resample))
                #new_T_cmd = torch_rand_float(self.cfg.commands.ee_timing[0], self.cfg.commands.ee_timing[1], (len(env_ids_resample), 1), device=self.device).view(len(env_ids_resample)) 
                # new_T_cmd = torch_rand_float(3.98, 4.02, (len(env_ids_resample), 1), device=self.device).view(len(env_ids_resample))                     

                new_roll_ori_cmd = torch_rand_float(self.cfg.commands.end_effector_roll[0], self.cfg.commands.end_effector_roll[1], (len(env_ids_resample), 1), device=self.device).view(len(env_ids_resample))
                new_pitch_ori_cmd = torch_rand_float(self.cfg.commands.end_effector_pitch[0], self.cfg.commands.end_effector_pitch[1], (len(env_ids_resample), 1), device=self.device).view(len(env_ids_resample))
                new_yaw_ori_cmd = torch_rand_float(self.cfg.commands.end_effector_yaw[0], self.cfg.commands.end_effector_yaw[1], (len(env_ids_resample), 1), device=self.device).view(len(env_ids_resample))
                                                    
                # Resample commands until they are all above the ground 
                commands_feasible, env_ids_reresample = self.is_ee_cmd_feasible(new_radius_cmd, new_pitch_cmd)
                while not commands_feasible:
                    # print("in resample")
                    new_radius_cmd[env_ids_reresample] = torch_rand_float(self.cfg.commands.ee_sphe_radius[0], self.cfg.commands.ee_sphe_radius[1], (len(env_ids_reresample), 1), device=self.device).view(len(env_ids_reresample))
                    new_pitch_cmd[env_ids_reresample] = torch_rand_float(self.cfg.commands.ee_sphe_pitch[0], self.cfg.commands.ee_sphe_pitch[1], (len(env_ids_reresample), 1), device=self.device).view(len(env_ids_reresample))
                    
                    commands_feasible, env_ids_reresample = self.is_ee_cmd_feasible(new_radius_cmd, new_pitch_cmd)
                
                
                # Define the first ee position
                # self.initial_ee_pos[env_ids_resample] = self.initial_ee_pos[env_ids_resample, :]
                # self.initial_ee_rpy[env_ids_resample] = self.initial_ee_rpy[env_ids_resample, :]
                self.initial_ee_pos[env_ids_resample] = self.ee_target_pos_cmd[env_ids_resample, :]
                self.initial_ee_rpy[env_ids_resample] = self.ee_target_rpy_cmd[env_ids_resample, :]
                # print("Resample current cmd: ", self.initial_ee_pos)  
                
                # print("Final command: ", self.z_cmd_world)
                self.ee_target_pos_cmd[env_ids_resample, 0] = new_radius_cmd
                self.ee_target_pos_cmd[env_ids_resample, 1] = new_pitch_cmd
                self.ee_target_pos_cmd[env_ids_resample, 2] = new_yaw_cmd

                self.ee_target_rpy_cmd[env_ids_resample, 0] = new_roll_ori_cmd
                self.ee_target_rpy_cmd[env_ids_resample, 1] = new_pitch_ori_cmd
                self.ee_target_rpy_cmd[env_ids_resample, 2] = new_yaw_ori_cmd


                # print("Resample target cmd: ", self.ee_target_pos_cmd, "T ",self.commands[env_ids_resample, INDEX_EE_POS_TIMING_CMD])
          

        # Increase the time 
        self.trajectory_time += self.dt

        # print("self.commands[0, 15] =", self.commands[0, INDEX_EE_POS_RADIUS_CMD])

    def randomize_ball_state(self):
        reset_ball_pos_mark = np.random.choice([True, False],self.num_envs, p=[self.cfg.ball.pos_reset_prob,1-self.cfg.ball.pos_reset_prob])
        reset_ball_pos_env_ids = torch.tensor(np.array(np.nonzero(reset_ball_pos_mark)), device = self.device).flatten()# reset_ball_pos_mark.nonzero(as_tuple=False).flatten()
        ball_pos_env_ids = self.object_actor_idxs[reset_ball_pos_env_ids].to(device=self.device)
        reset_ball_pos_env_ids_int32 = ball_pos_env_ids.to(dtype = torch.int32)
        self.root_states[ball_pos_env_ids,0:3] += 2*(torch.rand(len(ball_pos_env_ids), 3, dtype=torch.float, device=self.device,
                                                     requires_grad=False)-0.5) * torch.tensor(self.cfg.ball.pos_reset_range,device=self.device,
                                                     requires_grad=False)
        # self.root_states[ball_pos_env_ids, 2] = 0.08* torch.ones(len(ball_pos_env_ids), device = self.device, requires_grad=False)
        # self.root_states[ball_pos_env_ids, :3] += self.env_origins[reset_ball_pos_env_ids]

        reset_ball_vel_mark = np.random.choice([True, False],self.num_envs, p=[self.cfg.ball.vel_reset_prob,1-self.cfg.ball.vel_reset_prob])
        reset_ball_vel_env_ids = torch.tensor(np.array(np.nonzero(reset_ball_vel_mark)), device = self.device).flatten()
        ball_vel_env_ids = self.object_actor_idxs[reset_ball_vel_env_ids].to(device=self.device)
        self.root_states[ball_vel_env_ids,7:10] = 2*(torch.rand(len(ball_vel_env_ids), 3, dtype=torch.float, device=self.device,
                                                     requires_grad=False)-0.5) * torch.tensor(self.cfg.ball.vel_reset_range,device=self.device,
                                                     requires_grad=False)
                                            
        reset_ball_vel_env_ids_int32 = ball_vel_env_ids.to(dtype = torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(torch.cat((reset_ball_pos_env_ids_int32, reset_ball_vel_env_ids_int32))), len(reset_ball_pos_env_ids_int32) + len(reset_ball_vel_env_ids_int32))

    def simulate_ball_pos_delay(self, new_ball_pos, last_ball_pos):
        receive_mark = np.random.choice([True, False],self.num_envs, p=[self.cfg.ball.vision_receive_prob,1-self.cfg.ball.vision_receive_prob])
        last_ball_pos[receive_mark,:] = new_ball_pos[receive_mark,:]

        return last_ball_pos

    def check_termination(self):
        """ Check if environments need to be reset
        """

        self.contact_buf = torch.zeros_like(self.reset_buf, dtype=torch.bool)

        # # Terminate on contacts only with envs to which no force is being applied: avoid earfly termination for negative forces gripper  
        # env_ids = torch.arange(self.num_envs, device=self.device)
        # env_ids_no_force_applied = env_ids[self.selected_env_ids_perturb == 0] # 0 = no force applied 
        # if env_ids_no_force_applied.nelement() > 0 :
        #     # if torch.any((torch.norm(self.contact_forces[env_ids_no_force_applied, :, :][:, self.termination_contact_indices, :], dim=-1) > 1.).view(env_ids_no_force_applied.shape[0], self.termination_contact_indices.shape[0]),
        #     #                         dim=1):
        #     #     print("norm force: ", (torch.norm(self.contact_forces[env_ids_no_force_applied, :, :][:, self.termination_contact_indices, :], dim=-1) > 1.).view(env_ids_no_force_applied.shape[0], self.termination_contact_indices.shape[0]))
        #     # print(" guess: ", (torch.norm(self.contact_forces[env_ids_no_force_applied, :, :][:, 5, :], dim=-1) > 1.).view(env_ids_no_force_applied.shape[0], 1))
        #     self.contact_buf[env_ids_no_force_applied] = torch.any((torch.norm(self.contact_forces[env_ids_no_force_applied, :, :][:, self.termination_contact_indices, :], dim=-1) > 1.).view(env_ids_no_force_applied.shape[0], self.termination_contact_indices.shape[0]),
        #                             dim=1)
        # # same for base 
        # env_ids_no_force_applied_robot = env_ids[self.selected_env_ids_robot == 0] # 0 = no force applied 
        # if env_ids_no_force_applied_robot.nelement() > 0 :
        #     self.contact_buf[env_ids_no_force_applied_robot] = torch.any((torch.norm(self.contact_forces[env_ids_no_force_applied_robot, :, :][:, self.termination_contact_indices, :], dim=-1) > 1.).view(env_ids_no_force_applied_robot.shape[0], self.termination_contact_indices.shape[0]),
        #                             dim=1)
            
        self.reset_buf = torch.clone(self.contact_buf)
        # print(f'1. contact: {torch.any(self.reset_buf)}')
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length  # no terminal reward for time-outs


        self.reset_buf |= self.time_out_buf
        # print(f'2. timeout: {torch.any(self.reset_buf)}')

        if self.cfg.rewards.use_terminal_body_height:
            self.body_height_buf = torch.mean(self.root_states[self.robot_actor_idxs, 2].unsqueeze(1) - self.measured_heights, dim=1) \
                                   < self.cfg.rewards.terminal_body_height

            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

        if self.cfg.rewards.use_terminal_torque_legs_limits:
            above_torque_lim = torch.any((torch.abs(self.torques[:,:12]) - self.torque_limits[:12] * self.cfg.rewards.soft_torque_limit_leg) > 0.0, dim=1).view(self.num_envs)
            sim_started_while_ago = self.episode_length_buf > self.cfg.rewards.termination_torque_min_time

            self.legs_torque_lim_buff = torch.logical_and(above_torque_lim, sim_started_while_ago)
            # if torch.any(self.legs_torque_lim_buff): print("--------------------------- Reset because of torque lim legs!")
            self.reset_buf = torch.logical_or(self.legs_torque_lim_buff, self.reset_buf)

        if self.cfg.rewards.use_terminal_torque_arm_limits:
            above_torque_lim = torch.any((torch.abs(self.torques[:,12:]) - self.torque_limits[12:] * self.cfg.rewards.soft_torque_limit_arm) > 0.0, dim=1).view(self.num_envs)
            sim_started_while_ago = self.episode_length_buf > self.cfg.rewards.termination_torque_min_time

            self.arm_torque_lim_buff = torch.logical_and(above_torque_lim, sim_started_while_ago)
            # if torch.any(self.arm_torque_lim_buff): print("--------------------------- Reset because of torque lim arm!")
            self.reset_buf = torch.logical_or(self.arm_torque_lim_buff, self.reset_buf)
            
        if self.cfg.rewards.use_terminal_ee_position:
            ee_position_outside_box = (torch.norm(self.gripper_position - self.ee_init_pos_world, dim=1) > self.cfg.rewards.terminal_ee_distance).view(self.num_envs)
            sim_started_while_ago = self.episode_length_buf > self.cfg.rewards.termination_torque_min_time

            self.ee_position_lim_buff = torch.logical_and(ee_position_outside_box, sim_started_while_ago)
            # if torch.any(self.ee_position_lim_buff): print("--------------------------- Reset because of ee posit!: ", torch.norm(self.gripper_position - self.ee_init_pos_world, dim=1))
            self.reset_buf = torch.logical_or(self.ee_position_lim_buff, self.reset_buf)

        if self.cfg.rewards.use_terminal_roll_pitch:
            self.body_ori_buf = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) \
                                > self.cfg.rewards.terminal_body_ori

            self.reset_buf = torch.logical_or(self.body_ori_buf, self.reset_buf)

            
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """


        if len(env_ids) == 0:
            return

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # reset robot states
        self._resample_commands(env_ids)
        self._randomize_dof_props(env_ids, self.cfg)
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids, self.cfg)
            self.refresh_actor_rigid_shape_props(env_ids, self.cfg)

        if self.cfg.env.force_control_init_poses:
            self.init_idx = torch.randint(self.force_control_base_pos_init.shape[0], (len(env_ids),))
        # print("init idx: ", self.init_idx)
        self._reset_dofs(env_ids, self.cfg)
        self._reset_root_states(env_ids, self.cfg)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.path_distance[env_ids] = 0.
        self.past_base_pos[env_ids] = self.base_pos.clone()[env_ids]
        self.reset_buf[env_ids] = 1
        self.torques[env_ids] = 0

        # Reset forces
        # if self.cfg.commands.force_control:
            # self.forces[env_ids, self.gripper_stator_index, :3] = 0.
            # self.selected_env_ids[env_ids] = 0
            # self.push_end_time[env_ids] = 0.
            # self.force_target[env_ids, :3] = 0.
            # self.push_duration[env_ids] = 0.
            # self.current_Fxyz_cmd[env_ids, :3] = 0.
            # self.commands[env_ids, INDEX_EE_FORCE_MAGNITUDE] = 0.0
            # self.commands[env_ids, INDEX_EE_FORCE_DIRECTION] = 0.0
            # self.commands[env_ids, INDEX_EE_FORCE_Z] = 0.0
        
        force_control_envs = env_ids[self.force_or_position_control[env_ids] == 1]
        self.forces[force_control_envs, self.gripper_stator_index, :3] = 0.
        self.selected_env_ids[force_control_envs] = 0
        self.push_end_time[force_control_envs] = 0.
        self.force_target[force_control_envs, :3] = 0.
        self.push_duration[force_control_envs] = 0.
        self.current_Fxyz_cmd[force_control_envs, :3] = 0.
        self.commands[force_control_envs, INDEX_EE_FORCE_X] = 0.0
        self.commands[force_control_envs, INDEX_EE_FORCE_Y] = 0.0
        self.commands[force_control_envs, INDEX_EE_FORCE_Z] = 0.0

        # Reset push gripper 
        if self.cfg.domain_rand.push_gripper_stators:
            self.forces[env_ids, self.gripper_stator_index, :3] = 0.
            self.selected_env_ids[env_ids] = 0
            self.push_end_time[env_ids] = 0.
            self.push_force[env_ids, :3] = 0.

        # Reset push robot base 
        if self.cfg.domain_rand.push_robot_base:
            self.forces[env_ids, self.robot_base_index, :3] = 0.
            self.selected_env_ids_robot[env_ids] = 0
            self.push_end_time_robot[env_ids] = 0.
            self.push_force_robot[env_ids, :3] = 0.
        
        
        # reset history buffers
        if hasattr(self, "obs_history"):
            self.obs_history_buf[env_ids, :] = 0
            self.obs_history[env_ids, :] = 0

        self.extras = self.logger.populate_log(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.gait_indices[env_ids] = 0

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

    def set_idx_pose(self, env_ids, dof_pos, base_state):
        if len(env_ids) == 0:
            return

        env_ids_long = env_ids.to(dtype=torch.long, device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32, device=self.device)
        robot_actor_idxs_int32 = self.robot_actor_idxs.to(dtype=torch.int32)

        # joints
        if dof_pos is not None:
            self.dof_pos[env_ids] = dof_pos
            self.dof_vel[env_ids] = 0.

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(robot_actor_idxs_int32[env_ids_long]), len(env_ids_long))

        # base position
        self.root_states[self.robot_actor_idxs[env_ids_long]] = base_state.to(self.device)
        # self.root_states[self.object_actor_idxs[env_ids_long]] = ball_init_state.to(self.device)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(torch.cat((self.robot_actor_idxs[env_ids_long], self.object_actor_idxs[env_ids_long]))), 2*len(env_ids_int32))

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]

            # FORCE_REWARDS = ["ee_force_z", "ee_force_magnitude", "ee_force_direction_angle"]
            # POS_REWARDS = ["manip_ori_tracking", "manip_pos_tracking"]

            # if name in FORCE_REWARDS:
            #     rew = rew * self.force_or_position_control
            # elif name in POS_REWARDS:
            #     rew = rew * (1 - self.force_or_position_control)
            
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
            if name == "ee_force_z":
                self.save_ee_force_z = self.reward_functions[i]()* self.reward_scales[name]
            if name == "ee_force_magnitude":
                self.save_ee_force_magnitude = self.reward_functions[i]()* self.reward_scales[name]
            if name == "ee_force_direction_angle":
                self.save_ee_force_direction_angle = self.reward_functions[i]()* self.reward_scales[name]
            if name == "torque_limits":
                self.save_rew_torque = rew
            if name == "feet_contact_forces":
                self.save_rew_feet_contact = rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
        self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew
            
        self.rew_buf[:] = self.rew_buf * self.cfg.rewards.total_rew_scale

        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1

    def initialize_sensors(self):
        """ Initializes sensors
        """
        from b1_gym.sensors import ALL_SENSORS
        self.sensors = []
        for sensor_name in self.cfg.sensors.sensor_names:
            if sensor_name in ALL_SENSORS.keys():
                self.sensors.append(ALL_SENSORS[sensor_name](self, **self.cfg.sensors.sensor_args[sensor_name]))
            else:
                raise ValueError(f"Sensor {sensor_name} not found.")

        # privileged sensors
        self.privileged_sensors = []
        for privileged_sensor_name in self.cfg.sensors.privileged_sensor_names:
            if privileged_sensor_name in ALL_SENSORS.keys():
                # print(privileged_sensor_name)
                self.privileged_sensors.append(ALL_SENSORS[privileged_sensor_name](self, **self.cfg.sensors.privileged_sensor_args[privileged_sensor_name]))
            else:
                raise ValueError(f"Sensor {privileged_sensor_name} not found.")
        

        # initialize noise vec
        self.add_noise = self.cfg.noise.add_noise
        noise_vec = []
        for sensor in self.sensors:
            noise_vec.append(sensor.get_noise_vec())

        self.noise_scale_vec = torch.cat(noise_vec, dim=-1).to(self.device)

    def compute_observations(self):
        """ Computes observations
        """
        # aggregate the sensor data
        self.pre_obs_buf = []
        for sensor in self.sensors:
            self.pre_obs_buf += [sensor.get_observation()]
            #print("sensor type: ", sensor, " observation: ", sensor.get_observation())

        # print(torch.cat(self.pre_obs_buf, dim=-1).shape)
        
        self.pre_obs_buf = torch.reshape(torch.cat(self.pre_obs_buf, dim=-1), (self.num_envs, -1))
        self.obs_buf[:] = self.pre_obs_buf

        self.privileged_obs_buf_list = []
        # aggregate the privileged observations
        for sensor in self.privileged_sensors:
            self.privileged_obs_buf_list += [sensor.get_observation()]
        # print("privileged_obs_buf: ", self.privileged_obs_buf)
        if len(self.privileged_obs_buf_list):
            self.privileged_obs_buf = torch.reshape(torch.cat(self.privileged_obs_buf_list, dim=-1), (self.num_envs, -1))
        # print("self.privileged_obs_buf: ", self.privileged_obs_buf)
        # add noise if needed
        if self.cfg.noise.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        

        # assert self.privileged_obs_buf.shape[
        #            1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"

    
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type

        from b1_gym.terrains import ALL_TERRAINS
        if mesh_type not in ALL_TERRAINS.keys():
            raise ValueError(f"Terrain mesh type {mesh_type} not recognised. Allowed types are {ALL_TERRAINS.keys()}")
        
        self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        self.terrain_obj = ALL_TERRAINS[mesh_type](self)

        self._create_envs()
        
        self.terrain_obj.initialize()

        self.set_lighting()

    def set_lighting(self):
        light_index = 0
        intensity = gymapi.Vec3(0.5, 0.5, 0.5)
        ambient = gymapi.Vec3(0.2, 0.2, 0.2)
        direction = gymapi.Vec3(0.01, 0.01, 1.0)
        self.gym.set_light_parameters(self.sim, light_index, intensity, ambient, direction)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def add_actor_critic(self, actor_critic):
        self.reward_container.actor_critic = actor_critic

    def add_teacher_actor_critic(self, teacher_actor_critic):
        self.reward_container.teacher_actor_critic = teacher_actor_critic

    def set_main_agent_pose(self, loc, quat):
        agent_id = 0

        self.root_states[self.robot_actor_idxs[agent_id], 0:3] = torch.Tensor(loc)
        self.root_states[self.robot_actor_idxs[agent_id], 3:7] = torch.Tensor(quat)


        robot_env_ids = self.robot_actor_idxs[agent_id].to(device=self.device)
        if self.cfg.env.add_balls:
            self.root_states[self.object_actor_idxs[agent_id], 0:2] = torch.Tensor(loc[0:2]) + torch.Tensor([2.5, 0.0])
            self.root_states[self.object_actor_idxs[agent_id], 3] = 0.0
            self.root_states[self.object_actor_idxs[agent_id], 3:7] = torch.Tensor(quat)
            print(self.root_states)
            
            object_env_ids = self.object_actor_idxs[agent_id].to(device=self.device)
            all_subject_env_ids = torch.tensor([robot_env_ids, object_env_ids]).to(device=self.device)
        else:
            all_subject_env_ids = torch.tensor([robot_env_ids]).to(device=self.device)
        all_subject_env_ids_int32 = all_subject_env_ids.to(dtype = torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                gymtorch.unwrap_tensor(all_subject_env_ids_int32), len(all_subject_env_ids_int32))

    def _randomize_gravity(self, external_force = None):

        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity

            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)
        
    def _randomize_ball_drag(self):
        if self.cfg.domain_rand.randomize_ball_drag:
            min_drag, max_drag = self.cfg.domain_rand.drag_range
            ball_drags = torch.rand(self.num_envs, dtype=torch.float, device=self.device,
                                    requires_grad=False) * (max_drag - min_drag) + min_drag
            self.ball_drags[:, :]  = ball_drags.unsqueeze(1)

    def _apply_drag_force(self):
        if self.cfg.domain_rand.randomize_ball_drag:
            force_tensor = torch.zeros((self.num_envs, self.num_bodies + 1, 3), dtype=torch.float32, device=self.device)
            force_tensor[:, self.num_bodies, :2] = - self.ball_drags * torch.square(self.ball_lin_vel[:, :2]) * torch.sign(self.ball_lin_vel[:, :2])
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(force_tensor), None, gymapi.GLOBAL_SPACE)


    def get_ground_frictions(self, env_ids):
        # get terrain cell indices
        # optimize later
        positions = self.root_states[env_ids, 0:3]
        terrain_cell_indices = torch.zeros((self.num_envs, 2), device=self.device)
        terrain_cell_indices[:, 0] = torch.clamp(positions[:, 0] / self.terrain.cfg.env_width, 0,
                                                 self.terrain.cfg.num_rows - 1)
        terrain_cell_indices[:, 1] = torch.clamp(positions[:, 1] / self.terrain.cfg.env_length, 0,
                                                 self.terrain.cfg.num_cols - 1)

        # get frictions
        ground_frictions = torch.zeros(self.num_envs, device=self.device)
        ground_frictions[:] = self.terrain_obj.terrain_cell_frictions[
            terrain_cell_indices[:, 0].long(), terrain_cell_indices[:, 1].long()]

        return ground_frictions

    def get_ground_restitutions(self, env_ids):
        # get terrain cell indices
        # optimize later
        positions = self.root_states[env_ids, 0:3]
        terrain_cell_indices = torch.zeros((self.num_envs, 2), device=self.device)
        terrain_cell_indices[:, 0] = torch.clamp(positions[:, 0] / self.terrain.cfg.env_width, 0,
                                                 self.terrain.cfg.num_rows - 1)
        terrain_cell_indices[:, 1] = torch.clamp(positions[:, 1] / self.terrain.cfg.env_length, 0,
                                                 self.terrain.cfg.num_cols - 1)

        # get frictions
        restitutions = torch.zeros(self.num_envs, device=self.device)
        restitutions[:] = self.terrain_obj.terrain_cell_restitutions[
                terrain_cell_indices[:, 0].long(), terrain_cell_indices[:, 1].long()]

        return restitutions

    def get_ground_roughness(self, env_ids):
        # get terrain cell indices
        # optimize later
        positions = self.base_pos
        terrain_cell_indices = torch.zeros((self.num_envs, 1, 2), device=self.device)
        terrain_cell_indices[:, 0, 0] = torch.clamp(positions[:, 0] / self.terrain.cfg.env_width, 0,
                                                       self.terrain.cfg.num_rows - 1)
        terrain_cell_indices[:, 0, 1] = torch.clamp(positions[:, 1] / self.terrain.cfg.env_length, 0,
                                                       self.terrain.cfg.num_cols - 1)

        # get roughnesses
        roughnesses = torch.zeros(self.num_envs, device=self.device)
        roughnesses[:] = self.terrain_obj.terrain_cell_roughnesses[
            terrain_cell_indices[:, 0, 0].long(), terrain_cell_indices[:, 0, 1].long()]

        return roughnesses

    def get_stair_height(self, env_ids):
        points = self.base_pos
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        stair_heights = self.stair_heights_samples[px, py]
        return stair_heights

    def get_stair_run(self, env_ids):
        points = self.base_pos
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        stair_runs = self.stair_runs_samples[px, py]
        return stair_runs

    def get_stair_ori(self, env_ids):
        points = self.base_pos
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        stair_oris = self.stair_oris_samples[px, py]
        return stair_oris

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id, 0]
            props[s].restitution = self.restitutions[env_id, 0]

        return props
    
    def _process_ball_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.ball_friction_coeffs[env_id]
            props[s].restitution = self.ball_restitutions[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_damping = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_friction = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                self.dof_damping[i] = props["damping"][i].item()
                self.dof_friction[i] = props["friction"][i].item()
                
                if self.cfg.asset.default_dof_drive_mode == 0: # mixed
                    if i > 11:
                        props["stiffness"][i] = self.cfg.commands.p_gains_arm[i-12] #kp
                        props["damping"][i] = self.cfg.commands.d_gains_arm[i-12] #kd
                        props['driveMode'] = gymapi.DOF_MODE_POS
                    else:
                        props["stiffness"][i] = 0
                        props["damping"][i] = 0
                        props['driveMode'] = gymapi.DOF_MODE_EFFORT
                elif self.cfg.asset.default_dof_drive_mode == 1:
                    if i > 11 and i < 19:
                        props["stiffness"][i] = self.cfg.commands.p_gains_arm[i-12] #kp
                        props["damping"][i] = self.cfg.commands.d_gains_arm[i-12] #kd
                    else:
                        props["stiffness"][i] = self.cfg.commands.p_gains_legs[i] #kp
                        props["damping"][i] = self.cfg.commands.d_gains_legs[i] #kd
                
                    props['driveMode'] = gymapi.DOF_MODE_POS
                elif self.cfg.asset.default_dof_drive_mode == 3:
                    props["stiffness"][i] = 0
                    props["damping"][i] = 0
                    props['driveMode'] = gymapi.DOF_MODE_EFFORT
                else:
                    raise Exception("Drive mode not found!")
                    

                # soft limits for legs only
                if i < 12 and i not in [2, 5]:
                    m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                    r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                    self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                    self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                # Increase the minimum angle of the calf angle of the front legs
                self.dof_pos_limits[2, 0] -= 0.25
                self.dof_pos_limits[5, 0] -= 0.25
            
        # print("troque lim legs =: ", self.torque_limits[:12])
        # print("troque lim arm =: ", self.torque_limits[12:])

        return props

    def _randomize_rigid_body_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            # self.payloads[env_ids] = -1.0
            self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (max_payload - min_payload) + min_payload
        if cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                            requires_grad=False) * (
                                                         max_com_displacement - min_com_displacement) + min_com_displacement

        if cfg.domain_rand.randomize_friction:
            min_friction, max_friction = cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                          requires_grad=False) * (
                                                       max_friction - min_friction) + min_friction

        if cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                    requires_grad=False) * (
                                                 max_restitution - min_restitution) + min_restitution

        if cfg.env.add_balls:
            if cfg.domain_rand.randomize_ball_restitution:
                min_restitution, max_restitution = cfg.domain_rand.ball_restitution_range
                self.ball_restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                        requires_grad=False) * (
                                                    max_restitution - min_restitution) + min_restitution

            if cfg.domain_rand.randomize_ball_friction:
                min_friction, max_friction = cfg.domain_rand.ball_friction_range
                self.ball_friction_coeffs[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                        requires_grad=False) * (
                                                    max_friction - min_friction) + min_friction

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _randomize_dof_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_strength - min_strength) + min_strength
        if cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     max_offset - min_offset) + min_offset
        if cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kp_factor - min_Kp_factor) + min_Kp_factor
        if cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor
        if cfg.domain_rand.randomize_gripper_force_gains:
            min_kp, max_kp = cfg.domain_rand.gripper_force_kp_range
            min_kd, max_kd = cfg.domain_rand.gripper_force_kd_range
            self.gripper_force_kps[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_kp - min_kp) + min_kp
            if cfg.domain_rand.prop_kd > 0:
                self.gripper_force_kds[env_ids, :] = self.gripper_force_kps[env_ids, :] * cfg.domain_rand.prop_kd
            else:
                self.gripper_force_kds[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                        requires_grad=False).unsqueeze(1) * (
                                                    max_kd - min_kd) + min_kd

    def _process_rigid_body_props(self, props, env_id):
        self.default_body_mass = props[0].mass

        props[0].mass = self.default_body_mass + self.payloads[env_id]
        props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
                                   self.com_displacements[env_id, 2])
        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        # teleport robots to prevent falling off the edge
        self._teleport_robots(torch.arange(self.num_envs, device=self.device), self.cfg)

        # resample commands
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        # print(self.episode_length_buf, sample_interval, env_ids)
        
        self._resample_commands(env_ids)
        # if len(env_ids) > 0: print(self.commands[0, 0:3])
        self._step_contact_targets()
        
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]) - self.heading_offsets
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.heading_commands - heading), -1., 1.)

        # measure terrain heights
        if self.cfg.perception.measure_heights:
            self.measured_heights = self.heightmap_sensor.get_observation()

        # push robots
        self._push_robots(torch.arange(self.num_envs, device=self.device), self.cfg)

        # # push gripper 
        # self._push_gripper(torch.arange(self.num_envs, device=self.device), self.cfg)      

        # # push robot base 
        # self._push_robot_base(torch.arange(self.num_envs, device=self.device), self.cfg) 

        # randomize dof properties
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._randomize_dof_props(env_ids, self.cfg)

        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if self.common_step_counter % int(self.cfg.domain_rand.ball_drag_rand_interval) == 0:
            self._randomize_ball_drag()
        if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(
                self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity(torch.tensor([0, 0, 0]))
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids, self.cfg)
            self.refresh_actor_rigid_shape_props(env_ids, self.cfg)

    def _resample_commands(self, env_ids):

        if len(env_ids) == 0: return

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.cfg.env.max_episode_length, timesteps)


        # update curricula based on terminated environment bins and categories
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            # env_ids_in_category = env_ids[env_ids_in_category]
            env_ids_in_category = env_ids

            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                
                    
                # if self.cfg.commands.control_ee_ori:
                #     curriculum.update(old_bins, task_rewards, success_thresholds,
                #                     local_range=np.array(
                #                         [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                #                             0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]))
                # elif self.cfg.commands.control_ee_ori_only_yaw:
                #     curriculum.update(old_bins, task_rewards, success_thresholds,
                #                     local_range=np.array(
                #                         [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                #                             0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]))
                # else:
                curriculum.update(old_bins, task_rewards, success_thresholds,
                            local_range=np.array(
                                [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                                0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]))
        # assign resampled environments to new categories
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1. / len(self.category_names)
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                      random_env_floats < probability_per_category * (i + 1))] for i in
                            range(len(self.category_names))]

        # sample from new category curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :self.cfg.commands.num_commands]).to(
                self.device)

            # print(self.commands[0, 0:3])



        if self.cfg.commands.num_commands > 5:
            if self.cfg.commands.gaitwise_curricula:
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    if category == "pronk":  # pronking
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    elif category == "trot":  # trotting
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":  # pacing
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":  # bounding
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

            elif self.cfg.commands.exclusive_phase_offset:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                trotting_envs = env_ids[random_env_floats < 0.34]
                pacing_envs = env_ids[torch.logical_and(0.34 <= random_env_floats, random_env_floats < 0.67)]
                bounding_envs = env_ids[0.67 <= random_env_floats]
                self.commands[pacing_envs, 5] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[trotting_envs, 6] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 7] = 0

            elif self.cfg.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[torch.logical_and(0.25 <= random_env_floats, random_env_floats < 0.50)]
                pacing_envs = env_ids[torch.logical_and(0.50 <= random_env_floats, random_env_floats < 0.75)]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                self.commands[pronking_envs, 5] = (self.commands[pronking_envs, 5] / 2 - 0.25) % 1
                self.commands[pronking_envs, 6] = (self.commands[pronking_envs, 6] / 2 - 0.25) % 1
                self.commands[pronking_envs, 7] = (self.commands[pronking_envs, 7] / 2 - 0.25) % 1
                self.commands[trotting_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 5] = 0
                self.commands[pacing_envs, 7] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 5] = self.commands[trotting_envs, 5] / 2 + 0.25
                self.commands[pacing_envs, 6] = self.commands[pacing_envs, 6] / 2 + 0.25
                self.commands[bounding_envs, 7] = self.commands[bounding_envs, 7] / 2 + 0.25

            if self.cfg.commands.binary_phases:
                self.commands[env_ids, 5] = (torch.round(2 * self.commands[env_ids, 5])) / 2.0 % 1
                self.commands[env_ids, 6] = (torch.round(2 * self.commands[env_ids, 6])) / 2.0 % 1
                self.commands[env_ids, 7] = (torch.round(2 * self.commands[env_ids, 7])) / 2.0 % 1

        # # setting the smaller commands to zero
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.
            
        # respect command constriction
        self._update_command_ranges(env_ids)
            
        # heading commands
        if self.cfg.commands.heading_command:
            self.heading_commands[env_ids] = torch_rand_float(self.cfg.commands.heading[0],
                                                         self.cfg.commands.heading[1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
            
        if self.cfg.commands.gait_phase_cmd_range[0] == self.cfg.commands.gait_phase_cmd_range[1]:
            self.commands[env_ids_in_category, 5] = self.cfg.commands.gait_phase_cmd_range[0]
        if self.cfg.commands.gait_offset_cmd_range[0] == self.cfg.commands.gait_offset_cmd_range[1]:
            self.commands[env_ids_in_category, 6] = self.cfg.commands.gait_offset_cmd_range[0]
        if self.cfg.commands.gait_bound_cmd_range[0] == self.cfg.commands.gait_bound_cmd_range[1]:
            self.commands[env_ids_in_category, 7] = self.cfg.commands.gait_bound_cmd_range[0]

    def _step_contact_targets(self):
        # if self.cfg.env.observe_gait_commands:
        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        bounds = self.commands[:, 7]
        durations = self.commands[:, 8]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        if self.cfg.commands.pacing_offset:
            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + phases]
        else:
            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + bounds,
                            self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                        0.5 / (1 - durations[swing_idxs]))

        # if self.cfg.commands.durations_warp_clock_inputs:

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
        self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
        self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
        self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

        self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
        self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
        self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
        self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

        # von mises distribution
        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        env_ids = torch.arange(self.num_envs, device=self.device)
        static_env_ids = env_ids[torch.logical_and(torch.logical_and(torch.abs(self.commands[:, 0]) < 0.2, torch.abs(self.commands[:, 1]) < 0.2), torch.abs(self.commands[:, 2]) < 0.2)]

        self.desired_contact_states[static_env_ids, 0] = 1.0
        self.desired_contact_states[static_env_ids, 1] = 1.0
        self.desired_contact_states[static_env_ids, 2] = 1.0
        self.desired_contact_states[static_env_ids, 3] = 1.0

        self.clock_inputs[static_env_ids, 0] = 1.0
        self.clock_inputs[static_env_ids, 1] = 1.0
        self.clock_inputs[static_env_ids, 2] = 1.0
        self.clock_inputs[static_env_ids, 3] = 1.0



        # print("self.foot_indices", self.foot_indices, " self.desired_contact_states", self.desired_contact_states, " self.clock_inputs", self.clock_inputs)

        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = torch.zeros(self.num_envs, self.num_dof, device = self.device)
        actions_scaled[:, :self.num_actions] = actions[:, :self.num_actions] * self.cfg.control.action_scale
        if self.num_actions >= 12:
            actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range
            actions_scaled[:, 12:] *= self.cfg.control.arm_scale_reduction

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            actions_post = self.lag_buffer[0]
        else:
            actions_post = actions_scaled
            
        if self.cfg.control.control_type == "P":
            self.joint_pos_target = actions_post + self.default_dof_pos
        elif self.cfg.control.control_type == "dP":
            self.joint_pos_target += actions_post * self.dt / self.cfg.control.decimation

        #self.joint_pos_target[:, 12] = -70/180*3.14
        # Force gripper state to 0
        if self.cfg.commands.teleop_occulus:
            self.joint_pos_target[:, 17] = self.teleop_joint6_value
            self.joint_pos_target[:, 18] = self.teleop_gripper_value
        # else:
        #     self.joint_pos_target[:, 18] = 0.0

        if self.cfg.commands.sample_feasible_commands:
            self.joint_pos_target[:, :self.num_actuated_dof] = torch.tensor(self.target_joint_values, device=self.device)

        

        # actuator model
        dof_pos_error = (self.dof_pos - self.joint_pos_target)

        # account for full joint rotation
        # dof_pos_error = wrap_to_pi(dof_pos_error)
        # print(dof_pos_error, self.dof_pos, self.joint_pos_target)
        dof_vel = self.dof_vel
        
        # self.joint_pos_target[:, 18] = -0.5
        # print(self.joint_pos_target[:, 16:], self.dof_pos[:, 16:], self.p_gains[16:], self.d_gains[16:])
        
        # print(self.p_gains, self.d_gains)
        
        frictions = 0 #(self.d_gains * 0.5)
        # input((self.p_gains, self.d_gains))
        ideal_torques = -self.p_gains * self.Kp_factors * dof_pos_error - (self.d_gains * self.Kd_factors + frictions) * dof_vel
        stall_torque = self.torque_limits
        max_vel = self.dof_vel_limits
        vel_torque_max = stall_torque * (1 - torch.clip(torch.abs(dof_vel / max_vel), max=1))
        # vel_torque_max[vel_torque_max>stall_torque] = stall_torque
        
        vel_torque_min = -stall_torque * (1 - torch.clip(torch.abs(dof_vel / max_vel), max=1))
        # vel_torque_min[vel_torque_min<=-stall_torque] = -stall_torque
        
        clipped_torques = ideal_torques.clone()
        
        # clipped_torques[dof_vel > 0] = vel_torque_min[dof_vel > 0] * torch.tanh(ideal_torques[dof_vel > 0] / vel_torque_min[dof_vel > 0])
        # clipped_torques[dof_vel <= 0] = vel_torque_max[dof_vel <= 0] * torch.tanh(ideal_torques[dof_vel <= 0] / vel_torque_max[dof_vel <= 0])
        clipped_torques = clipped_torques.clamp(vel_torque_min, vel_torque_max)    
        self.torques = clipped_torques

        # print(ideal_torques, clipped_torques)

        # idealized torques
        # self.torques = self.p_gains * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.dof_vel #- self.dof_damping*self.dof_vel - self.dof_friction*torch.sign(self.dof_vel)
        
        

    def _reset_dofs(self, env_ids, cfg):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos #+ torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids, 12:18] += torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)
        self.joint_pos_target[env_ids] = self.default_dof_pos
        # if len(self.default_dof_pos_RL):
        #     self.dof_pos[env_ids, :12] = self.default_dof_pos_RL * torch_rand_float(0.9, 1.1, (len(env_ids), 12), device=self.device)

        # if self.cfg.env.force_control_init_poses:
        #     self.dof_pos[env_ids, :self.num_actuated_dof] = self.force_control_dof_pos_init[env_ids, :self.num_actuated_dof]

        self.dof_vel[env_ids] = 0.

        all_subject_env_ids = self.robot_actor_idxs[env_ids].to(device=self.device)
        if self.cfg.env.add_balls and self.cfg.ball.asset == "door": 
            object_env_ids = self.object_actor_idxs[env_ids].to(device=self.device)
            all_subject_env_ids = torch.cat((all_subject_env_ids, object_env_ids))
        all_subject_env_ids_int32 = all_subject_env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_subject_env_ids_int32), len(all_subject_env_ids_int32))

    def _reset_root_states(self, env_ids, cfg):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        robot_env_ids = self.robot_actor_idxs[env_ids].to(device=self.device)

        ### Reset robots
        # base origins
        if self.custom_origins:
            self.root_states[robot_env_ids] = self.base_init_state
            self.root_states[robot_env_ids, :3] += self.env_origins[env_ids]
            self.root_states[robot_env_ids, 0:1] += torch_rand_float(-cfg.terrain.x_init_range,
                                                               cfg.terrain.x_init_range, (len(robot_env_ids), 1),
                                                               device=self.device)
            self.root_states[robot_env_ids, 1:2] += torch_rand_float(-cfg.terrain.y_init_range,
                                                               cfg.terrain.y_init_range, (len(robot_env_ids), 1),
                                                               device=self.device)
            self.root_states[robot_env_ids, 0] += cfg.terrain.x_init_offset
            self.root_states[robot_env_ids, 1] += cfg.terrain.y_init_offset
        else:
            self.root_states[robot_env_ids] = self.base_init_state
            self.root_states[robot_env_ids, :3] += self.env_origins[env_ids]

        # base yaws
        # init_yaws = torch_rand_float(-cfg.terrain.yaw_init_range,
        #                              cfg.terrain.yaw_init_range, (len(robot_env_ids), 1),
        #                              device=self.device)
        # print(init_yaws.shape, len(robot_env_ids))
        # quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        # print(quat.shape)
        # self.root_states[robot_env_ids, 3:7] = quat
        
        random_yaw_angle = 2*(torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                     requires_grad=False)-0.5)*torch.tensor([0, 0, cfg.terrain.yaw_init_range], device=self.device)
        self.root_states[robot_env_ids,3:7] = quat_from_euler_xyz(random_yaw_angle[:,0], random_yaw_angle[:,1], random_yaw_angle[:,2])
            

        if self.cfg.env.offset_yaw_obs:
            self.heading_offsets[env_ids] = torch_rand_float(-cfg.terrain.yaw_init_range,
                                     cfg.terrain.yaw_init_range, (len(env_ids), 1),
                                     device=self.device).flatten()

        # base velocities
        self.root_states[robot_env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(robot_env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel


        ### Reset objects
        if self.cfg.env.add_balls:
            object_env_ids = self.object_actor_idxs[env_ids].to(device=self.device)
            # base origins
            self.root_states[object_env_ids] = self.object_init_state
            self.root_states[object_env_ids, :3] += self.env_origins[env_ids]

            self.root_states[object_env_ids,0:3] += 2*(torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                     requires_grad=False)-0.5) * torch.tensor(cfg.ball.init_pos_range,device=self.device,
                                                     requires_grad=False)
            self.root_states[object_env_ids,7:10] += 2*(torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                     requires_grad=False)-0.5) * torch.tensor(cfg.ball.init_vel_range,device=self.device,
                                                     requires_grad=False)
                                                     
        if self.cfg.env.force_control_init_poses:
            self.root_states[robot_env_ids, 2] = self.force_control_base_pos_init[robot_env_ids, 2]
            self.root_states[robot_env_ids, 3:7] = self.force_control_base_pos_init[robot_env_ids, 3:7]
        
        # apply reset states
        all_subject_env_ids = robot_env_ids
        if self.cfg.env.add_balls: 
            all_subject_env_ids = torch.cat((robot_env_ids, object_env_ids))
        all_subject_env_ids_int32 = all_subject_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(all_subject_env_ids_int32), len(all_subject_env_ids_int32))

        if cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []
        
    def _push_robots(self, env_ids, cfg):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        if cfg.domain_rand.push_robots:
            env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_interval) == 0]

            max_vel = cfg.domain_rand.max_push_vel_xy
            self.root_states[self.robot_actor_idxs[env_ids], 7:9] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2),
                                                              device=self.device)  # lin vel x/y
            # if len(env_ids): print("PUSH BASE ", self.root_states[self.robot_actor_idxs[env_ids], 7:9])
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    
    def override_ext_forces(self, Fx=0, Fy=0, Fz=0):
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz
        self.forces_deactivated = True

    def _push_gripper(self, env_ids_all, cfg):
        """ Randomly pushes the gripper stators. Emulates an impulse by setting a randomized gripper stator velocity.
        """

        force_control_envs = self.force_or_position_control == 1
        position_control_envs = self.force_or_position_control == 0
        
        # FORCE CONTROLLED ENVS
        new_selected_env_ids = env_ids_all[(self.episode_length_buf % self.push_interval[:, 0]) == 0]
        
        # print('newselect', new_selected_env_ids)
        # self.push_force[force_control_envs] = 0.0
        
        # Define force and duration for the push 
        if new_selected_env_ids.nelement() > 0:
            self.freed_envs[new_selected_env_ids] = torch.rand(len(new_selected_env_ids), dtype=torch.float, device=self.device,
                                        requires_grad=False) > self.cfg.domain_rand.gripper_forced_prob
            min_force = cfg.domain_rand.max_push_force_xyz_gripper[0]
            max_force = cfg.domain_rand.max_push_force_xyz_gripper[1]

            self.force_target[new_selected_env_ids, 0] = torch_rand_float(min_force, max_force, (len(new_selected_env_ids), 1), device=self.device).view(len(new_selected_env_ids))
            self.force_target[new_selected_env_ids, 1] = torch_rand_float(min_force, max_force, (len(new_selected_env_ids), 1), device=self.device).view(len(new_selected_env_ids))
            self.force_target[new_selected_env_ids, 2] = torch_rand_float(min_force, max_force, (len(new_selected_env_ids), 1), device=self.device).view(len(new_selected_env_ids))
            push_duration = torch_rand_float(cfg.domain_rand.push_duration_gripper_min, cfg.domain_rand.push_duration_gripper_max, (len(new_selected_env_ids), 1), device=self.device).view(len(new_selected_env_ids)) # 4.0/self.dt
            self.push_end_time[new_selected_env_ids] = self.episode_length_buf[new_selected_env_ids] + push_duration
            self.push_duration[new_selected_env_ids] = push_duration

            self.selected_env_ids[new_selected_env_ids] = 1
            
        # Get ids of all envs to apply a force to 
        if self.episode_length_buf[self.selected_env_ids == 1].nelement() > 0:
            subset_env_ids_selected = env_ids_all[self.selected_env_ids == 1]

            # Step 1: apply force from 0 to force_target
            env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids == 1] < (self.push_end_time[self.selected_env_ids == 1]).type(torch.int32)]
            # print(env_ids_apply_push_step1)
            if env_ids_apply_push_step1.nelement() > 0:
                push_duration_reshaped = self.push_duration[env_ids_apply_push_step1].unsqueeze(-1)
                # world frame
                self.current_Fxyz_cmd[env_ids_apply_push_step1, :3] = (self.force_target[env_ids_apply_push_step1, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1) - (self.push_end_time[env_ids_apply_push_step1].unsqueeze(-1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                # World frame
                self.commands[env_ids_apply_push_step1, INDEX_EE_FORCE_X] = self.current_Fxyz_cmd[env_ids_apply_push_step1, 0] #torch.norm(self.current_Fxyz_cmd[env_ids_apply_push_step1, :2], dim=1)
                self.commands[env_ids_apply_push_step1, INDEX_EE_FORCE_Y] = self.current_Fxyz_cmd[env_ids_apply_push_step1, 1] #torch.atan2(self.current_Fxyz_cmd[env_ids_apply_push_step1, 1], self.current_Fxyz_cmd[env_ids_apply_push_step1, 0])
                self.commands[env_ids_apply_push_step1, INDEX_EE_FORCE_Z] = self.current_Fxyz_cmd[env_ids_apply_push_step1, 2]
            
                # print(self.commands[env_ids_apply_push_step1, INDEX_EE_FORCE_MAGNITUDE:INDEX_EE_FORCE_Z+1])
                
            # Step 2: apply force from force_target back to 0
            env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids == 1] > (self.push_end_time[self.selected_env_ids == 1] + self.cfg.commands.settling_nb_it_force_gripper).type(torch.int32)]
            if env_ids_apply_push_step2.nelement() > 0:
                push_duration_reshaped = self.push_duration[env_ids_apply_push_step2].unsqueeze(-1)
                self.current_Fxyz_cmd[env_ids_apply_push_step2, :3] = self.force_target[env_ids_apply_push_step2, :3] - (self.force_target[env_ids_apply_push_step2, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1) - (self.push_end_time[env_ids_apply_push_step2].unsqueeze(-1)+self.cfg.commands.settling_nb_it_force_gripper), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                # print("current target: ", self.current_Fxyz_cmd[env_ids_apply_push_step2, :3])
            
                # World frame
                self.commands[env_ids_apply_push_step2, INDEX_EE_FORCE_X] = self.current_Fxyz_cmd[env_ids_apply_push_step2, 0] #torch.norm(self.current_Fxyz_cmd[env_ids_apply_push_step2, :2], dim=1)
                self.commands[env_ids_apply_push_step2, INDEX_EE_FORCE_Y] = self.current_Fxyz_cmd[env_ids_apply_push_step2, 1] #torch.atan2(self.current_Fxyz_cmd[env_ids_apply_push_step2, 1], self.current_Fxyz_cmd[env_ids_apply_push_step2, 0])
                self.commands[env_ids_apply_push_step2, INDEX_EE_FORCE_Z] = self.current_Fxyz_cmd[env_ids_apply_push_step2, 2]
                
                # print(self.commands[env_ids_apply_push_step2, INDEX_EE_FORCE_MAGNITUDE:INDEX_EE_FORCE_Z+1])
            
            # Reset the tensors
            env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids == 1] >= (2*self.push_end_time[self.selected_env_ids == 1] + self.cfg.commands.settling_nb_it_force_gripper).type(torch.int32)]                                        
            if env_ids_to_reset.nelement() > 0:
                self.selected_env_ids[env_ids_to_reset] = 0
                self.force_target[env_ids_to_reset, :3] = 0.
                self.current_Fxyz_cmd[env_ids_to_reset, :3] = 0.
                self.push_end_time[env_ids_to_reset] = 0.
                self.push_duration[env_ids_to_reset] = 0.
                self.commands[env_ids_to_reset, INDEX_EE_FORCE_X] = 0.0
                self.commands[env_ids_to_reset, INDEX_EE_FORCE_Y] = 0.0

                self.commands[env_ids_to_reset, INDEX_EE_FORCE_Z] = 0.0

                self.push_interval[env_ids_to_reset, 0] = torch.randint(int(cfg.domain_rand.push_interval_gripper_min), int(cfg.domain_rand.push_interval_gripper_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                
        self.selected_env_ids[self.freed_envs] = 0
        self.force_target[self.freed_envs, :3] = 0.
        self.current_Fxyz_cmd[self.freed_envs, :3] = 0.
        self.push_end_time[self.freed_envs] = 0.
        self.push_duration[self.freed_envs] = 0. 
        self.commands[self.freed_envs, INDEX_EE_FORCE_X] = 0.0
        self.commands[self.freed_envs, INDEX_EE_FORCE_Y] = 0.0
        self.commands[self.freed_envs, INDEX_EE_FORCE_Z] = 0.0

        # Commands in spherical coordinates in the arm base frame 
        radius_cmd = self.commands[:, INDEX_EE_POS_RADIUS_CMD].view(self.num_envs, 1) 
        pitch_cmd = self.commands[:, INDEX_EE_POS_PITCH_CMD].view(self.num_envs, 1) 
        yaw_cmd = self.commands[:, INDEX_EE_POS_YAW_CMD].view(self.num_envs, 1) 

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
        base_quat_world = self.base_quat.view(self.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        # Make the commands roll and pitch independent 
        base_rpy_world[:, 0] = 0.0
        base_rpy_world[:, 1] = 0.0
        base_quat_world_indep = quat_from_euler_xyz(base_rpy_world[:, 0], base_rpy_world[:, 1], base_rpy_world[:, 2]).view(self.num_envs,4)

        # Make the commands independent from base height 
        x_base_pos_world = self.base_pos[:, 0].view(self.num_envs, 1) 
        y_base_pos_world = self.base_pos[:, 1].view(self.num_envs, 1) 
        z_base_pos_world = torch.ones_like(self.base_pos[:, 2].view(self.num_envs, 1))*DEFAULT_BASE_HEIGHT
        base_position_world = torch.cat((x_base_pos_world, y_base_pos_world, z_base_pos_world), dim=1)

        # Command in cartesian coordinates in world frame 
        self.ee_position_cmd_world = quat_rotate_inverse(quat_conjugate(base_quat_world_indep), ee_position_cmd_base) + base_position_world

        # Get current ee position in world frame 
        ee_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[0], "gripperStator")
        self.ee_pos_world = self.rigid_body_state.view(self.num_envs, -1, 13)[:,ee_idx,0:3].view(self.num_envs,3)

        # ee_position_error = torch.sum(torch.abs(ee_position_cmd_world - ee_pos_world), dim=1)
        # self.forces[:, self.gripper_stator_index, :3] = torch.clamp(((self.ee_position_cmd_world - self.ee_pos_world) * self.cfg.domain_rand.gripper_force_kp*3 +  (0 - self.gripper_velocity) * self.cfg.domain_rand.gripper_force_kd), self.cfg.domain_rand.max_push_force_xyz_gripper_freed[0], self.cfg.domain_rand.max_push_force_xyz_gripper_freed[1])
        self.forces[:, self.gripper_stator_index, :3] = torch.clamp(((self.ee_position_cmd_world - self.ee_pos_world) * self.gripper_force_kps +  (0 - self.gripper_velocity) * self.gripper_force_kds), self.cfg.domain_rand.max_push_force_xyz_gripper_freed[0], self.cfg.domain_rand.max_push_force_xyz_gripper_freed[1])
        self.forces[self.freed_envs, self.gripper_stator_index, :3] = 0
        
        # don't force the position envs
        self.forces[position_control_envs, self.gripper_stator_index, :3] = 0
        self.commands[position_control_envs, INDEX_EE_FORCE_X] = 0.0
        self.commands[position_control_envs, INDEX_EE_FORCE_Y] = 0.0
        self.commands[position_control_envs, INDEX_EE_FORCE_Z] = 0.0
        
        # compute the error
        force_magn_meas = torch.norm(self.forces[:, self.gripper_stator_index, :2], dim=1).view(self.num_envs)
        force_magn_target = torch.norm(self.commands[:, INDEX_EE_FORCE_X:INDEX_EE_FORCE_Y+1], dim=1).view(self.num_envs) 
        force_magn_error_xy = torch.abs(force_magn_meas - force_magn_target).view(self.num_envs)

        force_magn_meas = (self.forces[:, self.gripper_stator_index, 2]).view(self.num_envs, 1)
        force_magn_cmd = (self.commands[:, INDEX_EE_FORCE_Z]).view(self.num_envs, 1)
        force_magn_error_z = torch.abs(force_magn_meas - force_magn_cmd).view(self.num_envs)


        self.gripper_xy_force_tracking_error_buf = force_magn_error_xy
        self.gripper_z_force_tracking_error_buf = force_magn_error_z
        
        if self.forces_deactivated:
            self.forces[:, self.gripper_stator_index, 0] = self.Fx
            self.forces[:, self.gripper_stator_index, 1] = self.Fy
            self.forces[:, self.gripper_stator_index, 2] = self.Fz

        # print(self.commands[:, 12:15])

        
    def _push_robot_base(self, env_ids, cfg):
        """ 
            Randomly pushes the robot base.
        """

        if cfg.domain_rand.push_robot_base:
            
            # Get new envs to which a force is applied 
            new_selected_env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_interval_base) == 0]

            # Define force and duration for the push 
            min_force = cfg.domain_rand.max_push_vel_xyz_robot[0]
            max_force = cfg.domain_rand.max_push_vel_xyz_robot[1]
            self.push_force_robot[new_selected_env_ids, 0] = torch_rand_float(min_force, max_force, (len(new_selected_env_ids), 1), device=self.device).view(len(new_selected_env_ids))
            self.push_force_robot[new_selected_env_ids, 1] = torch_rand_float(min_force, max_force, (len(new_selected_env_ids), 1), device=self.device).view(len(new_selected_env_ids))
            self.push_force_robot[new_selected_env_ids, 2] = torch_rand_float(min_force, max_force, (len(new_selected_env_ids), 1), device=self.device).view(len(new_selected_env_ids))
            push_duration = torch_rand_float(cfg.domain_rand.push_duration_robot_min, cfg.domain_rand.push_duration_robot_max, (len(new_selected_env_ids), 1), device=self.device).view(len(new_selected_env_ids)) # 4.0/self.dt
            self.push_end_time_robot[new_selected_env_ids] = self.episode_length_buf[new_selected_env_ids] + push_duration

            # if new_selected_env_ids.nelement() > 0:
            #     print(self.episode_length_buf[new_selected_env_ids])
            #     print(" New Force ", self.push_force_robot[new_selected_env_ids] ," applied for envs: ", new_selected_env_ids, ' for ', (push_duration)*self.dt, ' seconds')
            #     print(" ")

            # Update the tensor saving all envs to apply a push to 
            self.selected_env_ids_robot[new_selected_env_ids] = 1

            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_robot == 1].nelement() > 0:
                subset_env_ids_selected = env_ids[self.selected_env_ids_robot == 1]
                env_ids_apply_push = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_robot == 1] <= (self.push_end_time_robot[self.selected_env_ids_robot == 1]).type(torch.int32)]

                # Apply force
                self.forces[env_ids_apply_push, self.robot_base_index, :3] = self.push_force_robot[env_ids_apply_push]

                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_robot == 1] > (self.push_end_time_robot[self.selected_env_ids_robot == 1]).type(torch.int32)]
                self.selected_env_ids_robot[env_ids_to_reset] = 0
                self.push_force_robot[env_ids_to_reset, :3] = 0.
                self.push_end_time_robot[env_ids_to_reset] = 0.
                self.forces[env_ids_to_reset, self.robot_base_index, :3] = 0. 

                # if env_ids_to_reset.nelement() > 0:
                #     print("end ", self.episode_length_buf[env_ids_to_reset])

                        
    
            
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        robot_env_ids = self.robot_actor_idxs[env_ids]
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[robot_env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.cfg.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        # move_down = (distance < torch.norm(self.commands[env_ids, :2],
        #                                    dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        move_down = (self.path_distance[env_ids] < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random xfone
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      low=self.min_terrain_level,
                                                                      high=self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              self.min_terrain_level))  # (the minumum level is zero)
        # self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_origins[env_ids] = self.cfg.terrain.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
    def _update_path_distance(self):
        path_distance_interval = 25
        env_ids = (self.episode_length_buf % path_distance_interval == 0).nonzero(as_tuple=False).flatten()
        distance_traversed = torch.linalg.norm(self.base_pos[env_ids, 0:2] - self.past_base_pos[env_ids, 0:2], dim=1)
        self.path_distance[env_ids] += distance_traversed
        self.past_base_pos[env_ids] = self.base_pos.clone()[env_ids]

    def _update_command_ranges(self, env_ids):
        constrict_indices = self.cfg.rewards.constrict_indices
        constrict_ranges = self.cfg.rewards.constrict_ranges

        if self.cfg.rewards.constrict and self.common_step_counter >= self.cfg.rewards.constrict_after:
            for idx, range in zip(constrict_indices, constrict_ranges):
                self.commands[env_ids, idx] = range[0]

    def _teleport_robots(self, env_ids, cfg):
        """ Teleports any robots that are too close to the edge to the other side
        """
        robot_env_ids = self.robot_actor_idxs[env_ids].to(device=self.device)
        # object_env_ids = self.object_actor_idxs[env_ids].to(device=self.device)
        if cfg.terrain.teleport_robots:
            thresh = cfg.terrain.teleport_thresh

            x_offset = int(cfg.terrain.x_offset * cfg.terrain.horizontal_scale)

            low_x_ids = robot_env_ids[self.root_states[robot_env_ids, 0] < thresh + x_offset]
            self.root_states[low_x_ids, 0] += cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            high_x_ids = robot_env_ids[
                self.root_states[robot_env_ids, 0] > cfg.terrain.terrain_length * cfg.terrain.num_rows - thresh + x_offset]
            self.root_states[high_x_ids, 0] -= cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            low_y_ids = robot_env_ids[self.root_states[robot_env_ids, 1] < thresh]
            self.root_states[low_y_ids, 1] += cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            high_y_ids = robot_env_ids[
                self.root_states[robot_env_ids, 1] > cfg.terrain.terrain_width * cfg.terrain.num_cols - thresh]
            self.root_states[high_y_ids, 1] -= cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        # torques = self.gym.acquire_dof_force_tensor(self.sim)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # create some wrapper tensors for different slices
        # self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.base_pos = self.root_states[self.robot_actor_idxs, 0:3]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[self.robot_actor_idxs, 3:7]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,13)[:,0:self.num_bodies, :]#.contiguous().view(self.num_envs*self.num_bodies,13)
        self.rigid_body_state_object = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,13)[:,self.num_bodies:self.num_bodies + self.num_object_bodies, :]#.contiguous().view(self.num_envs*self.num_bodies,13)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, -1, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, -1, 13)[:, self.feet_indices,
                              0:3]
        self.gripper_position = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_stator_index, 0:3]
        self.gripper_velocity = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.gripper_stator_index, 7:10]
        self.prev_base_pos = self.base_pos.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]

        # self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :].view(self.num_envs, -1,
        #                                                                     3)  # shape: num_envs, num_bodies, xyz axis
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,3)[:,0:self.num_bodies, :]

        if self.cfg.env.add_balls:
            self.object_pos_world_frame = self.root_states[self.object_actor_idxs, 0:3]
            # if self.cfg.ball.asset == "door":
            #     handle_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.object_actor_handles[0], "handle") - self.num_bodies
            #     handle_pos_global = self.rigid_body_state_object.view(self.num_envs, -1, 13)[:,handle_idx,0:3].view(self.num_envs,3)
            #     robot_object_vec = handle_pos_global - self.base_pos
            # else:
            #     robot_object_vec = self.root_states[self.object_actor_idxs, 0:3] - self.base_pos
            robot_object_vec = self.asset.get_local_pos()
            self.object_local_pos = quat_rotate_inverse(self.base_quat, robot_object_vec)
            self.object_local_pos[:, 2] = 0.0*torch.ones(self.num_envs, dtype=torch.float,
                                       device=self.device, requires_grad=False)

            self.last_object_local_pos = torch.clone(self.object_local_pos)
            self.object_lin_vel = self.asset.get_lin_vel()
            self.object_ang_vel = self.asset.get_ang_vel()

        
        if self.cfg.env.force_control_init_poses:
            try:
                data = np.load(MINI_GYM_ROOT_DIR + '/b1_gym/envs/base/wbc_data_isaac.npz')
                self.force_control_dof_pos_init = torch.tensor(data['dof_pos'], dtype=torch.float, device=self.device)
                self.force_control_base_pos_init = torch.tensor(data['base_pos'], dtype=torch.float, device=self.device)
            except:
                print("WARNING, something went wrong when downloading the init states for the force controller")
                       
         

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        # if self.cfg.perception.measure_heights:
        self.height_points = self._init_height_points(torch.arange(self.num_envs, device=self.device), self.cfg)
        self.measured_heights = 0

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.save_rew_feet_contact = torch.zeros(1, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.save_rew_torque = torch.zeros(1, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.save_ee_force_magnitude = torch.zeros(1, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.save_ee_force_z = torch.zeros(1, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.save_ee_force_direction_angle = torch.zeros(1, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        # self.dof_arm_vel_in_decimation = torch.zeros(self.cfg.control.decimation, self.num_envs, 7 , dtype=torch.float, device=self.device,
        #                            requires_grad=False)
        # self.read_dof_vel = torch.zeros(self.num_envs, 7, dtype=torch.float, device=self.device,
        #                            requires_grad=False)
        # self.my_arm_dof_vel = torch.zeros(self.cfg.control.decimation, self.num_envs, 7 , dtype=torch.float, device=self.device,
        #                            requires_grad=False)
        # self.last_dof_pos = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
        #                            requires_grad=False)
        
        self.forces = torch.zeros(self.num_envs, self.total_rigid_body_num, 3, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                                      device=self.device,
                                                      requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[self.robot_actor_idxs, 7:13])
        self.path_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.past_base_pos = self.base_pos.clone()


        self.commands_value = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.commands = torch.zeros_like(self.commands_value)  # x vel, y vel, yaw vel, heading
        self.heading_commands = torch.zeros(self.num_envs, dtype=torch.float,
                                          device=self.device, requires_grad=False)  # heading
        self.heading_offsets = torch.zeros(self.num_envs, dtype=torch.float,
                                            device=self.device, requires_grad=False)  # heading offset
        
        # Push gripper 
        self.selected_env_ids = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.push_interval = torch.randint(int(self.cfg.domain_rand.push_interval_gripper_min), int(self.cfg.domain_rand.push_interval_gripper_max), (self.num_envs, 1), device=self.device, requires_grad=False)
        # print("push interval: ", self.push_interval, " range; ", int(self.cfg.domain_rand.push_interval_gripper_min), "-", self.cfg.domain_rand.push_interval_gripper_max)
        self.push_end_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_duration = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_target = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_forces_eval_time = torch.zeros(1, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_Fxyz_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.push_force = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_force_kps = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_force_kds = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Push robot 
        self.selected_env_ids_robot = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False)
        self.push_end_time_robot = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_force_robot = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self._resample_force_or_position_control(torch.arange(self.num_envs, device=self.device))

        # if self.cfg.commands.inverse_IK_door_opening:
            
            # self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
            #                                     self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
            #                                     self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
            #                                     self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
            #                                     self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
            #                                     self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
            #                                     self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd, 
            #                                     self.obs_scales.end_effector_pos_x_cmd, self.obs_scales.end_effector_pos_y_cmd,
            #                                     self.obs_scales.end_effector_pos_z_cmd, self.obs_scales.end_effector_roll_cmd,
            #                                     self.obs_scales.end_effector_pitch_cmd, self.obs_scales.end_effector_yaw_cmd,
            #                                     self.obs_scales.end_effector_gripper_cmd],
            #                                     device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]

            # if self.cfg.commands.control_ee_ori:
            #     self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
            #                                         self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
            #                                         self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
            #                                         self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
            #                                         self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
            #                                         self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
            #                                         self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd, 
            #                                         self.obs_scales.ee_sphe_radius_cmd, self.obs_scales.ee_sphe_pitch_cmd,
            #                                         self.obs_scales.ee_sphe_yaw_cmd, self.obs_scales.ee_timing_cmd, 
            #                                         self.obs_scales.end_effector_roll_cmd, self.obs_scales.end_effector_pitch_cmd, 
            #                                         self.obs_scales.end_effector_yaw_cmd],
            #                                         device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]

            # elif self.cfg.commands.control_ee_ori_only_yaw:
            #     self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
            #                                         self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
            #                                         self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
            #                                         self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
            #                                         self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
            #                                         self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
            #                                         self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd, 
            #                                         self.obs_scales.ee_sphe_radius_cmd, self.obs_scales.ee_sphe_pitch_cmd,
            #                                         self.obs_scales.ee_sphe_yaw_cmd, self.obs_scales.ee_timing_cmd, 
            #                                         self.obs_scales.end_effector_yaw_cmd],
            #                                         device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]
            # else:
            #   if self.cfg.commands.control_ee_ori:
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                            self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
                                            self.obs_scales.body_roll_cmd, self.obs_scales.ee_force_magnitude,
                                            self.obs_scales.ee_force_magnitude, self.obs_scales.ee_force_z,
                                            self.obs_scales.ee_sphe_radius_cmd, self.obs_scales.ee_sphe_pitch_cmd,
                                            self.obs_scales.ee_sphe_yaw_cmd, self.obs_scales.ee_timing_cmd, 
                                            self.obs_scales.end_effector_roll_cmd, self.obs_scales.end_effector_pitch_cmd, 
                                            self.obs_scales.end_effector_yaw_cmd, 1.0],
                                            device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]
        # else:
        #     self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
        #                                         self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
        #                                         self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
        #                                         self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
        #                                         self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
        #                                         self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
        #                                         self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd],
        #                                         device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]            
            
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.last_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool,
                                             device=self.device,
                                             requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 10:13])
       
        
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_RL = torch.tensor(self.cfg.env.default_leg_dof_pos_RL, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name] #26.0
                    # print("dof_name: ", dof_name, "p_gain: ", self.cfg.control.stiffness[dof_name])
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True

            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        if len(self.cfg.commands.p_gains_arm):
            self.p_gains[12:19] = torch.tensor(self.cfg.commands.p_gains_arm, device=self.device)
        if len(self.cfg.commands.d_gains_arm):
            self.d_gains[12:19] = torch.tensor(self.cfg.commands.d_gains_arm, device=self.device)
        
        if len(self.cfg.commands.p_gains_legs):
            self.p_gains[:12] = torch.tensor(self.cfg.commands.p_gains_legs, device=self.device)
        if len(self.cfg.commands.d_gains_legs):
            self.d_gains[:12] = torch.tensor(self.cfg.commands.d_gains_legs, device=self.device)

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.cfg.control.control_type == "actuator_net":
            actuator_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_go1.pt'
            actuator_network = torch.jit.load(actuator_path, map_location=self.device)

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                torques = actuator_network(xs.view(self.num_envs * 12, 6))
                return torques.view(self.num_envs, 12)

            self.actuator_network = eval_actuator_network

            self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.pos_err = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.ee_init_pos_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.ball_drags = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.ball_restitutions = self.default_restitution * torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.ball_friction_coeffs = self.default_friction * torch.ones(self.num_envs, dtype=torch.float,
                                                                  device=self.device,
                                                                  requires_grad=False)
        self.gripper_pos_tracking_error_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        self.gripper_ori_tracking_error_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        self.gripper_xy_force_tracking_error_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        self.gripper_z_force_tracking_error_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        self.lin_vel_tracking_error_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        self.ang_vel_tracking_error_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        # if custom initialization values were passed in, set them here
        dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
                           "Kp_factors", "Kd_factors"]
        if self.initial_dynamics_dict is not None:
            for k, v in self.initial_dynamics_dict.items():
                if k in dynamics_params:
                    setattr(self, k, v.to(self.device))

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        
        # commands curriculum 
        # if self.cfg.commands.inverse_IK_door_opening:
        self.initial_ee_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # 3 = (radius, pitch, yaw)
        self.initial_ee_rpy = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # 3 = (roll, pitch, yaw)
        self.ee_target_pos_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) 
        self.ee_target_rpy_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.trajectory_time = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_or_position_control = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        # To init the tensors only once at the beginning of a training 
        self.init_training = True
            
        if self.cfg.commands.force_control:
            try:
                data = np.load(MINI_GYM_ROOT_DIR + '/b1_gym/envs/base/wbc_data_isaac.npz')
                self.force_control_dof_pos_init_array = torch.tensor(data['dof_pos'], dtype=torch.float, device=self.device)
                self.force_control_base_pos_init_array = torch.tensor(data['base_pos'], dtype=torch.float, device=self.device)
                print("self.force_control_base_pos_init_array: ", self.force_control_base_pos_init_array.shape)
            except:
                print("WARNING, something went wrong when loading the init states for the force controller")
                exit(0)
                       
            self.force_control_dof_pos_init = torch.zeros(self.num_envs, self.num_actuated_dof, dtype=torch.float, device=self.device,
                                                    requires_grad=False)
            self.force_control_base_pos_init = torch.zeros(self.num_envs, 7, dtype=torch.float, device=self.device,
                                                    requires_grad=False)
            self.init_idx = torch.randint(self.force_control_base_pos_init_array.shape[0], (self.num_envs,), device=self.device) # one init pos per env for entire training

            self.force_control_base_pos_init[:] = self.force_control_base_pos_init_array[self.init_idx, :]
            self.force_control_dof_pos_init[:] = self.force_control_dof_pos_init_array[self.init_idx, :]

        self.freed_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
            

        if self.cfg.commands.teleop_occulus:
            self.teleop_gripper_value = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.teleop_joint6_value = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.commands.sample_feasible_commands:
            self.target_joint_values = [0.0]*self.num_actuated_dof

    def _init_command_distribution(self, env_ids):
        # new style curriculum
        self.category_names = ['nominal']
        if self.cfg.commands.gaitwise_curricula:
            self.category_names = ['pronk', 'trot', 'pace', 'bound']

        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            from .curriculum import RewardThresholdCurriculum
            CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                                        x_vel=(self.cfg.commands.limit_vel_x[0],
                                                                self.cfg.commands.limit_vel_x[1],
                                                                self.cfg.commands.num_bins_vel_x),
                                                        y_vel=(self.cfg.commands.limit_vel_y[0],
                                                                self.cfg.commands.limit_vel_y[1],
                                                                self.cfg.commands.num_bins_vel_y),
                                                        yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                                                                    self.cfg.commands.limit_vel_yaw[1],
                                                                    self.cfg.commands.num_bins_vel_yaw),
                                                        body_height=(self.cfg.commands.limit_body_height[0],
                                                                        self.cfg.commands.limit_body_height[1],
                                                                        self.cfg.commands.num_bins_body_height),
                                                        gait_frequency=(self.cfg.commands.limit_gait_frequency[0],
                                                                        self.cfg.commands.limit_gait_frequency[1],
                                                                        self.cfg.commands.num_bins_gait_frequency),
                                                        gait_phase=(self.cfg.commands.limit_gait_phase[0],
                                                                    self.cfg.commands.limit_gait_phase[1],
                                                                    self.cfg.commands.num_bins_gait_phase),
                                                        gait_offset=(self.cfg.commands.limit_gait_offset[0],
                                                                        self.cfg.commands.limit_gait_offset[1],
                                                                        self.cfg.commands.num_bins_gait_offset),
                                                        gait_bounds=(self.cfg.commands.limit_gait_bound[0],
                                                                        self.cfg.commands.limit_gait_bound[1],
                                                                        self.cfg.commands.num_bins_gait_bound),
                                                        gait_duration=(self.cfg.commands.limit_gait_duration[0],
                                                                        self.cfg.commands.limit_gait_duration[1],
                                                                        self.cfg.commands.num_bins_gait_duration),
                                                        footswing_height=(self.cfg.commands.limit_footswing_height[0],
                                                                            self.cfg.commands.limit_footswing_height[1],
                                                                            self.cfg.commands.num_bins_footswing_height),
                                                        body_pitch=(self.cfg.commands.limit_body_pitch[0],
                                                                    self.cfg.commands.limit_body_pitch[1],
                                                                    self.cfg.commands.num_bins_body_pitch),
                                                        body_roll=(self.cfg.commands.limit_body_roll[0],
                                                                    self.cfg.commands.limit_body_roll[1],
                                                                    self.cfg.commands.num_bins_body_roll),
                                                        stance_width=(0, 0, 1),#(self.cfg.commands.limit_ee_force_magnitude[0],
                                                                        #self.cfg.commands.limit_ee_force_magnitude[1],
                                                                        #self.cfg.commands.num_bins_stance_width),
                                                        stance_length=(0, 0, 1),#(self.cfg.commands.limit_ee_force_magnitude[0],
                                                                            #self.cfg.commands.limit_ee_force_magnitude[1],
                                                                            #self.cfg.commands.num_bins_stance_length),
                                                        aux_reward_coef=(0, 0, 1),#(self.cfg.commands.limit_ee_force_z[0],
                                                                    # self.cfg.commands.limit_ee_force_z[1],
                                                                    # self.cfg.commands.num_bins_aux_reward_coef),
                                                        ee_sphe_radius=(self.cfg.commands.limit_ee_sphe_radius[0],
                                                                    self.cfg.commands.limit_ee_sphe_radius[1],
                                                                    self.cfg.commands.num_bins_ee_sphe_radius),   
                                                        ee_sphe_pitch=(self.cfg.commands.limit_ee_sphe_pitch[0],
                                                                    self.cfg.commands.limit_ee_sphe_pitch[1],
                                                                    self.cfg.commands.num_bins_ee_sphe_pitch),   
                                                        ee_sphe_yaw=(self.cfg.commands.limit_ee_sphe_yaw[0],
                                                                    self.cfg.commands.limit_ee_sphe_yaw[1],
                                                                    self.cfg.commands.num_bins_ee_sphe_yaw),    
                                                        ee_timing=(self.cfg.commands.limit_ee_timing[0],
                                                                    self.cfg.commands.limit_ee_timing[1],
                                                                    self.cfg.commands.num_bins_ee_timing),                                                                                                              
                                                        # end_effector_pos_x=(self.cfg.commands.limit_end_effector_pos_x[0],
                                                        #                     self.cfg.commands.limit_end_effector_pos_x[1],
                                                        #                     self.cfg.commands.num_bins_end_effector_pos_x),
                                                        # end_effector_pos_y=(self.cfg.commands.limit_end_effector_pos_y[0],
                                                        #                     self.cfg.commands.limit_end_effector_pos_y[1],
                                                        #                     self.cfg.commands.num_bins_end_effector_pos_y),    
                                                        # end_effector_pos_z=(self.cfg.commands.limit_end_effector_pos_z[0],
                                                        #                     self.cfg.commands.limit_end_effector_pos_z[1],
                                                        #                     self.cfg.commands.num_bins_end_effector_pos_z),      
                                                        end_effector_roll=(self.cfg.commands.limit_end_effector_roll[0],
                                                                            self.cfg.commands.limit_end_effector_roll[1],
                                                                            self.cfg.commands.num_bins_end_effector_roll),      
                                                        end_effector_pitch=(self.cfg.commands.limit_end_effector_pitch[0],
                                                                            self.cfg.commands.limit_end_effector_pitch[1],
                                                                            self.cfg.commands.num_bins_end_effector_pitch),   
                                                        end_effector_yaw=(self.cfg.commands.limit_end_effector_yaw[0],
                                                                            self.cfg.commands.limit_end_effector_yaw[1],
                                                                            self.cfg.commands.num_bins_end_effector_yaw),   
                                                        # end_effector_gripper=(self.cfg.commands.limit_end_effector_gripper[0],
                                                        #                     self.cfg.commands.limit_end_effector_gripper[1],
                                                        #                     self.cfg.commands.num_bins_end_effector_gripper),                                                                                                                                                                                                                                                                                                                                                              
                                                        force_or_position_mode=(0, 1, 1),   
                                                        )
                                ]
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int32)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int32)    

        
        low = np.array(
            [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
            self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.body_height_cmd[0],
            self.cfg.commands.gait_frequency_cmd_range[0],
            self.cfg.commands.gait_phase_cmd_range[0], self.cfg.commands.gait_offset_cmd_range[0],
            self.cfg.commands.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
            self.cfg.commands.footswing_height_range[0], self.cfg.commands.body_pitch_range[0],
            self.cfg.commands.body_roll_range[0],self.cfg.commands.ee_force_magnitude[0],
            self.cfg.commands.ee_force_direction_angle[0], self.cfg.commands.ee_force_z[0], 
            self.cfg.commands.ee_sphe_radius[0], self.cfg.commands.ee_sphe_pitch[0], 
            self.cfg.commands.ee_sphe_yaw[0], self.cfg.commands.ee_timing[0],
            self.cfg.commands.end_effector_roll[0], self.cfg.commands.end_effector_pitch[0], 
            self.cfg.commands.end_effector_yaw[0], 0])
        high = np.array(
            [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
            self.cfg.commands.ang_vel_yaw[1], self.cfg.commands.body_height_cmd[1],
            self.cfg.commands.gait_frequency_cmd_range[1],
            self.cfg.commands.gait_phase_cmd_range[1], self.cfg.commands.gait_offset_cmd_range[1],
            self.cfg.commands.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1],
            self.cfg.commands.footswing_height_range[1], self.cfg.commands.body_pitch_range[1],
            self.cfg.commands.body_roll_range[1],self.cfg.commands.ee_force_magnitude[1],
            self.cfg.commands.ee_force_direction_angle[1], self.cfg.commands.ee_force_z[1], 
            self.cfg.commands.ee_sphe_radius[1], self.cfg.commands.ee_sphe_pitch[1], 
            self.cfg.commands.ee_sphe_yaw[1], self.cfg.commands.ee_timing[1],
            self.cfg.commands.end_effector_roll[1], self.cfg.commands.end_effector_pitch[1], 
            self.cfg.commands.end_effector_yaw[1], 1])
        
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

        # assign the control mode to each environment (force or position)
        # 0 = position, 1 = force
        self.force_or_position_control = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        

    def _resample_force_or_position_control(self, env_ids):
        if self.cfg.commands.hybrid_mode == "mixed":
            self.force_or_position_control[env_ids] = torch.rand(len(env_ids), device=self.device)
        elif self.cfg.commands.hybrid_mode == "binary":
            self.force_or_position_control[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device).float()
        elif self.cfg.commands.hybrid_mode == "force":
            self.force_or_position_control[env_ids] = torch.ones(len(env_ids), device=self.device)
        elif self.cfg.commands.hybrid_mode == "position":
            self.force_or_position_control[env_ids] = torch.zeros(len(env_ids), device=self.device)
        else:
            print("Error: hybrid_mode not recognized")
            exit(0)
        self.commands[env_ids, INDEX_FORCE_OR_POSITION_INDICATOR] = self.force_or_position_control[env_ids]

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from b1_gym.rewards.b1_loco_z1_gaitfree_rewards import B1LocoZ1GaitfreeRewards
        reward_containers = {"B1LocoZ1GaitfreeRewards": B1LocoZ1GaitfreeRewards,}
        

        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def _create_envs(self):

        all_assets = []

        # create robot

        from b1_gym.robots.go1 import Go1
        from b1_gym.robots.b1 import B1
        from b1_gym.robots.b1_plus_z1 import B1PlusZ1
        from b1_gym.robots.b1_plus_dismounted_z1 import B1PlusDismountedZ1
        from b1_gym.robots.z1 import Z1

        robot_classes = {
            'go1': Go1,
            'b1': B1,
            'b1_plus_z1': B1PlusZ1,
            # 'b1_plus_dismounted_z1': B1PlusDismountedZ1,
            'z1': Z1,
        }

        self.robot = robot_classes[self.cfg.robot.name](self)
        all_assets.append(self.robot)
        self.robot_asset, dof_props_asset, rigid_shape_props_asset = self.robot.initialize()
        self.dof_props_asset = dof_props_asset
        object_init_state_list = self.cfg.ball.ball_init_pos + self.cfg.ball.ball_init_rot + self.cfg.ball.ball_init_lin_vel + self.cfg.ball.ball_init_ang_vel
        self.object_init_state = to_torch(object_init_state_list, device=self.device, requires_grad=False)

        # create objects

        from b1_gym.assets.ball import Ball
        from b1_gym.assets.cube import Cube
        from b1_gym.assets.door import Door
        from b1_gym.assets.chair import Chair
        from b1_gym.assets.bucket_of_balls import BallBucket

        asset_classes = {
            "ball": Ball,
            "cube": Cube,
            "door": Door,
            "chair": Chair,
            "ballbucket": BallBucket,
        }

        if self.cfg.env.add_balls:
            self.asset = asset_classes[self.cfg.ball.asset](self)
            all_assets.append(self.asset)
            self.ball_asset, ball_rigid_shape_props_asset = self.asset.initialize()
            self.ball_force_feedback = self.asset.get_force_feedback()
            self.num_object_bodies = self.gym.get_asset_rigid_body_count(self.ball_asset)
        else:
            self.ball_force_feedback = None
            self.num_object_bodies = 0

        # aggregate the asset properties
        self.total_rigid_body_num = sum([asset.get_num_bodies() for asset in 
                                        all_assets])
        self.num_dof = sum([asset.get_num_dof() for asset in
                            all_assets])
        self.num_actuated_dof = sum([asset.get_num_actuated_dof() for asset in
                                        all_assets])

        if self.cfg.terrain.mesh_type == "boxes":
            self.total_rigid_body_num += self.cfg.terrain.num_cols * self.cfg.terrain.num_rows

        

        self.ball_init_pose = gymapi.Transform()
        self.ball_init_pose.p = gymapi.Vec3(*self.object_init_state[:3])

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.gripper_stator_index = [index for index, body_name in enumerate(body_names) if body_name == "link06"][0] # For random pushes
        self.robot_base_index = [index for index, body_name in enumerate(body_names) if body_name == "base"][0] # For random pushes
                  
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._get_env_origins(torch.arange(self.num_envs, device=self.device), self.cfg)
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.robot_actor_handles = []
        self.object_actor_handles = []
        self.imu_sensor_handles = []
        self.envs = []
        self.robot_actor_idxs = []
        self.object_actor_idxs = []

        self.object_rigid_body_idxs = []
        self.feet_rigid_body_idxs = []
        self.robot_rigid_body_idxs = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device), self.cfg)
        self._randomize_gravity()
        self._randomize_ball_drag()

        if self.cfg.env.all_agents_share:
            shared_env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

        for i in range(self.num_envs):
            # create env instance
            if self.cfg.env.all_agents_share:
                env_handle = shared_env
            else:
                env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # add robots
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            robot_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "robot", i,
                                                  self.cfg.asset.self_collisions, 0)
            self.gym.enable_actor_dof_force_sensors(env_handle, robot_handle) # to be able to read the torques applied 
            for bi in body_names:
                self.robot_rigid_body_idxs.append(self.gym.find_actor_rigid_body_handle(env_handle, robot_handle, bi))
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)
            
            self.robot_actor_handles.append(robot_handle)
            self.robot_actor_idxs.append(self.gym.get_actor_index(env_handle, robot_handle, gymapi.DOMAIN_SIM))

            # add objects
            if self.cfg.env.add_balls:
                ball_rigid_shape_props = self._process_ball_rigid_shape_props(ball_rigid_shape_props_asset, i)
                self.gym.set_asset_rigid_shape_properties(self.ball_asset, ball_rigid_shape_props)
                ball_handle = self.gym.create_actor(env_handle, self.ball_asset, self.ball_init_pose, "ball", i, 0)
                color = gymapi.Vec3(1, 1, 0)
                self.gym.set_rigid_body_color(env_handle, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                ball_idx = self.gym.get_actor_rigid_body_index(env_handle, ball_handle, 0, gymapi.DOMAIN_SIM)
                ball_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
                ball_body_props[0].mass = self.cfg.ball.mass*(np.random.rand()*0.3+0.5)
                self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_body_props, recomputeInertia=True)
                # self.gym.set_actor_rigid_shape_properties(env_handle, ball_handle, ball_shape_props)
                self.object_actor_handles.append(ball_handle)
                self.object_rigid_body_idxs.append(ball_idx)
                self.object_actor_idxs.append(self.gym.get_actor_index(env_handle, ball_handle, gymapi.DOMAIN_SIM))
                

            self.envs.append(env_handle)

        self.robot_actor_idxs = torch.Tensor(self.robot_actor_idxs).to(device=self.device,dtype=torch.long)
        self.object_actor_idxs = torch.Tensor(self.object_actor_idxs).to(device=self.device,dtype=torch.long)
        self.object_rigid_body_idxs = torch.Tensor(self.object_rigid_body_idxs).to(device=self.device,dtype=torch.long)
        
            
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[0],
                                                                         feet_names[i])

        

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.robot_actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.robot_actor_handles[0],
                                                                                        termination_contact_names[i])
        ################
        ### Add sensors
        ################

        self.initialize_sensors()
        
        # if perception is on, set up camera
        if self.cfg.perception.compute_segmentation or self.cfg.perception.compute_rgb or self.cfg.perception.compute_depth:
            self.initialize_cameras(range(self.num_envs))

        if self.cfg.perception.measure_heights:
            from b1_gym.sensors.heightmap_sensor import HeightmapSensor
            self.heightmap_sensor = HeightmapSensor(self)

        # if recording video, set up camera
        if self.cfg.env.record_video:
            from b1_gym.sensors.floating_camera_sensor import FloatingCameraSensor
            self.rendering_camera = FloatingCameraSensor(self)
            

        ################
        ### Initialize Logging
        ################

        from b1_gym.utils.logger import Logger
        self.logger = Logger(self)
        
        self.video_writer = None
        self.video_frames = []
        self.complete_video_frames = []


    def render(self, mode="rgb_array", target_loc=None, cam_distance=None):
        # if self.virtual_display and mode == "rgb_array":
        #     img = self.virtual_display.grab()
        #     print(img)
        #     img = np.array(img)
        #     print(img.shape)
        #     if len(img.shape) > 0:
        #         w, h = img.shape
        #         img = img.reshape([w, h // 4, 4])
        #     else:
        #         img = np.zeros((360, 360))
        #     return img
        # else:
        if self.cfg.viewer.follow_robot:
            forward_vec = quat_apply_yaw(self.base_quat, torch.tensor([2., 0., -1.0], dtype=torch.float32, device=self.device))
            self.rendering_camera.set_position(self.base_pos[0, :], -forward_vec)
        else:
            self.rendering_camera.set_position(target_loc, cam_distance)
        return self.rendering_camera.get_observation()

    def _render_headless(self):
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            bx, by, bz = self.root_states[self.robot_actor_idxs[0], 0], self.root_states[self.robot_actor_idxs[0], 1], self.root_states[self.robot_actor_idxs[0], 2]
            target_loc = [bx, by , bz]
            cam_distance = [0, -1.0, 1.0]
            self.rendering_camera.set_position(target_loc, cam_distance)
            self.video_frame = self.rendering_camera.get_observation()
            self.video_frames.append(self.video_frame)

    def start_recording(self):
        self.complete_video_frames = None
        self.record_now = True

    def pause_recording(self):
        self.complete_video_frames = []
        self.video_frames = []
        self.record_now = False

    def get_complete_frames(self):
        if self.complete_video_frames is None:
            return []
        return self.complete_video_frames

    def _get_env_origins(self, env_ids, cfg):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            # put robots at the origins defined by the terrain
            max_init_level = cfg.terrain.max_init_terrain_level
            min_init_level = cfg.terrain.min_init_terrain_level
            if not cfg.terrain.curriculum: max_init_level = cfg.terrain.num_rows - 1
            if not cfg.terrain.curriculum: min_init_level = 0
            assert cfg.terrain.center_span <= cfg.terrain.num_rows // 2, f"Center span {cfg.terrain.center_span} is greater than the terrain grid size, decrease!"
            if cfg.terrain.center_robots:
                min_terrain_level = cfg.terrain.num_rows // 2 - cfg.terrain.center_span
                max_terrain_level = cfg.terrain.num_rows // 2 + cfg.terrain.center_span - 1
                min_terrain_type = cfg.terrain.num_cols // 2 - cfg.terrain.center_span
                max_terrain_type = cfg.terrain.num_cols // 2 + cfg.terrain.center_span - 1
                self.terrain_levels[env_ids] = torch.randint(min_terrain_level, max_terrain_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = torch.randint(min_terrain_type, max_terrain_type + 1, (len(env_ids),),
                                                            device=self.device)
            else:
                self.terrain_levels[env_ids] = torch.randint(min_init_level, max_init_level + 1, (len(env_ids),),
                                                            device=self.device)
                self.terrain_types[env_ids] = torch.div(torch.arange(len(env_ids), device=self.device),
                                                    (len(env_ids) / cfg.terrain.num_cols), rounding_mode='floor').to(
                    torch.long)
            cfg.terrain.max_terrain_level = cfg.terrain.num_rows
            cfg.terrain.terrain_origins = torch.from_numpy(cfg.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[env_ids] = cfg.terrain.terrain_origins[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        elif cfg.terrain.mesh_type in ["boxes", "boxes_tm"]:
            self.custom_origins = True
            # put robots at the origins defined by the terrain
            max_init_level = int(cfg.terrain.max_init_terrain_level + cfg.terrain.num_border_boxes)
            min_init_level = int(cfg.terrain.min_init_terrain_level + cfg.terrain.num_border_boxes)
            if not cfg.terrain.curriculum: max_init_level = int(cfg.terrain.num_rows - 1 - cfg.terrain.num_border_boxes)
            if not cfg.terrain.curriculum: min_init_level = int(0 + cfg.terrain.num_border_boxes)

            if cfg.terrain.center_robots:
                self.min_terrain_level = cfg.terrain.num_rows // 2 - cfg.terrain.center_span
                self.max_terrain_level = cfg.terrain.num_rows // 2 + cfg.terrain.center_span - 1
                min_terrain_type = cfg.terrain.num_cols // 2 - cfg.terrain.center_span
                max_terrain_type = cfg.terrain.num_cols // 2 + cfg.terrain.center_span - 1
                self.terrain_levels[env_ids] = torch.randint(self.min_terrain_level, self.max_terrain_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = torch.randint(min_terrain_type, max_terrain_type + 1, (len(env_ids),),
                                                            device=self.device)
            else:
                self.terrain_levels[env_ids] = torch.randint(min_init_level, max_init_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = (torch.div(torch.arange(len(env_ids), device=self.device),
                                                        (len(env_ids) / (cfg.terrain.num_cols - 2 * cfg.terrain.num_border_boxes)),
                                                        rounding_mode='floor') + cfg.terrain.num_border_boxes).to(torch.long)
                self.min_terrain_level = int(cfg.terrain.num_border_boxes)
                self.max_terrain_level = int(cfg.terrain.num_rows - cfg.terrain.num_border_boxes)
            cfg.terrain.env_origins[:, :, 2] = self.terrain_obj.terrain_cell_center_heights.cpu().numpy()
            cfg.terrain.terrain_origins = torch.from_numpy(cfg.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[env_ids] = cfg.terrain.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            # self.env_origins[env_ids, 2] = self.terrain_cell_center_heights[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        else:
            self.custom_origins = False
            # create a grid of robots
            num_cols = np.floor(np.sqrt(len(env_ids)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = cfg.env.env_spacing
            self.env_origins[env_ids, 0] = spacing * xx.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 1] = spacing * yy.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.obs_scales
        self.reward_scales = vars(self.cfg.reward_scales)
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        cfg.command_ranges = vars(cfg.commands)
        if cfg.terrain.mesh_type not in ['heightfield', 'trimesh', 'boxes', 'boxes_tm']:
            cfg.terrain.curriculum = False
        self.max_episode_length_s = cfg.env.episode_length_s
        cfg.env.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.max_episode_length = cfg.env.max_episode_length

        cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)
        # Gripper 
        cfg.domain_rand.push_interval_gripper_min = np.ceil(cfg.domain_rand.push_gripper_interval_s[0] / self.dt)     
        cfg.domain_rand.push_interval_gripper_max = np.ceil(cfg.domain_rand.push_gripper_interval_s[1] / self.dt)       
        cfg.domain_rand.push_duration_gripper_min = np.ceil(cfg.domain_rand.push_gripper_duration_s[0] / self.dt)   
        cfg.domain_rand.push_duration_gripper_max = np.ceil(cfg.domain_rand.push_gripper_duration_s[1] / self.dt)   
        cfg.commands.settling_nb_it_force_gripper = np.ceil(cfg.commands.settling_time_force_gripper_s / self.dt)           
        # Base   
        cfg.domain_rand.push_interval_base = np.ceil(cfg.domain_rand.push_robot_interval_s / self.dt)  
        cfg.domain_rand.push_duration_robot_min = np.ceil(cfg.domain_rand.push_robot_duration_s[0] / self.dt)   
        cfg.domain_rand.push_duration_robot_max = np.ceil(cfg.domain_rand.push_robot_duration_s[1] / self.dt)  

        
        
        cfg.domain_rand.rand_interval = np.ceil(cfg.domain_rand.rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_interval = np.ceil(cfg.domain_rand.gravity_rand_interval_s / self.dt)
        cfg.domain_rand.ball_drag_rand_interval = np.ceil(cfg.domain_rand.ball_drag_rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_duration = np.ceil(
        cfg.domain_rand.gravity_rand_interval * cfg.domain_rand.gravity_impulse_duration)

    def _init_height_points(self, env_ids, cfg):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(cfg.perception.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(cfg.perception.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        cfg.perception.num_height_points = grid_x.numel()
        points = torch.zeros(len(env_ids), cfg.perception.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids, cfg):
        if cfg.terrain.mesh_type == 'plane':
            return torch.zeros(len(env_ids), cfg.perception.num_height_points, device=self.device, requires_grad=False)
        elif cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, cfg.perception.num_height_points),
                                self.height_points[env_ids]) + (self.root_states[self.robot_actor_idxs[env_ids], :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.terrain_obj.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.terrain_obj.height_samples.shape[1] - 2)

        heights1 = self.terrain_obj.height_samples[px, py]
        heights2 = self.terrain_obj.height_samples[px + 1, py]
        heights3 = self.terrain_obj.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(len(env_ids), -1) * self.terrain.cfg.vertical_scale

    def get_heights_points(self, global_positions):
        points = global_positions + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, self.terrain_obj.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.terrain_obj.height_samples.shape[1] - 2)
        heights = self.terrain_obj.height_samples[px, py]
        return heights * self.terrain.cfg.vertical_scale
    
    def get_frictions(self, env_ids, cfg):
        if cfg.terrain.mesh_type == 'plane':
            return torch.zeros(len(env_ids), cfg.perception.num_height_points, device=self.device, requires_grad=False)
        elif cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure friction field with terrain mesh type 'none'")

        points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, cfg.perception.num_height_points),
                                self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)

        # points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.terrain_obj.friction_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.terrain_obj.friction_samples.shape[1] - 2)

        frictions = self.terrain_obj.friction_samples[px, py]

        return frictions.view(len(env_ids), -1)
    
    def get_frictions_points(self, global_positions):
        points = global_positions + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, self.terrain_obj.friction_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.terrain_obj.friction_samples.shape[1] - 2)
        frictions = self.terrain_obj.friction_samples[px, py]
        return frictions
    
    def initialize_cameras(self, env_ids):
        self.cams = {label: [] for label in self.cfg.perception.camera_names}
        self.camera_sensors = {}

        from b1_gym.sensors.attached_camera_sensor import AttachedCameraSensor

        for camera_label, camera_pose, camera_rpy, camera_gimbal in zip(self.cfg.perception.camera_names,
                                                             self.cfg.perception.camera_poses,
                                                             self.cfg.perception.camera_rpys,
                                                             self.cfg.perception.camera_gimbals):
            self.camera_sensors[camera_label] = AttachedCameraSensor(self)
            self.camera_sensors[camera_label].initialize(camera_label, camera_pose, camera_rpy, camera_gimbal, env_ids=env_ids)
        
    def get_segmentation_images(self, env_ids):
        segmentation_images = []
        for camera_name in self.cfg.perception.camera_names:
            segmentation_images = self.camera_sensors[camera_name].get_segmentation_images(env_ids)
        return segmentation_images

    def get_rgb_images(self, env_ids):
        rgb_images = {}
        for camera_name in self.cfg.perception.camera_names:
            rgb_images[camera_name] = self.camera_sensors[camera_name].get_rgb_images(env_ids)
        return rgb_images

    def get_depth_images(self, env_ids):
        depth_images = {}
        for camera_name in self.cfg.perception.camera_names:
            depth_images[camera_name] = self.camera_sensors[camera_name].get_depth_images(env_ids)
        return depth_images

    def compute_energy(self):
        torques = self.torques[:, :12]
        joint_vels = self.dof_vel[:, :12]

        gear_ratios = torch.tensor([1.0, 1.0, 1./1.5,
                            1.0, 1.0, 1./1.5,
                            1.0, 1.0, 1./1.5,
                            1.0, 1.0, 1./1.5,], device=self.device,
                            ) # knee has extra gearing

        power_joule = torch.sum((torques * gear_ratios)**2 * 0.07, dim=1)
        power_mechanical = torch.sum(torch.clip(torques * joint_vels, -3, 10000), dim=1)
        power_battery = 42.0

        return power_joule + power_mechanical + power_battery