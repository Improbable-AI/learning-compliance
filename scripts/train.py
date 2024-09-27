import isaacgym
assert isaacgym
import numpy as np

from params_proto import PrefixProto

import torch

from b1_gym.envs.base.legged_robot_config import Cfg
from b1_gym.envs.b1.b1_plus_z1_config import config_b1_plus_z1
from b1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from b1_gym_learn.ppo_cse import Runner
from b1_gym_learn.ppo_cse.actor_critic import AC_Args
from b1_gym_learn.ppo_cse.ppo import PPO_Args
from b1_gym_learn.ppo_cse import RunnerArgs

from b1_gym.envs.wrappers.history_wrapper import HistoryWrapper

class RunCfg(PrefixProto, cli=False):
    experiment_group = "example_sweep"
    experiment_job_type= "default"


def configure_env():

    config_b1_plus_z1(Cfg)

    # Cfg.commands.control_only_z1 = True
    # Cfg.commands.interpolate_ee_cmds = False 
    Cfg.commands.control_ee_ori = False 
    Cfg.commands.control_ee_ori_only_yaw = False
    
    # Cfg.env.num_envs = 30
    # Cfg.sim.physx.max_gpu_contact_pairs = 2 ** 25

    # RunnerArgs.resume = False
    # RunnerArgs.resume_path = "improbableai/b1-z1-IK/gtkntvpq"
    # RunnerArgs.resume_checkpoint = 'tmp/legged_data/ac_weights_5600.pt' 

    Cfg.robot.name = "b1_plus_z1"

    # Observations
    Cfg.sensors.sensor_names = [
                        "OrientationSensor",    #          : size 3   projected gravity 
                        "RCSensor",             # Commands : size 19
                        "JointPositionSensor",  #          : size 19
                        "JointVelocitySensor",  #          : size 19
                        "ActionSensor",         #          : size 19
                        "ClockSensor",          #          : size 4
                        ]
    Cfg.sensors.sensor_args = {
                        "OrientationSensor": {},
                        "RCSensor": {},
                        "JointPositionSensor": {},
                        "JointVelocitySensor": {},
                        "ActionSensor": {},
                        "ClockSensor": {},
                        }
    Cfg.env.num_scalar_observations = 83 # 95 for arm
    Cfg.env.num_observations = 83 # 99
    Cfg.env.episode_length_s = 20
    Cfg.commands.resampling_time = 10

    # Privileged observations
    Cfg.sensors.privileged_sensor_names = [
                        # "EeGripperForceSensor": {},  # size 1
                        # "EeBaseForceSensor": {},  # size 3
                        # "FrictionSensor": {},       # size 1
                        # "RestitutionSensor": {},    # size 1 
                        "BodyVelocitySensor",   # size 3
                        "JointDynamicsSensor",  # size 3
                        "EeGripperForceSensor", # size 3
                        "FrictionSensor",
                        "EeGripperPositionSensor",
                        "EeGripperTargetPositionSensor",
                        # "ComDisplacementSensor",
                        # "EeGripperPosSensor",
    ]
    Cfg.sensors.privileged_sensor_args = {
                        # "EeGripperForceSensor": {},  # size 1
                        # # "EeBaseForceSensor": {},  # size 3
                        # "FrictionSensor": {},
                        # "RestitutionSensor": {},
                        "BodyVelocitySensor": {},  
                        "JointDynamicsSensor": {},
                        "EeGripperForceSensor": {},
                        "FrictionSensor": {},
                        "EeGripperPositionSensor": {},
                        "EeGripperTargetPositionSensor": {},
                        # "ComDisplacementSensor": {},
                        # "EeGripperPosSensor": {},
    }
    Cfg.env.num_privileged_obs = 16
    AC_Args.adaptation_labels = ["motion_loss", "dynamics_loss", "force_loss", "friction_loss", "gripper_pos_loss", "gripper_target_pos_loss"]
    AC_Args.adaptation_dims = [3, 3, 3, 1, 3, 3]
    AC_Args.adaptation_weights = [1, 1, 0.05, 1, 10, 1]
    
    AC_Args.init_noise_std = 1.0

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    Cfg.commands.distributional_commands = True

    # Domain randomization 
    Cfg.domain_rand.rand_interval_s = 4
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [0.6, 5.0]
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cfg.domain_rand.randomize_gravity = True
    Cfg.domain_rand.gravity_range = [-0.01, 0.01]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.99
    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [-0.05, 0.05]
    Cfg.domain_rand.randomize_ground_friction = True
    Cfg.domain_rand.ground_friction_range = [0.0, 0.01]
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_tile_roughness = True
    Cfg.domain_rand.tile_roughness_range = [0.0, 0.1]

    # Privileged info 
    # Cfg.env.priv_observe_Kd_factor = False
    # Cfg.env.priv_observe_body_velocity = False
    # Cfg.env.priv_observe_body_height = False
    # Cfg.env.priv_observe_desired_contact_states = False
    # Cfg.env.priv_observe_contact_forces = False
    # Cfg.env.priv_observe_foot_displacement = False
    # Cfg.env.priv_observe_gravity_transformed_foot_displacement = False
    # Cfg.env.priv_observe_Kp_factor = False
    # Cfg.env.priv_observe_motor_offset = False
    # Cfg.env.priv_observe_motor_strength = False
    # Cfg.env.priv_observe_ground_friction = False
    # Cfg.env.priv_observe_ground_friction_per_foot = False
    # Cfg.env.priv_observe_com_displacement = False
    # Cfg.env.priv_observe_base_mass = False
    # Cfg.env.priv_observe_restitution = True
    # Cfg.env.priv_observe_friction = True
    # Cfg.env.priv_observe_friction_indep = False
    # Cfg.env.priv_observe_motion = False
    # Cfg.env.priv_observe_gravity_transformed_motion = False
    # Cfg.env.priv_observe_gravity = False

    Cfg.env.num_observation_history = 10

    Cfg.commands.num_commands = 19

    Cfg.terrain.border_size = 0.0
    Cfg.terrain.mesh_type = "boxes_tm"
    Cfg.terrain.num_cols = 20
    Cfg.terrain.num_rows = 20
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.terrain_length = 5.0
    Cfg.terrain.x_init_range = 1.0
    Cfg.terrain.y_init_range = 1.0
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 4
    Cfg.terrain.horizontal_scale = 0.10

    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.terminal_body_height = 0.3  #0.2
    Cfg.rewards.use_terminal_roll_pitch = True
    Cfg.rewards.terminal_body_ori = 0.9
    # Cfg.rewards.base_height_target = 0.5 #0.30
    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.
    Cfg.rewards.gait_vel_sigma = 10.
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = True
    Cfg.rewards.sigma_rew_neg = 0.02

    # Rewards in use 
    Cfg.reward_scales.manip_pos_tracking = 3.0    
    Cfg.reward_scales.manip_energy = -0.000    
    Cfg.reward_scales.tracking_lin_vel_x = -0.0    
    Cfg.reward_scales.tracking_ang_vel_yaw = 0.0  
    Cfg.reward_scales.alive = 0.0    
    Cfg.reward_scales.loco_energy = -0.00000       
    
    Cfg.reward_scales.tracking_lin_vel = 0.0  #1.0
    Cfg.reward_scales.tracking_ang_vel = 0.0
    Cfg.reward_scales.orientation = 0.0 #-5.0
    Cfg.reward_scales.torques = 0.0
    # Cfg.reward_scales.dof_vel = -0.0001 #-1e-4
    # Cfg.reward_scales.dof_acc = -2.5e-7
    # Cfg.reward_scales.collision = -5.0 # -5.0
    # Cfg.reward_scales.action_rate = -0.01 # -0.01
    # Cfg.reward_scales.action_smoothness_1 = -0.1
    # Cfg.reward_scales.action_smoothness_2 = -0.1 # -0.1 
    Cfg.reward_scales.tracking_contacts_shaped_force = 0.0 # 4.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 0.0 # 4.0
    Cfg.reward_scales.dof_pos_limits = -10.0

    # Rewards not in use 
    Cfg.reward_scales.orientation_control = 0.0
    Cfg.reward_scales.feet_contact_forces = 0.0
    Cfg.reward_scales.feet_slip = 0.0
    Cfg.reward_scales.dof_pos = 0.0
    Cfg.reward_scales.jump = 0.0
    Cfg.reward_scales.base_height = 0.0
    Cfg.reward_scales.estimation_bonus = 0.0
    Cfg.reward_scales.feet_impact_vel = -0.0
    Cfg.reward_scales.feet_clearance = -0.0
    Cfg.reward_scales.feet_clearance_cmd = -0.0
    Cfg.reward_scales.feet_clearance_cmd_linear = 0.0
    Cfg.reward_scales.tracking_stance_width = -0.0
    Cfg.reward_scales.tracking_stance_length = -0.0
    Cfg.reward_scales.lin_vel_z = -0.0
    Cfg.reward_scales.ang_vel_xy = -0.00
    Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.hop_symmetry = 0.0
    Cfg.reward_scales.tracking_contacts_shaped_force = 0.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 0.0

    # Commands 
    Cfg.commands.teleop_occulus = False
    Cfg.commands.ee_sphe_radius = [0.4, 0.7] # 
    Cfg.commands.ee_sphe_pitch = [-2*np.pi/5, 0] # 
    Cfg.commands.ee_sphe_yaw = [-3*np.pi/5, 3*np.pi/5] # 
    if Cfg.commands.interpolate_ee_cmds:
        Cfg.commands.ee_timing = [4.0, 7.0] # 
    else:
        Cfg.commands.ee_timing = [0.0, 0.0] # 

    Cfg.commands.lin_vel_x = [0.0, 0.0 ] #[-1.0, 1.0]
    Cfg.commands.lin_vel_y = [0.0, 0.0] # [-0.6, 0.6]
    Cfg.commands.ang_vel_yaw = [0.0, 0.0] #[-1.0, 1.0]
    Cfg.commands.body_height_cmd = [0.0, 0.0]
    Cfg.commands.gait_frequency_cmd_range = [1.5, 1.5]
    Cfg.commands.gait_phase_cmd_range = [0.5, 0.5]
    Cfg.commands.gait_offset_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    Cfg.commands.footswing_height_range = [0.2, 0.2]
    Cfg.commands.body_pitch_range = [0.0, 0.0]
    Cfg.commands.body_roll_range = [0.0, 0.0]
    Cfg.commands.stance_width_range = [0.6, 0.6]
    Cfg.commands.stance_length_range = [0.65, 0.65]
    Cfg.commands.aux_reward_coef_range = [0.0, 0.0]

    # Limits
    Cfg.commands.limit_ee_sphe_radius = [0.4, 0.7] # 
    Cfg.commands.limit_ee_sphe_pitch = [-2*np.pi/5, 0] # 
    Cfg.commands.limit_ee_sphe_yaw = [-3*np.pi/5, 3*np.pi/5] # 
    if Cfg.commands.interpolate_ee_cmds:
        Cfg.commands.limit_ee_timing = [4.0, 7.0] # 
    else:
        Cfg.commands.limit_ee_timing = [0.0, 0.0] # 

    Cfg.commands.limit_vel_x = [0.0, 0.0] #[-1.0, 1.0]
    Cfg.commands.limit_vel_y = [0.0, 0.0] # [-0.6, 0.6]
    Cfg.commands.limit_vel_yaw = [0.0, 0.0] # = [-1.0, 1.0]
    Cfg.commands.limit_body_height = [0.0, 0.0]
    Cfg.commands.limit_gait_frequency = [1.5, 1.5]
    Cfg.commands.limit_gait_phase = [0.5, 0.5]
    Cfg.commands.limit_gait_offset = [0.0, 0.0]
    Cfg.commands.limit_gait_bound = [0.0, 0.0]
    Cfg.commands.limit_gait_duration = [0.5, 0.5]
    Cfg.commands.limit_footswing_height = [0.2, 0.2]
    Cfg.commands.limit_body_pitch = [0.0, 0.0]
    Cfg.commands.limit_body_roll = [0.0, 0.0]
    Cfg.commands.limit_stance_width = [0.6, 0.6]
    Cfg.commands.limit_stance_length = [0.65, 0.65]
    Cfg.commands.limit_aux_reward_coef = [0.0, 0.0]

    # Num bins
    Cfg.commands.num_bins_ee_sphe_radius = 1 # 
    Cfg.commands.num_bins_ee_sphe_pitch = 1 # 
    Cfg.commands.num_bins_ee_sphe_yaw = 1 # 
    Cfg.commands.num_bins_ee_timing = 1 # 

    Cfg.commands.num_bins_vel_x = 1
    Cfg.commands.num_bins_vel_y = 1
    Cfg.commands.num_bins_vel_yaw = 1
    Cfg.commands.num_bins_body_height = 1
    Cfg.commands.num_bins_gait_frequency = 1
    Cfg.commands.num_bins_gait_phase = 1
    Cfg.commands.num_bins_gait_offset = 1
    Cfg.commands.num_bins_gait_bound = 1
    Cfg.commands.num_bins_gait_duration = 1
    Cfg.commands.num_bins_footswing_height = 1
    Cfg.commands.num_bins_body_roll = 1
    Cfg.commands.num_bins_body_pitch = 1
    Cfg.commands.num_bins_stance_width = 1

    Cfg.viewer.follow_robot = False

    Cfg.normalization.friction_range = [0, 1]
    Cfg.normalization.ground_friction_range = [0, 1]
    Cfg.terrain.yaw_init_range = 3.14
    Cfg.normalization.clip_actions = 10.0

    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.binary_phases = False
    Cfg.commands.gaitwise_curricula = False
    Cfg.commands.balance_gait_distribution = False 

    ############################
    # Inverse IK: door opening #
    ############################
    # Cfg.rewards.reward_container_name = "InverseKinematicsRewards"
    Cfg.commands.inverse_IK_door_opening = True      # Specify which commands to define in leggedrobot.py/_init_command_distribution 
    

    Cfg.obs_scales.ee_sphe_radius_cmd = 0.5   # 0.2 - 0.7 
    Cfg.obs_scales.ee_sphe_pitch_cmd = 1.0    # -1.3 , 1.3 
    Cfg.obs_scales.ee_sphe_yaw_cmd = 1.3      # -1.9 , 1.9 
    Cfg.obs_scales.ee_timing_cmd = 0.1       # 1.0, 3.0 # 
    
    Cfg.obs_scales.ee_force_magnitude = 0.01
    Cfg.obs_scales.ee_force_direction_angle = 0.3
    Cfg.obs_scales.ee_force_z = 0.01

    # Collisons
    #Cfg.asset.penalize_contacts_on = ["gripperStator", "gripperMover"]
    Cfg.asset.terminate_after_contacts_on = [] #["thigh", "calf"] # ["base"]


    Cfg.init_state.default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.10,    # 0.3110,  # [rad]
        'RL_hip_joint': 0.10,    # 0.5512,  # [rad]
        'FR_hip_joint':  -0.10,    # -0.2273, # [rad]
        'RR_hip_joint':  -0.10,    # -0.4806, # [rad]

        'FL_thigh_joint': 0.6, # 0.8530,  # [rad]
        'RL_thigh_joint': 1.0, #  0.9293,  # [rad]
        'FR_thigh_joint': 0.6, #  0.7936,  # [rad]
        'RR_thigh_joint': 1.0, #  1.0087,  # [rad]

        'FL_calf_joint': -1.3, # -1.3280,  # [rad]
        'RL_calf_joint': -1.3, #-0.8820,  # [rad]
        'FR_calf_joint': -1.3, #-1.4317,  # [rad]
        'RR_calf_joint': -1.3, #-0.7590,  # [rad]

        'joint1': 0.0,
        'joint2': 1.0, # 1.5
        'joint3': -1.8, # -1.5
        'joint4': -0.1, # -0.54
        'joint5': 0.0,
        'joint6': 0.0,
        'jointGripper': 0.0,

        'hinge1': 0.0,
        'handle_joint': 0.0,
    }

    Cfg.init_state.pos = [0.0, 0.0, 0.65]
    
    Cfg.rewards.reward_container_name = "B1LocoZ1GaitfreeRewards"
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.terminal_body_height = 0.3

    Cfg.env.default_leg_dof_pos_RL = [ 0.0951,  1.0503, -1.2972, -0.0347,  0.9972, -1.3584,  0.4487,  0.8049, -0.8918, -0.3377,  0.8800, -0.8595]

    ######################
    ######## ARM #########
    ######################
    Cfg.reward_scales.manip_pos_tracking_radius = 0.0 #0.5  
    Cfg.reward_scales.torque_limits_arm = -0.005
    Cfg.rewards.soft_torque_limit_arm = 1.0        
    Cfg.reward_scales.dof_vel_arm = -0.003                    
    Cfg.reward_scales.dof_acc_arm = -3e-8                                   
    Cfg.reward_scales.action_rate_arm = 0.0 #-0.003              
    Cfg.reward_scales.action_smoothness_1_arm = -0.01 #-0.05       
    Cfg.reward_scales.action_smoothness_2_arm = -0.01 #-0.02          
    Cfg.reward_scales.dof_pos_limits_arm = -10.0    

    Cfg.rewards.only_positive_rewards_ji22_style = False         
    Cfg.rewards.only_positive_rewards = True
    Cfg.rewards.total_rew_scale = 0.2
    
    ######################
    ######## LEG #########
    ######################
    
    Cfg.reward_scales.dof_vel_leg = -0.0008              
    Cfg.reward_scales.dof_acc_leg = -1.5e-6 #0.0 #-1.5e-7 #-6e-6    
    Cfg.reward_scales.torques_arm = -1e-5
    Cfg.reward_scales.lin_vel_z = -4.0
    Cfg.reward_scales.ang_vel_xy = -0.05                               
    Cfg.reward_scales.action_rate_leg = 0.0 #-0.003          
    Cfg.reward_scales.action_smoothness_1_leg = -0.03 #0.0 #-0.03     
    Cfg.reward_scales.action_smoothness_2_leg = 0.0 #-0.015   
    Cfg.reward_scales.dof_pos_limits_leg = -1.0 # -3     
    Cfg.rewards.soft_dof_pos_limit = 0.9 
    Cfg.reward_scales.dof_pos = -0.5
    
    Cfg.rewards.soft_torque_limit_leg = 1.0
    Cfg.reward_scales.torque_limits_leg = -0.005 # -0.1
    
    Cfg.rewards.swing_ratio = 0.3
    Cfg.rewards.stance_ratio = 0.3

    ### LEG locomotion ###
    Cfg.reward_scales.tracking_lin_vel_x = 0.0
    Cfg.rewards.tracking_sigma_v_x = 0.25
    Cfg.reward_scales.tracking_lin_vel_y = 0.0
    Cfg.rewards.tracking_sigma_v_y = 0.25
    Cfg.reward_scales.tracking_lin_vel = 1.0
    Cfg.reward_scales.tracking_ang_vel_yaw = 2.0 #[0.5, 0.75, 1.0, 1.5, 2.0] 
    Cfg.rewards.tracking_sigma_v_yaw = 0.25
    Cfg.reward_scales.tracking_contacts_shaped_force = 3.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 3.0
    Cfg.commands.lin_vel_x = [-1.0, 1.0] 
    Cfg.commands.limit_vel_x = [-1.0, 1.0]
    Cfg.commands.lin_vel_y = [-0.0, 0.0] 
    Cfg.commands.limit_vel_y = [-0.0, 0.0]
    Cfg.commands.ang_vel_yaw = [-1.5, 1.5] 
    Cfg.commands.limit_vel_yaw = [-1.5, 1.5]
    Cfg.commands.gait_frequency_cmd_range = [2.2, 2.2]
    Cfg.commands.limit_gait_frequency = [2.2, 2.2]
    
    
    
    Cfg.commands.end_effector_pitch = [0., 0.]
    Cfg.commands.end_effector_roll = [0., 0.]
    Cfg.commands.end_effector_yaw = [0., 0.]
    Cfg.commands.limit_end_effector_pitch = [0., 0.]
    Cfg.commands.limit_end_effector_roll = [0., 0.]
    Cfg.commands.limit_end_effector_yaw = [0., 0.]
    
    Cfg.commands.num_commands = 23
    Cfg.env.num_scalar_observations = 87
    Cfg.env.num_observations = 87

    Cfg.reward_scales.feet_contact_forces = -0.1

    Cfg.reward_scales.raibert_heuristic = -30.0
        
    Cfg.rewards.stance_length = 0.65
    Cfg.rewards.stance_width = 0.45
    Cfg.reward_scales.survival = 5.0
    
    Cfg.rewards.max_contact_force = 550.0
    # Cfg.reward_scales.feet_contact_forces = -0.01
    Cfg.reward_scales.torques = -8e-5

    Cfg.rewards.sigma_force_magnitude = 1/50
    Cfg.rewards.sigma_force_z = 1/50
    Cfg.reward_scales.manip_ori_tracking = 2.5 
    Cfg.reward_scales.manip_ori_tracking_yaw_only = 0.0
    Cfg.asset.default_dof_drive_mode = 1

    Cfg.reward_scales.feet_clearance_cmd = -10.0 #-15.0
    Cfg.rewards.footswing_height = 0.10
        
    Cfg.rewards.maintain_ori_force_envs = True
    
    Cfg.env.priv_passthrough = False #True
        
    # reward_scales.manip_ori_tracking = 0.0 #1.5
    Cfg.reward_scales.manip_pos_tracking = 3.0
    Cfg.reward_scales.ee_force_z = 3.0
    Cfg.reward_scales.ee_force_x = 3.0
    Cfg.reward_scales.ee_force_y = 3.0
    Cfg.reward_scales.ee_force_magnitude = 0.0
    Cfg.reward_scales.ee_force_direction_angle = 0.0 #3.0
    
    Cfg.domain_rand.gripper_forced_prob = 0.8
    Cfg.domain_rand.randomize_gripper_force_gains = True
    Cfg.domain_rand.gripper_force_kp_range = [25., 400.]
    Cfg.domain_rand.gripper_force_kd_range = [3.0, 10.0]
    Cfg.domain_rand.prop_kd = 0.1
    Cfg.commands.ee_force_z = [-70, 70]
    Cfg.commands.limit_ee_force_z = [-70, 70]
    Cfg.commands.ee_force_magnitude = [-70, 70]
    Cfg.commands.limit_ee_force_magnitude = [-70, 70]
    
    Cfg.domain_rand.max_push_force_xyz_gripper = [-70, 70]
    Cfg.domain_rand.max_push_force_xyz_gripper_freed = [-70, 70]
    
    Cfg.reward_scales.base_height = 0.0
    Cfg.rewards.base_height_target = 0.55


    Cfg.reward_scales.body_height_tracking = 0.0 #5.0
    Cfg.reward_scales.dof_stand_up_pos_tracking = 0.0 #3.0  

    # reward_scales.feet_contact_forces = 0.0 # 0.001
    # rewards.min_contact_force = 30.0 
    # rewards.max_contact_force = 250.0 # 100

    Cfg.rewards.gait_force_sigma = 30000

    ############ BOTH ###########
    Cfg.reward_scales.collision = -5.0   



    ####### TERMINATION #######

    Cfg.rewards.use_terminal_torque_legs_limits = False #True
    Cfg.rewards.soft_torque_limit_leg = 1.0
    Cfg.rewards.termination_torque_min_time = 25

    Cfg.rewards.use_terminal_torque_arm_limits = False #True
    Cfg.rewards.soft_torque_limit_arm = 1.0 

    Cfg.commands.control_only_z1 = False
    Cfg.commands.interpolate_ee_cmds = True 
    Cfg.commands.sample_feasible_commands = False
    Cfg.commands.teleop_occulus = False

    Cfg.asset.fix_base_link = False
    Cfg.asset.penalize_contacts_on = ["thigh", "calf", "link02", "link03", "link06", "hip"]
    Cfg.asset.terminate_after_contacts_on = ["gripperMover"]

    # noise_scales.dof_vel = 0.0

    Cfg.commands.ee_sphe_radius = [0.3, 0.9]
    Cfg.commands.limit_ee_sphe_radius = [0.3, 0.9]
    Cfg.commands.ee_sphe_pitch = [-2*np.pi/5, 2*np.pi/5]
    Cfg.commands.limit_ee_sphe_pitch = [-2*np.pi/5, 2*np.pi/5]
    Cfg.commands.ee_timing = [1.0, 4.0]
    Cfg.commands.limit_ee_timing = [1.0, 4.0]
    Cfg.commands.settle_time = 2.0

    # Push gripper 
    Cfg.domain_rand.push_gripper_stators = False
    Cfg.domain_rand.push_gripper_interval_s = [3.5, 9.0]
    Cfg.domain_rand.max_push_vel_xyz_gripper = [-40.0, 40.0] # N
    Cfg.domain_rand.push_gripper_duration_s = [1.0, 3.0]
    # domain_rand.push_interval_s = 4. #15.

    # Push robot with v_max 
    Cfg.domain_rand.push_robots = True
    Cfg.domain_rand.max_push_vel_xy = 0.8

    # Push base 
    Cfg.domain_rand.push_robot_base = False
    Cfg.domain_rand.push_robot_interval_s = 5.0
    Cfg.domain_rand.max_push_vel_xyz_robot = [-40.0, 40.0] # N
    Cfg.domain_rand.push_robot_duration_s = [1.0, 2.0] 

    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = True

    # domain_rand.tile_roughness_range = [0.0, 0.1]
    Cfg.domain_rand.tile_roughness_range = [0.0, 0.25]
    
    Cfg.domain_rand.gravity_range = [-0.5, 0.5] #[-1.5, 1.5]

    # Arm PD gains
    default_p_gains = [20.0, 30.0, 30.0, 20.0, 15.0, 10.0, 20.0]
    default_d_gains = [2000.0]*7

    unitree_p_gains = [kp*25.6 for kp in default_p_gains]
    unitree_d_gains = [kd*0.0128 for kd in default_d_gains]

    unitree_p_gains_div6 = [kp/6 for kp in unitree_p_gains]
    unitree_d_gains_div6 = [kd/6 for kd in unitree_d_gains]
    unitree_p_gains_div6[5] = unitree_p_gains_div6[5]*6/4
    unitree_d_gains_div6[5] = unitree_d_gains_div6[5]*6/4

    Cfg.commands.p_gains_arm = unitree_p_gains_div6
    Cfg.commands.d_gains_arm = unitree_d_gains_div6

    # # Legs PD gains

    Cfg.commands.p_gains_legs = [180.0, 180.0, 300.0]*4
    Cfg.commands.d_gains_legs = [8.0, 8.0, 15.0]*4

    # Position control
    Cfg.control.decimation = 4
    # override position gains
    Cfg.commands.p_gains_arm = [64., 128., 64., 64., 64., 64., 64.]
    Cfg.commands.d_gains_arm = [1.5, 3.0, 1.5, 1.5, 1.5, 1.5, 1.5]

    
    Cfg.commands.hybrid_mode = "binary"

    Cfg.control.arm_scale_reduction = 2.0

    Cfg.env.recording_width_px = 180
    Cfg.env.recording_height_px = 120

    return Cfg



def train_b1_z1_IK(headless=True, **deps):

    sim_device = 'cuda:0'
    Cfg = configure_env()

    PPO_Args.entropy_coef = 0.005
    RunnerArgs.num_steps_per_env = 48
    RunnerArgs.save_video_interval = 500
    RunCfg.experiment_job_type = "release"
    RunCfg.experiment_group = "wbc"

    env = VelocityTrackingEasyEnv(sim_device=sim_device, headless=headless, cfg=Cfg)
    env = HistoryWrapper(env, reward_scaling=1.0)

    # log the experiment parameters
    import wandb
    wandb.init(
      # set the wandb project where this run will be logged
      project="b1-loco-z1-manip",
      group=RunCfg.experiment_group,
      job_type=RunCfg.experiment_job_type,

      # track hyperparameters and run metadata
      config={
      "AC_Args": vars(AC_Args),
      "PPO_Args": vars(PPO_Args),
      "RunnerArgs": vars(RunnerArgs),
      "Cfg": vars(Cfg),
      },
    )

    runner = Runner(env, device=sim_device)
    runner.learn(num_learning_iterations=100000, init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    from pathlib import Path

    stem = Path(__file__).stem

    train_b1_z1_IK(headless=True)