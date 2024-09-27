from typing import Union

from params_proto.proto import Meta

from b1_gym.envs.base.legged_robot_config import Cfg

def config_b1_plus_z1(Cnfg: Union[Cfg, Meta]):
    _ = Cnfg.init_state

    _.pos = [0.0, 0.0, 0.65]  # x,y,z [m]
    _.default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,    # 0.3110,  # [rad]
        'RL_hip_joint': 0.1,    # 0.5512,  # [rad]
        'FR_hip_joint':  -0.1,    # -0.2273, # [rad]
        'RR_hip_joint':  -0.1,    # -0.4806, # [rad]

        'FL_thigh_joint': 0.6, # 0.8530,  # [rad]
        'RL_thigh_joint': 1.0, #  0.9293,  # [rad]
        'FR_thigh_joint': 0.6, #  0.7936,  # [rad]
        'RR_thigh_joint': 1.0, #  1.0087,  # [rad]

        'FL_calf_joint': -1.3, # -1.3280,  # [rad]
        'RL_calf_joint': -1.3, #-0.8820,  # [rad]
        'FR_calf_joint': -1.3, #-1.4317,  # [rad]
        'RR_calf_joint': -1.3, #-0.7590,  # [rad]

     # real values isaac [ 0.3110,  0.8530, -1.3280, -0.2273,  0.7936, -1.4317,  0.5512,  0.9293, -0.8820, -0.4806,  1.0087, -0.7590]
     # target stand up   [-0.1,     0.67,    -1.3,    0.1, 0.67, -1.3, -0.1, 0.67, -1.3, 0.1, 0.67, -1.3]

     # ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

    

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
    default_hip_scales = 1.0
    default_thigh_scales = 1.0
    default_calf_scales = 1.0

    _ = Cnfg.control
    _.control_type = 'P'
    _.stiffness = {'joint': 40.}  # [N*m/rad]
    _.damping = {'joint': 2.0}  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    _.action_scale = 0.25
    _.hip_scale_reduction = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    _.decimation = 4

    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/b1/urdf/b1_plus_z1.urdf'
    _.foot_name = "foot"
    _.penalize_contacts_on = ["thigh", "calf", "link02", "link03", "link06", "hip"] #["gripperStator", "gripperMover"]#, "base"]
    _.terminate_after_contacts_on = ["gripperMover"] #["thigh", "calf"] # ["base"]
    _.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    _.flip_visual_attachments = False
    _.fix_base_link = False
    _default_dof_drive_mode = 1
    

    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9
    _.base_height_target = 0.65
    _.reward_container_name = "B1LocoZ1GaitfreeRewards"
    _.only_positive_rewards = True
    _.only_positive_rewards_ji22_style = False
    _.total_rew_scale = 0.2

    _ = Cnfg.reward_scales
    _.torques = -0.0001
    _.orientation = -5.
    _.base_height = -30.

    _ = Cnfg.terrain
    _.mesh_type = 'trimesh'
    _.terrain_noise_magnitude = 0.0
    _.teleport_robots = True
    _.border_size = 0.0
    _.horizontal_scale = 0.10
    _.center_robots = True
    _.center_span = 4
    

    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    _.curriculum = False

    _ = Cnfg.env
    _.num_observations = 87
    _.num_scalar_observations = 87
    _.num_privileged_obs = 16
    _.num_observation_history = 10
    _.episode_length_s = 20
    _.observe_vel = False
    _.num_actions = 19 #19
    _.num_envs = 4000

    _ = Cnfg.robot
    _.name = "b1_plus_z1"

    _ = Cnfg.commands
    _.lin_vel_x = [-1.0, 1.0]
    _.lin_vel_y = [-1.0, 1.0]
    _.p_gains_arm = [64., 128., 64., 64., 64., 64., 64.]
    _.d_gains_arm = [1.5, 3.0, 1.5, 1.5, 1.5, 1.5, 1.5]
    _.p_gains_legs = [180.0, 180.0, 300.0]*4
    _.d_gains_legs = [8.0, 8.0, 15.0]*4

    _ = Cnfg.commands
    _.heading_command = False
    _.resampling_time = 10.0
    _.command_curriculum = True
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30
    _.lin_vel_x = [-0.6, 0.6]
    _.lin_vel_y = [-0.6, 0.6]
    _.ang_vel_yaw = [-1, 1]
    _.num_commands = 23
    _.teleop_occulus = False
    _.ee_sphe_radius = [0.4, 0.7] # 
    _.ee_sphe_pitch = [-2*3.14/5, 0] # 
    _.ee_sphe_yaw = [-3*3.14/5, 3*3.14/5] # 
    _.ee_timing = [4.0, 7.0] # 

    _.lin_vel_x = [-1.0, 1.0 ] #[-1.0, 1.0]
    _.lin_vel_y = [-1.0, 1.0] # [-0.6, 0.6]
    _.ang_vel_yaw = [0.0, 0.0] #[-1.0, 1.0]
    _.body_height_cmd = [0.0, 0.0]
    _.gait_frequency_cmd_range = [1.5, 1.5]
    _.gait_phase_cmd_range = [0.5, 0.5]
    _.gait_offset_cmd_range = [0.0, 0.0]
    _.gait_bound_cmd_range = [0.0, 0.0]
    _.gait_duration_cmd_range = [0.5, 0.5]
    _.footswing_height_range = [0.2, 0.2]
    _.body_pitch_range = [0.0, 0.0]
    _.body_roll_range = [0.0, 0.0]
    _.stance_width_range = [0.6, 0.6]
    _.stance_length_range = [0.65, 0.65]
    _.aux_reward_coef_range = [0.0, 0.0]

    _.limit_ee_sphe_radius = [0.4, 0.7] # 
    _.limit_ee_sphe_pitch = [-2*3.14/5, 0] # 
    _.limit_ee_sphe_yaw = [-3*3.14/5, 3*3.14/5] # 
    _.limit_ee_timing = [4.0, 7.0] # 

    _.limit_vel_x = [-1.0, 1.0] #[-1.0, 1.0]
    _.limit_vel_y = [-1.5, 1.5] # [-0.6, 0.6]
    _.limit_vel_yaw = [0.0, 0.0] # = [-1.0, 1.0]
    _.limit_body_height = [0.0, 0.0]
    _.limit_gait_frequency = [1.5, 1.5]
    _.limit_gait_phase = [0.5, 0.5]
    _.limit_gait_offset = [0.0, 0.0]
    _.limit_gait_bound = [0.0, 0.0]
    _.limit_gait_duration = [0.5, 0.5]
    _.limit_footswing_height = [0.2, 0.2]
    _.limit_body_pitch = [0.0, 0.0]
    _.limit_body_roll = [0.0, 0.0]
    _.limit_stance_width = [0.6, 0.6]
    _.limit_stance_length = [0.65, 0.65]
    _.limit_aux_reward_coef = [0.0, 0.0]

    _.exclusive_phase_offset = False
    _.pacing_offset = False
    _.binary_phases = False
    _.gaitwise_curricula = False
    _.balance_gait_distribution = False

    # Num bins
    _.num_bins_ee_sphe_radius = 1 # 
    _.num_bins_ee_sphe_pitch = 1 # 
    _.num_bins_ee_sphe_yaw = 1 # 
    _.num_bins_ee_timing = 1 # 

    _.num_bins_vel_x = 1
    _.num_bins_vel_y = 1
    _.num_bins_vel_yaw = 1
    _.num_bins_body_height = 1
    _.num_bins_gait_frequency = 1
    _.num_bins_gait_phase = 1
    _.num_bins_gait_offset = 1
    _.num_bins_gait_bound = 1
    _.num_bins_gait_duration = 1
    _.num_bins_footswing_height = 1
    _.num_bins_body_roll = 1
    _.num_bins_body_pitch = 1
    _.num_bins_stance_width = 1

    _ = Cnfg.domain_rand
    _.randomize_base_mass = True
    _.added_mass_range = [-1, 3]
    _.push_robots = False
    _.max_push_vel_xy = 0.5
    _.randomize_friction = True
    _.friction_range = [0.05, 4.5]
    _.randomize_restitution = True
    _.restitution_range = [0.0, 1.0]
    _.restitution = 0.5  # default terrain restitution
    _.randomize_com_displacement = True
    _.com_displacement_range = [-0.1, 0.1]
    _.randomize_motor_strength = True
    _.motor_strength_range = [0.9, 1.1]
    _.randomize_Kp_factor = False
    _.Kp_factor_range = [0.8, 1.3]
    _.randomize_Kd_factor = False
    _.Kd_factor_range = [0.5, 1.5]
    _.rand_interval_s = 6
    
    _ = Cnfg.sensors
    _.sensor_names = [
                        "OrientationSensor",    #          : size 3   projected gravity 
                        "RCSensor",             # Commands : size 19
                        "JointPositionSensor",  #          : size 19
                        "JointVelocitySensor",  #          : size 19
                        "ActionSensor",         #          : size 19
                        "ClockSensor",          #          : size 4
                        ]

    # _ = Cnfg.perception
    # _.measure_heights = False
