# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import numpy as np
from params_proto import PrefixProto, ParamsProto


class Cfg(PrefixProto, cli=False):
    class env(PrefixProto, cli=False):
        num_envs = 4096
        num_observations = 235
        num_scalar_observations = 42
        force_control_init_poses = False
        default_leg_dof_pos_RL = []
        # if not None a privilige_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_privileged_obs = 18
        privileged_future_horizon = 1
        num_actions = 12
        num_observation_history = 15
        history_frame_skip = 1
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        observe_vel = True
        observe_only_ang_vel = False
        observe_only_lin_vel = False
        observe_yaw = False
        offset_yaw_obs = False
        observe_contact_states = False
        observe_command = True
        observe_height_command = False
        observe_gait_commands = False
        observe_timing_parameter = False
        observe_clock_inputs = False
        observe_two_prev_actions = False
        observe_imu = False
        observe_ball_pos = False
        record_video = True
        recording_width_px = 360
        recording_height_px = 240
        recording_mode = "COLOR"
        num_recording_envs = 1
        debug_viz = False

        all_agents_share = False
        add_balls = False

        # priv_observe_friction = True
        # priv_observe_friction_indep = True
        # priv_observe_ground_friction = False
        # priv_observe_ground_friction_per_foot = False
        # priv_observe_restitution = True
        # priv_observe_ground_restitution = False
        # priv_observe_ground_roughness = False
        # priv_observe_stair_height = False
        # priv_observe_stair_run = False
        # priv_observe_stair_ori = False
        # priv_observe_base_mass = True
        # priv_observe_com_displacement = True
        # priv_observe_motor_strength = False
        # priv_observe_motor_offset = False
        # priv_observe_joint_friction = True
        # priv_observe_Kp_factor = True
        # priv_observe_Kd_factor = True
        # priv_observe_contact_forces = False
        # priv_observe_contact_states = False
        # priv_observe_body_velocity = False
        # priv_observe_ball_velocity = False
        # priv_observe_foot_height = False
        # priv_observe_body_height = False
        # priv_observe_gravity = False
        # priv_observe_ball_drag = False
        # priv_observe_terrain_type = False
        # priv_observe_clock_inputs = False
        # priv_observe_doubletime_clock_inputs = False
        # priv_observe_halftime_clock_inputs = False
        # priv_observe_desired_contact_states = False
        # priv_observe_motion = False
        # priv_observe_dummy_variable = False
        
        priv_passthrough = False

    class robot(PrefixProto, cli=False):
        name = "go1"

    class ball(PrefixProto, cli=False):
        asset = "ball"
        mass = 0.318
        radius = 0.0889
        ball_init_pos = [0.0, 0.0, 0.50]
        ball_init_rot = [0, 0, 0, 1]
        ball_init_lin_vel = [0, 0, 0]
        ball_init_ang_vel = [0, 0, 0]
        init_pos_range = [1.0, 1.0, 0.2]
        init_vel_range = [0.5, 0.5, 0.3]
        pos_reset_prob = 0.0002
        vel_reset_prob = 0.0008
        pos_reset_range = [1.0, 1.0, 0.0]
        vel_reset_range = [0.3, 0.3, 0.3]
        vision_receive_prob = 0.7
        

    class sensors(PrefixProto, cli=False):
        sensor_names = ["OrientationSensor",
                        "RCSensor",
                        "JointPositionSensor",
                        "JointVelocitySensor",
                        "ActionSensor",
                        "ActionSensor",
                        "ClockSensor",
                        ]
        sensor_args = {"OrientationSensor": {},
                       "RCSensor": {},
                        "JointPositionSensor": {},
                        "JointVelocitySensor": {},
                        "ActionSensor": {},
                        "ActionSensor": {"delay": 1},
                        "ClockSensor": {}}
        
        privileged_sensor_names = []
        privileged_sensor_args = {}
        

    class terrain(PrefixProto, cli=False):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 0  # 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        terrain_noise_magnitude = 0.1
        # rough terrain only:
        terrain_smoothness = 0.005
        # measure_heights = True
        # 1mx1.6m rectangle (without center line)
        # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        min_init_terrain_level = 0
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        num_border_boxes = 0
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        difficulty_scale = 1.
        x_init_range = 1.
        y_init_range = 1.
        yaw_init_range = 0.
        x_init_offset = 0.
        y_init_offset = 0.
        teleport_robots = True
        teleport_thresh = 2.0
        max_platform_height = 0.2
        max_step_height = 0.26
        min_step_run = 0.25
        max_step_run = 0.4
        center_robots = False
        center_span = 5

    class commands(PrefixProto, cli=False):

        ############################
        # Inverse IK: door opening #
        ############################\
        p_gains_arm = []
        d_gains_arm = []
        p_gains_legs = []
        d_gains_legs = []
        b1_stand_up_joint_pos = [-0.1, 0.67, -1.3, 0.1, 0.67, -1.3, -0.1, 0.67, -1.3, 0.1, 0.67, -1.3]
        b1_stand_up_height = 0.54
        settling_time_force_gripper_s = 1.0
        
        force_control = False
        control_ee_ori_only_yaw = False
        control_ee_ori = False
        sample_feasible_commands = False 
        control_only_z1 = False 
        inverse_IK_door_opening = False      # Specify which commands to define in leggedrobot.py/_init_command_distribution 
        teleop_occulus = False
        interpolate_ee_cmds = False
        sample_feasible_commands = False
        settle_time = 2.0 # sec
        # end_effector_pos_x   = [-0.8, 0.8]   # min/max [m] in base frame 
        # end_effector_pos_y   = [-0.8, 0.8]   # min/max [m] in base frame 
        # end_effector_pos_z   = [0.2, 1.0]    # min/max [m] in base frame 
        end_effector_roll    = [0.0, 0.0]   # min/max [rad] in base frame (0.5 rad = 30 degrees)
        end_effector_pitch   = [0.0, 0.0]   # min/max [rad] in base frame 
        end_effector_yaw     = [0.0, 0.0]   # min/max [rad] in base frame 
        # end_effector_gripper = [-0.17, 0.17] # min/max [rad] (0.17 rad = 10 degrees)
        # # Limit 
        # limit_end_effector_pos_x   = [-0.8, 0.8]   # min/max [m] in base frame 
        # limit_end_effector_pos_y   = [-0.8, 0.8]   # min/max [m] in base frame 
        # limit_end_effector_pos_z   = [0.2, 1.0]    # min/max [m] in base frame 
        limit_end_effector_roll    = [0.0, 0.0]   # min/max [rad] in base frame (0.5 rad = 30 degrees)
        limit_end_effector_pitch   = [0.0, 0.0]   # min/max [rad] in base frame 
        limit_end_effector_yaw     = [0.0, 0.0]   # min/max [rad] in base frame 
        # limit_end_effector_gripper = [-0.17, 0.17] # min/max [rad] (0.17 rad = 10 degrees)
        # # Number of bins : np.linspace(min + bin_size / 2, max - bin_size / 2, nb_bins)
        # num_bins_end_effector_pos_x   = 20   # 1.6/20 = 8cm per bin
        # num_bins_end_effector_pos_y   = 20   # 1.6/20 = 8cm per bin
        # num_bins_end_effector_pos_z   = 10   # 0.8/10 = 8cm per bin
        num_bins_end_effector_roll    = 1    # 10 degrees per bin
        num_bins_end_effector_pitch   = 1    # 10 degrees per bin
        num_bins_end_effector_yaw     = 1    # 10 degrees per bin
        # num_bins_end_effector_gripper = 2    # 10 degrees per bin

        ##############
        ##############     
        ee_sphe_radius = [0.2, 0.7] # 
        ee_sphe_pitch = [-2*np.pi/5, 2*np.pi/5] # 
        ee_sphe_yaw = [-3*np.pi/5, 3*np.pi/5] # 
        ee_timing = [1.0, 3.0] #   

        limit_ee_sphe_radius = [0.2, 0.7] # 
        limit_ee_sphe_pitch = [-2*np.pi/5, 2*np.pi/5] # 
        limit_ee_sphe_yaw = [-3*np.pi/5, 3*np.pi/5] # 
        limit_ee_timing = [1.0, 3.0] #   

        num_bins_ee_sphe_radius = 1 # 
        num_bins_ee_sphe_pitch = 1 # 
        num_bins_ee_sphe_yaw = 1 # 
        num_bins_ee_timing = 1 # 
        
        ee_force_magnitude = [-120, 120]
        limit_ee_force_magnitude = [-120, 120]
        num_bins_ee_force_magnitude = 1 

        ee_force_z = [-0.0, 0.0]
        limit_ee_force_z = [-0.0, 0.0]
        num_bins_ee_force_z = 1 

        ee_force_direction_angle = [-np.pi, np.pi]
        limit_ee_force_direction_angle = [-np.pi, np.pi]
        num_bins_ee_force_direction_angle = 1 

        hybrid_mode = "position" # ["position", "force", "mixed"]

        command_curriculum = False
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        exclusive_command_sampling = False
        num_commands = 3
        resampling_time = 10.  # time before command are changed[s]
        subsample_gait = False
        gait_interval_s = 10.  # time between resampling gait params
        vel_interval_s = 10.
        jump_interval_s = 20.  # time between jumps
        jump_duration_s = 0.1  # duration of jump
        jump_height = 0.3
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False
        observe_accel = False
        distributional_commands = False
        curriculum_type = "RewardThresholdCurriculum"
        lipschitz_threshold = 0.9

        num_lin_vel_bins = 30
        lin_vel_step = 0.3
        num_ang_vel_bins = 30
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100

        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y = [-1.0, 1.0]  # min max [m/s]
        ang_vel_yaw = [-1, 1]  # min max [rad/s]
        body_height_cmd = [-0.05, 0.05]
        impulse_height_commands = False

        limit_vel_x = [-10.0, 10.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-10.0, 10.0]
        limit_body_height = [-0.05, 0.05]
        limit_gait_phase = [0, 0.01]
        limit_gait_offset = [0, 0.01]
        limit_gait_bound = [0, 0.01]
        limit_gait_frequency = [2.0, 2.01]
        limit_gait_duration = [0.49, 0.5]
        limit_footswing_height = [0.06, 0.061]
        limit_body_pitch = [0.0, 0.01]
        limit_body_roll = [0.0, 0.01]
        limit_aux_reward_coef = [0.0, 0.01]
        limit_compliance = [0.0, 0.01]
        limit_stance_width = [0.0, 0.01]
        limit_stance_length = [0.0, 0.01]

        num_bins_vel_x = 25
        num_bins_vel_y = 3
        num_bins_vel_yaw = 25
        num_bins_body_height = 1
        num_bins_gait_frequency = 11
        num_bins_gait_phase = 11
        num_bins_gait_offset = 2
        num_bins_gait_bound = 2
        num_bins_gait_duration = 3
        num_bins_footswing_height = 1
        num_bins_body_pitch = 1
        num_bins_body_roll = 1
        num_bins_aux_reward_coef = 1
        num_bins_compliance = 1
        num_bins_compliance = 1
        num_bins_stance_width = 1
        num_bins_stance_length = 1

        heading = [-3.14, 3.14]

        gait_phase_cmd_range = [0.0, 0.01]
        gait_offset_cmd_range = [0.0, 0.01]
        gait_bound_cmd_range = [0.0, 0.01]
        gait_frequency_cmd_range = [2.0, 2.01]
        gait_duration_cmd_range = [0.49, 0.5]
        footswing_height_range = [0.06, 0.061]
        body_pitch_range = [0.0, 0.01]
        body_roll_range = [0.0, 0.01]
        aux_reward_coef_range = [0.0, 0.01]
        compliance_range = [0.0, 0.01]
        stance_width_range = [0.0, 0.01]
        stance_length_range = [0.0, 0.01]

        exclusive_phase_offset = True
        binary_phases = False
        pacing_offset = False
        balance_gait_distribution = True
        gaitwise_curricula = True

    class curriculum_thresholds(PrefixProto, cli=False):
        tracking_lin_vel = 0.8  # closer to 1 is tighter
        tracking_ang_vel = 0.7
        tracking_contacts_shaped_force = 0.9  # closer to 1 is tighter
        tracking_contacts_shaped_vel = 0.9
        dribbling_ball_vel = 0.8

    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 0.54]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # target angles when action = 0.0
        default_joint_angles = {"joint_a": 0., "joint_b": 0.}

    class control(PrefixProto, cli=False):
        control_type = 'actuator_net' #'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        hip_scale_reduction = 1.0
        arm_scale_reduction = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(PrefixProto, cli=False):
        file = ""
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 1  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand(PrefixProto, cli=False):
        rand_interval_s = 10
        randomize_rigids_after_start = True
        
        # Force controller
        gripper_force_kp = 700.0
        gripper_force_kd = 6.0
        gripper_forced_prob = 0.5
        gripper_forced_rand_interval_s = 2.0
        gripper_motion_duration = 0.3
        
        gripper_force_kp_range = [200., 700.]
        gripper_force_kd_range = [3.0, 10.0]
        prop_kd = -1.0
        randomize_gripper_force_gains = False
        

        # types of randomization
        randomize_friction = True
        randomize_friction_indep = False
        randomize_ground_friction = False
        randomize_restitution = False
        randomize_ground_friction = False
        randomize_ground_restitution = False
        randomize_tile_roughness = False
        randomize_base_mass = False
        randomize_com_displacement = False
        randomize_motor_strength = False
        randomize_motor_offset = False
        randomize_Kp_factor = False
        randomize_Kd_factor = False
        randomize_gravity = False
        randomize_ball_drag = False
        randomize_ball_restitution = False
        randomize_ball_friction = False

        # randomization ranges
        friction_range = [0.5, 1.25]  # increase range
        restitution_range = [0., 1.0]
        ground_friction_range = [0., 1.0]
        ground_restitution_range = [0, 1.0]
        tile_roughness_range = [0.0, 0.1]
        added_mass_range = [-1., 1.]
        com_displacement_range = [-0.15, 0.15]
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [0.0, 0.0]
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]
        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0
        gravity_range = [-1.0, 1.0]
        drag_range = [0.0, 1.0]
        ball_drag_rand_interval_s = 15
        ball_restitution_range = [0.5, 1.0]
        ball_friction_range = [0.5, 1.0]
        
        # random pushes and parameters
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_lag_timesteps = True
        lag_timesteps = 6

        # Push gripper stator
        push_gripper_stators = False
        push_gripper_interval_s = [6.0, 9.0]
        push_gripper_duration_s = [3.0, 4.0] # has to be smaller than push_gripper_interval_s
        # push_gripper_duration_s = [1.0, 3.0] # has to be smaller than push_gripper_interval_s
        max_push_force_xyz_gripper = [-70, 70]
        max_push_vel_xyz_gripper = [0.0, 30.0]
        max_push_force_xyz_gripper_freed = [-120, 120]

        # Push robot base
        push_robot_base = False
        push_robot_interval_s = 3
        push_robot_duration_s = [1.0, 2.0] # has to be smaller than push_robot_interval_s
        max_push_vel_xyz_robot = [-200, 200]

    class rewards(PrefixProto, cli=False):
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_ji22_style = False
        sigma_rew_neg = 5
        reward_container_name = "CoRLRewards"
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_v_x = 0.25
        tracking_sigma_v_y = 0.25  
        tracking_sigma_v_yaw = 0.25      
        tracking_sigma_lat = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_long = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit_leg = 1.
        soft_torque_limit_arm = 1.
        termination_torque_min_time = 0
        base_height_target = 1.
        max_contact_force = 250.  # forces above this value are penalized
        min_contact_force = 30.
        use_terminal_body_height = False
        terminal_body_height = 0.20
        use_terminal_foot_height = False
        terminal_foot_height = -0.005
        use_terminal_roll_pitch = False
        use_terminal_torque_arm_limits = False
        use_terminal_torque_legs_limits = False
        terminal_body_ori = 0.5
        use_terminal_ee_position = False
        terminal_ee_distance = 0.0
        kappa_gait_probs = 0.07
        gait_force_sigma = 50.
        gait_vel_sigma = 0.5
        footswing_height = 0.09
        front_target = [[0.17, -0.09, 0]]
        estimation_bonus_dims = []
        estimation_bonus_weights = []
        manip_pos_tracking_coef = 15.0
        manip_ori_tracking_coef = 10.0
        stance_length = 0.85
        stance_width = 0.55
        
        sigma_force_magnitude = 1/70
        sigma_force_z = 1/70
        maintain_ori_force_envs = False
        
        constrict = False
        constrict_indices = []
        constrict_ranges = [[]]
        constrict_after = 0
        
        swing_ratio = 0.5
        stance_ratio = 0.5
        
        total_rew_scale = 1.0

    class reward_scales(ParamsProto, cli=False):

        ############################
        # Inverse IK: door opening #
        ############################
        dof_stand_up_pos_tracking = 0.0
        body_height_tracking = 0.0
        manip_pos_tracking_radius = 0.0
        manip_pos_tracking = 0.0
        manip_ori_tracking = 0.0
        manip_ori_tracking_yaw_only = 0.0
        manip_combo_tracking = 0.0
        manip_energy = 0.0
        loco_energy = 0.0
        tracking_lin_vel_x = 0.0
        tracking_lin_vel_y = 0.0
        tracking_ang_vel_yaw = 0.0
        alive = 0.0
        # end_effector_position_tracking = 0.0
        # end_effector_orientation_tracking = 0.0
        # body_height_tracking = 0.0
        # end_effector_pos_x_tracking = 0.0
        # end_effector_pos_y_tracking = 0.0
        # end_effector_pos_z_tracking = 0.0
        end_effector_ori_roll_tracking = 0.0
        end_effector_ori_pitch_tracking = 0.0
        end_effector_ori_yaw_tracking = 0.0
        
        ee_force_x = 0.0
        ee_force_y = 0.0
        ee_force_z = 0.0
        ee_force_magnitude = 0.0
        ee_force_direction_angle = 0.0
        ee_force_magnitude_x_pen = 0.0
        ee_force_magnitude_y_pen = 0.0
        ee_force_magnitude_z_pen = 0.0

        termination = -0.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        orientation = -0.
        torques = -0.00001
        torques_arm = 0.0
        dof_vel_arm = -0.
        dof_vel_leg = -0.
        arm_dof_vel = 0.
        dof_acc_arm = -2.5e-7
        dof_acc_leg = -2.5e-7
        arm_dof_acc = 0.
        base_height = -0.
        feet_air_time = 1.0
        collision = -1.
        feet_stumble = -0.0
        action_rate_arm = -0.01
        action_rate_leg = -0.01
        stand_still = -0.
        tracking_lin_vel_lat = 0.
        tracking_lin_vel_long = 0.
        tracking_contacts = 0.
        tracking_contacts_shaped = 0.
        tracking_contacts_shaped_force = 0.
        tracking_contacts_shaped_vel = 0.
        jump = 0.0
        energy = 0.0
        energy_expenditure = 0.0
        survival = 0.0
        dof_pos_limits_arm = 0.0
        dof_pos_limits_leg = 0.0
        dof_vel_limits = 0.0
        torque_limits_arm = 0.0
        torque_limits_leg = 0.0
        feet_contact_forces = 0.
        feet_slip = 0.
        feet_clearance_cmd = 0.
        feet_clearance_cmd_linear = 0.
        feet_accel = 0.
        dof_pos = 0.
        action_smoothness_1_arm = 0.
        action_smoothness_2_arm = 0.
        action_smoothness_1_leg = 0.
        action_smoothness_2_leg = 0.
        base_motion = 0.
        feet_impact_vel = 0.0
        raibert_heuristic = 0.0
        dribbling_robot_ball_vel = 0.0
        dribbling_robot_ball_pos = 0.0
        dribbling_ball_vel = 0.0
        dribbling_robot_ball_yaw = 0.0
        dribbling_ball_vel_norm = 0.0
        dribbling_ball_vel_angle = 0.0
        gripper_handle_pos = 0.0
        gripper_handle_height = 0.0
        turn_handle = 0.0
        open_door = 0.0
        robot_door_pos = 0.0
        robot_door_ori = 0.0
        estimation_bonus = 0.0
        bc = 0.0

    class normalization(PrefixProto, cli=False):
        clip_observations = 100.
        clip_actions = 100.

        friction_range = [0.05, 4.5]
        ground_friction_range = [0.05, 4.5]
        restitution_range = [0, 1.0]
        roughness_range= [0.0, 0.1]
        added_mass_range = [-1., 3.]
        com_displacement_range = [-0.1, 0.1]
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [-0.05, 0.05]
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]
        joint_friction_range = [0.0, 0.7]
        contact_force_range = [0.0, 50.0]
        contact_state_range = [0.0, 1.0]
        body_velocity_range = [-6.0, 6.0]
        foot_height_range = [0.0, 0.15]
        body_height_range = [0.0, 0.60]
        gravity_range = [-1.0, 1.0]
        motion = [-0.01, 0.01]
        stair_height_range = [0.0, 0.3]
        stair_run_range = [0.0, 0.5]
        stair_ori_range = [-3.14, 3.14]
        ball_velocity_range = [-5.0, 5.0]
        ball_drag_range = [0.0, 1.0]

    class obs_scales(PrefixProto, cli=False):
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        imu = 0.1
        height_measurements = 5.0
        friction_measurements = 1.0
        body_height_cmd = 2.0
        gait_phase_cmd = 1.0
        gait_freq_cmd = 1.0
        footswing_height_cmd = 0.15
        body_pitch_cmd = 0.3
        body_roll_cmd = 0.3
        aux_reward_cmd = 1.0
        compliance_cmd = 1.0
        stance_width_cmd = 1.0
        stance_length_cmd = 1.0
        segmentation_image = 1.0
        rgb_image = 1.0
        depth_image = 1.0
        ball_pos = 1.0

        ############################
        # Inverse IK: door opening #
        ############################
        
        # end_effector_pos_x_cmd = 1.0
        # end_effector_pos_y_cmd = 1.0
        # end_effector_pos_z_cmd = 0.7
        end_effector_roll_cmd = 0.5
        end_effector_pitch_cmd = 0.5
        end_effector_yaw_cmd = 0.5
        # end_effector_gripper_cmd = 0.25

        ee_sphe_radius_cmd = 0.5   # 0.2 - 0.7 
        ee_sphe_pitch_cmd = 1.0    # -1.3 , 1.3 
        ee_sphe_yaw_cmd = 1.3      # -1.9 , 1.9 
        ee_timing_cmd = 2.0        # 1.0, 3.0 # 
        
        ee_force_magnitude = 0.01
        ee_force_direction_angle = 0.3
        ee_force_z = 0.01

    class noise(PrefixProto, cli=False):
        add_noise = True
        noise_level = 1.0  # scales other values

    class noise_scales(PrefixProto, cli=False):
        dof_pos = 0.01
        dof_vel = 1.5
        lin_vel = 0.1
        ang_vel = 0.2
        imu = 0.1
        gravity = 0.05
        contact_states = 0.05
        height_measurements = 0.1
        friction_measurements = 0.0
        segmentation_image = 0.0
        rgb_image = 0.0
        depth_image = 0.0
        ball_pos = 0.05

    # viewer camera:
    class viewer(PrefixProto, cli=False):
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]
        follow_robot = False
        virtual_screen_capture = True

    class sim(PrefixProto, cli=False):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        use_gpu_pipeline = True

        class physx(PrefixProto, cli=False):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class perception(PrefixProto, cli=False):
        measure_heights = False
        compute_heights = False
        measure_frictions = False
        compute_frictions = False
        measure_roughnesses = False
        compute_roughnesses = False

        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        num_height_points = 187

        camera_names = ["front", "left", "right", "bottom", "rear"]
        camera_poses = [[0.3, 0, 0], [0, 0.15, 0], [0, -0.15, 0], [0.1, 0, -0.1], [-0.2, 0, -0.1]]
        camera_rpys = [[0.0, 0, 0], [0, 0, 3.14 / 2], [0, 0, -3.14 / 2], [0, -3.14 / 2, 0],
                       [0, -3.14 / 2, 0]]
        camera_gimbals = [False, False, False, False, False]
        compute_depth = False
        compute_rgb = False
        compute_segmentation = False
        # observe_depth = False
        # observe_rgb = False
        # observe_segmentation = False
        image_height = 100
        image_width = 100
        image_horizontal_fov = 110.0 # 110 degrees