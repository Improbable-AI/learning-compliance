from typing import Union

from params_proto.proto import Meta

from b1_gym.envs.base.legged_robot_config import Cfg


def config_z1(Cnfg: Union[Cfg, Meta]):
    _ = Cnfg.init_state

    _.pos = [0.0, 0.0, 0.70]  # x,y,z [m]
    _.default_joint_angles = {  # = target angles [rad] when action = 0.0
        # 'FL_hip_joint': 0.2,  # [rad]
        # 'RL_hip_joint': 0.2,  # [rad]
        # 'FR_hip_joint': -0.2,  # [rad]
        # 'RR_hip_joint': -0.2,  # [rad]

        # 'FL_thigh_joint': 0.6,  # [rad]
        # 'RL_thigh_joint': 1.0,  # [rad]
        # 'FR_thigh_joint': 0.6,  # [rad]
        # 'RR_thigh_joint': 1.0,  # [rad]

        # 'FL_calf_joint': -1.3,  # [rad]
        # 'RL_calf_joint': -1.3,  # [rad]
        # 'FR_calf_joint': -1.3,  # [rad]
        # 'RR_calf_joint': -1.3,  # [rad]

        'joint1': 0.0,
        'joint2': 1.5,
        'joint3': -1.5,
        'joint4': -0.54,
        'joint5': 0.0,
        'joint6': 0.0,
        'jointGripper': 0.0,
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
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/b1/urdf/z1.urdf'
    _.foot_name = "foot"
    _.penalize_contacts_on = []#, "base"]
    _.terminate_after_contacts_on = [] # ["base"]
    _.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    _.flip_visual_attachments = False
    _.fix_base_link = False

    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9
    _.base_height_target = 0.65

    _ = Cnfg.reward_scales
    _.torques = -0.0001
    _.action_rate = -0.01
    _.dof_pos_limits = -10.0
    _.orientation = -5.
    _.base_height = -30.

    _ = Cnfg.terrain
    _.mesh_type = 'trimesh'
    _.terrain_noise_magnitude = 0.0
    _.teleport_robots = True
    _.border_size = 50

    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    _.curriculum = False

    _ = Cnfg.env
    _.num_observations = 42
    _.observe_vel = False
    _.num_actions = 7
    _.num_envs = 4000

    _ = Cnfg.commands
    _.lin_vel_x = [-1.0, 1.0]
    _.lin_vel_y = [-1.0, 1.0]

    _ = Cnfg.commands
    _.heading_command = False
    _.resampling_time = 10.0
    _.command_curriculum = True
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30
    _.lin_vel_x = [-0.6, 0.6]
    _.lin_vel_y = [-0.6, 0.6]
    _.ang_vel_yaw = [-1, 1]

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

    # _ = Cnfg.perception
    # _.measure_heights = False
