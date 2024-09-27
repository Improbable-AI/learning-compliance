from typing import Union

from params_proto.neo_proto import Meta

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


def config_vision_platforms(Cnfg: Union[LeggedRobotCfg, Meta]):
    from legged_gym.envs.go1.go1_config import config_go1

    _ = Cnfg.terrain
    _.mesh_segmented = True
    _.terrain_length = 2.
    _.terrain_width = 2.
    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    _.max_platform_height = 0.1
    _.x_init_range = 0.
    _.y_init_range = 0.
    # set the gap width: 0 = no gaps
    _.curriculum = False
    _.selected = True
    _.gap_scale = 0.0 # gap width, m
    _.platform_width = 1.5
    _.difficulty_scale = 1.0

    _ = Cnfg.rewards
    # terminate when the robot falls off the platform
    _.use_terminal_body_height = True
    _.terminal_body_height = 0.
    _.use_terminal_roll_pitch = True
    _.terminal_body_ori = 0.5

    _ = Cnfg.perception
    _.observe_segmentation = True
    _.observe_rgb = True
    _.observe_depth = True
    _.image_width = 60
    _.image_height = 60