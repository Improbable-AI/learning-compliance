from .sensor import Sensor
from .attached_camera_sensor import AttachedCameraSensor
from .floating_camera_sensor import FloatingCameraSensor
from .joint_position_sensor import JointPositionSensor
from .joint_velocity_sensor import JointVelocitySensor
from .joint_position_target_sensor import JointPositionTargetSensor
from .orientation_sensor import OrientationSensor
from .heightmap_sensor import HeightmapSensor
from .rc_sensor import RCSensor
from .action_sensor import ActionSensor
from .last_action_sensor import LastActionSensor
from .clock_sensor import ClockSensor
from .yaw_sensor import YawSensor
from .object_sensor import ObjectSensor
from .timing_sensor import TimingSensor
from .body_velocity_sensor import BodyVelocitySensor
from .object_velocity_sensor import ObjectVelocitySensor
from .restitution_sensor import RestitutionSensor
from .friction_sensor import FrictionSensor
from .ground_friction_sensor import GroundFrictionSensor
from .ground_roughness_sensor import GroundRoughnessSensor
from .egomotion_sensor import EgomotionSensor
from .ee_gripper_force_sensor import EeGripperForceSensor
from .ee_base_force_sensor import EeBaseForceSensor
from .ee_gripper_force_dir_sensor import EeGripperForceDirSensor
from .ee_gripper_force_magn_sensor import EeGripperForceMagnSensor
from .joint_dynamics_sensor import JointDynamicsSensor
from .ee_gripper_position_sensor import EeGripperPositionSensor
from .ee_gripper_target_position_sensor import EeGripperTargetPositionSensor

ALL_SENSORS = { "AttachedCameraSensor": AttachedCameraSensor,
                "FloatingCameraSensor": FloatingCameraSensor,
                "JointPositionSensor": JointPositionSensor,
                "JointVelocitySensor": JointVelocitySensor,
                "JointPositionTargetSensor": JointPositionTargetSensor,
                "OrientationSensor": OrientationSensor,
                "HeightmapSensor": HeightmapSensor,
                "RCSensor": RCSensor,
                "ActionSensor": ActionSensor,
                "LastActionSensor": LastActionSensor,
                "ClockSensor": ClockSensor,
                "YawSensor": YawSensor,
                "ObjectSensor": ObjectSensor,
                "TimingSensor": TimingSensor,
                "BodyVelocitySensor": BodyVelocitySensor,
                "ObjectVelocitySensor": ObjectVelocitySensor,
                "RestitutionSensor": RestitutionSensor,
                "FrictionSensor": FrictionSensor,
                "GroundFrictionSensor": GroundFrictionSensor,
                "GroundRoughnessSensor": GroundRoughnessSensor,
                "EgomotionSensor": EgomotionSensor,
                "EeGripperForceSensor": EeGripperForceSensor,
                "EeGripperForceMagnSensor": EeGripperForceMagnSensor,
                "EeGripperForceDirSensor": EeGripperForceDirSensor,
                "EeBaseForceSensor": EeBaseForceSensor,
                "JointDynamicsSensor": JointDynamicsSensor,
                "EeGripperPositionSensor": EeGripperPositionSensor,
                "EeGripperTargetPositionSensor": EeGripperTargetPositionSensor,
                }
