from b1_gym.assets.asset import Asset
from isaacgym import gymapi
import os
from b1_gym import MINI_GYM_ROOT_DIR
import torch

class Door(Asset):
    def __init__(self, env):
        super().__init__(env)
        # self.reset()

    def initialize(self):
        door_asset_path = '{MINI_GYM_ROOT_DIR}/resources/objects/door/lever_baxter_leftarm.urdf'.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        door_asset_root = os.path.dirname(door_asset_path)
        door_asset_file = os.path.basename(door_asset_path)
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        asset = self.env.gym.load_asset(self.env.sim, door_asset_root, door_asset_file, asset_options)
        rigid_shape_props = self.env.gym.get_asset_rigid_shape_properties(asset)
        self.num_dof = self.env.gym.get_asset_dof_count(asset)
        # self.ball_force_feedback = self._door_feedback
        
        self.num_bodies = self.env.gym.get_asset_rigid_body_count(asset)
        
        return asset, rigid_shape_props
    
    def get_force_feedback(self):
        def compute_torques():
            door_joint_pos = self.env.dof_pos[:, self.env.num_actuated_dof:]
            door_joint_vel = self.env.dof_vel[:, self.env.num_actuated_dof:]

            door_hinge_kd = 3.0
            door_hinge_kp = 500 # when locked
            door_handle_kp = 15
            door_handle_kd = 0.5

            door_hinge_max_torque = 3.0
            door_handle_max_torque = 3.0

            door_hinge_torque = -door_hinge_kd * door_joint_vel[:, 0]
            door_hinge_torque = torch.clamp(door_hinge_torque, -door_hinge_max_torque, door_hinge_max_torque)

            door_handle_torque = -door_handle_kp * door_joint_pos[:, 1] - door_handle_kd * door_joint_vel[:, 1]
            door_handle_torque = torch.clamp(door_handle_torque, -door_handle_max_torque, door_handle_max_torque)

            # if door handle is not turned, lock the door
            locked_doors = torch.abs(door_joint_pos[:, 1]) < 0.3
            door_hinge_torque[locked_doors] += -door_hinge_kp * door_joint_pos[locked_doors, 0]

            torque = torch.cat((door_hinge_torque.unsqueeze(1), door_handle_torque.unsqueeze(1)), dim=1)
            force = None

            return torque, force
        
        return compute_torques

    def get_observation(self):
        handle_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.object_actor_handles[0], "handle") - self.env.num_bodies
        handle_pos_global = self.env.rigid_body_state_object.view(self.env.num_envs, -1, 13)[:,handle_idx,0:3].view(self.env.num_envs,3)
        robot_object_vec = handle_pos_global - self.env.base_pos
        return robot_object_vec