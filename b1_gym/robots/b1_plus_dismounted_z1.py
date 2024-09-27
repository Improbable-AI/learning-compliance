from .robot import Robot

from b1_gym import MINI_GYM_ROOT_DIR
import os

from isaacgym import gymapi

class B1PlusDismountedZ1(Robot):
    def __init__(self, env):
        super().__init__(env)

    def initialize(self):
        b1_path = '{MINI_GYM_ROOT_DIR}/resources/robots/b1/urdf/b1_plus_dismounted_z1.urdf'.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(b1_path)
        asset_file = os.path.basename(b1_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.env.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.env.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.env.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.env.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.env.cfg.asset.fix_base_link
        asset_options.density = self.env.cfg.asset.density
        asset_options.angular_damping = self.env.cfg.asset.angular_damping
        asset_options.linear_damping = self.env.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.env.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.env.cfg.asset.max_linear_velocity
        asset_options.armature = self.env.cfg.asset.armature
        asset_options.thickness = self.env.cfg.asset.thickness
        asset_options.disable_gravity = self.env.cfg.asset.disable_gravity
        asset_options.vhacd_enabled = False
        # asset_options.vhacd_params = gymapi.VhacdParams()
        # asset_options.vhacd_params.resolution = 500000

        asset = self.env.gym.load_asset(self.env.sim, asset_root, asset_file, asset_options)

        self.num_dof = self.env.gym.get_asset_dof_count(asset)
        self.num_actuated_dof = 19
        self.num_bodies = self.env.gym.get_asset_rigid_body_count(asset)
        dof_props_asset = self.env.gym.get_asset_dof_properties(asset)
        rigid_shape_props_asset = self.env.gym.get_asset_rigid_shape_properties(asset)

        return asset, dof_props_asset, rigid_shape_props_asset