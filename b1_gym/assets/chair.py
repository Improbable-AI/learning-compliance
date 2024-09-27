from b1_gym.assets.asset import Asset
from isaacgym import gymapi
import os
from b1_gym import MINI_GYM_ROOT_DIR

class Chair(Asset):
    def __init__(self, env):
        super().__init__(env)
        # self.reset()

    def initialize(self):
        asset_path = '{MINI_GYM_ROOT_DIR}/resources/objects/chair/chair.urdf'.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 500000
        asset = self.env.gym.load_asset(self.env.sim, asset_root, asset_file, asset_options)
        rigid_shape_props = self.env.gym.get_asset_rigid_shape_properties(asset)

        self.num_bodies = self.env.gym.get_asset_rigid_body_count(asset)

        return asset, rigid_shape_props
    
    def get_force_feedback(self):
        return None

    def get_observation(self):
        robot_object_vec = self.env.root_states[self.env.object_actor_idxs, 0:3] - self.env.base_pos
        return robot_object_vec