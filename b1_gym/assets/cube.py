from b1_gym.assets.asset import Asset
from isaacgym import gymapi

class Cube(Asset):
    def __init__(self, env):
        super().__init__(env)
        # self.reset()

    def initialize(self):
        ball_radius = self.env.cfg.ball.radius
        asset_options = gymapi.AssetOptions()
        asset = self.gym.create_box(self.env.sim, ball_radius, ball_radius, ball_radius, asset_options)
        rigid_shape_props = self.env.gym.get_asset_rigid_shape_properties(asset)

        self.num_bodies = self.env.gym.get_asset_rigid_body_count(asset)

        return asset, rigid_shape_props
    
    def get_force_feedback(self):
        return None

    def get_observation(self):
        robot_object_vec = self.env.root_states[self.env.object_actor_idxs, 0:3] - self.env.base_pos
        return robot_object_vec