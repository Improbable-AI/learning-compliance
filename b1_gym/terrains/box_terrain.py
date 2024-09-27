from .terrain import Terrain
from b1_gym.utils.terrain import perlin

from isaacgym import gymapi
import numpy as np
import torch
import os

class BoxTerrain(Terrain):
    def __init__(self, env):
        super().__init__(env)
        self.prepare()

    def prepare(self):
        """ Adds a box terrain to the simulation, sets parameters based on the cfg.
        # """
        from isaacgym import terrain_utils

        self.terrain_cell_bounds = {
            "x_min": torch.zeros(self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols, device=self.env.device),
            "y_min": torch.zeros(self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols, device=self.env.device),
            "x_max": torch.zeros(self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols, device=self.env.device),
            "y_max": torch.zeros(self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols, device=self.env.device)
        }
        min_friction, max_friction = self.env.cfg.domain_rand.ground_friction_range
        self.terrain_cell_frictions = torch.rand(self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols,
                                                 dtype=torch.float, device=self.env.device,
                                                 requires_grad=False) * (
                                              max_friction - min_friction) + min_friction

        min_restitution, max_restitution = self.env.cfg.domain_rand.ground_restitution_range
        self.terrain_cell_restitutions = torch.rand(self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols,
                                                 dtype=torch.float, device=self.env.device,
                                                 requires_grad=False) * (
                                              max_restitution - min_restitution) + min_restitution

        min_roughness, max_roughness = self.env.cfg.domain_rand.tile_roughness_range
        self.terrain_cell_roughnesses = torch.rand(self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols,
                                                 dtype=torch.float, device=self.env.device,
                                                 requires_grad=False) * (
                                              max_roughness - min_roughness) + min_roughness

        lengthwise_density = int(self.env.cfg.terrain.terrain_length / self.env.cfg.terrain.horizontal_scale)
        widthwise_density = int(self.env.cfg.terrain.terrain_width / self.env.cfg.terrain.horizontal_scale)

        self.stair_heights_samples = torch.zeros(self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols,
                                                 dtype=torch.float, device=self.env.device,
                                                 requires_grad=False).repeat_interleave(
                                                 lengthwise_density, dim=0).repeat_interleave(widthwise_density, dim=1)

        self.stair_runs_samples = torch.zeros(self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols,
                                                 dtype=torch.float, device=self.env.device,
                                                 requires_grad=False).repeat_interleave(
                                                 lengthwise_density, dim=0).repeat_interleave(widthwise_density, dim=1)

        self.stair_oris_samples = torch.zeros(self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols,
                                                 dtype=torch.float, device=self.env.device,
                                                 requires_grad=False).repeat_interleave(
                                                 lengthwise_density, dim=0).repeat_interleave(widthwise_density, dim=1)

        self.env.min_height, self.env.max_height = 0, 0
        self.terrain_cell_heights = torch.rand(self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols,
                                               dtype=torch.float, device=self.env.device,
                                               requires_grad=False) * (
                                            self.env.max_height - self.env.min_height) + self.env.min_height

        self.terrain_cell_center_heights = torch.zeros(self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols, dtype=torch.float, device=self.env.device, requires_grad=False)

        lengthwise_density = int(self.env.cfg.terrain.terrain_length / self.env.cfg.terrain.horizontal_scale)
        widthwise_density = int(self.env.cfg.terrain.terrain_width / self.env.cfg.terrain.horizontal_scale)
        self.height_samples = torch.tensor(self.terrain_cell_heights).view(self.env.cfg.terrain.num_rows,
                                                                           self.env.cfg.terrain.num_cols).repeat_interleave(
            lengthwise_density, dim=0).repeat_interleave(widthwise_density, dim=1).to(
            self.env.device) / self.env.cfg.terrain.vertical_scale

        n_border_boxes = self.env.cfg.terrain.num_border_boxes

        for i in range(self.env.cfg.terrain.num_rows):
            for j in range(self.env.cfg.terrain.num_cols):
                # border_px = int(self.env.cfg.terrain.border_size / self.env.cfg.terrain.horizontal_scale)
                start_px = i * self.env.terrain.cfg.width_per_env_pixels
                end_px = (i + 1) * self.env.terrain.cfg.width_per_env_pixels
                start_py = j * self.env.terrain.cfg.length_per_env_pixels
                end_py = (j + 1) * self.env.terrain.cfg.length_per_env_pixels
                heightfield_segment_raw = self.env.terrain.height_field_raw[start_px:end_px + 1, start_py:end_py + 1]
                horizontal_scale = self.env.cfg.terrain.horizontal_scale  # / self.env.cfg.terrain.num_rows
                vertical_scale = self.env.cfg.terrain.vertical_scale  # / self.env.cfg.terrain.num_cols

                tm_params = gymapi.TriangleMeshParams()
                vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(heightfield_segment_raw,
                                                                                   horizontal_scale,
                                                                                   vertical_scale,
                                                                                   self.env.cfg.terrain.slope_treshold)
                tm_params.nb_vertices = vertices.shape[0]
                tm_params.nb_triangles = triangles.shape[0]

                tm_params.transform.p.x = i * self.env.terrain.cfg.env_width
                tm_params.transform.p.y = j * self.env.terrain.cfg.env_length
                tm_params.transform.p.z = 0 #self.env.terrain_cell_heights[i, j]

                if self.env.terrain_props is not None:
                    self.terrain_cell_frictions[i, j] = self.env.terrain_props["friction"][i, j]
                min_friction, max_friction = self.env.cfg.domain_rand.ground_friction_range
                tm_params.static_friction = self.terrain_cell_frictions[i, j]
                tm_params.dynamic_friction = self.terrain_cell_frictions[i, j]
                tm_params.restitution = self.terrain_cell_restitutions[i, j]
                
                tm_params.segmentation_id = int(
                    (tm_params.static_friction - min_friction) / (max_friction - min_friction) * 255)

                self.terrain_cell_bounds["x_min"][i, j] = i * self.env.terrain.cfg.env_width
                self.terrain_cell_bounds["y_min"][i, j] = j * self.env.terrain.cfg.env_length
                self.terrain_cell_bounds["y_min"][i, j] = j * self.env.terrain.cfg.env_length
                self.terrain_cell_bounds["x_max"][i, j] = (i + 1) * self.env.terrain.cfg.env_width
                self.terrain_cell_bounds["y_max"][i, j] = (j + 1) * self.env.terrain.cfg.env_length

                
                if self.env.terrain_props is not None:
                    if self.env.cfg.terrain.curriculum:
                        terrain_type = self.env.terrain_props["type"][i, j]
                        difficulty = self.env.cfg.terrain.difficulty_scale
                    else:
                        terrain_type = self.env.terrain_props["type"][i, j]
                        difficulty = self.env.cfg.terrain.difficulty_scale
                    if i < n_border_boxes or i >= self.env.cfg.terrain.num_rows - n_border_boxes or j < n_border_boxes or j >= self.env.cfg.terrain.num_cols - n_border_boxes:
                        terrain_type = 0
                        self.terrain_cell_roughnesses[i, j] = 0.0
                        self.terrain_cell_frictions[i, j] = 3.0
                else:
                    if self.env.cfg.terrain.curriculum:
                        terrain_type = (j - n_border_boxes) / (self.env.cfg.terrain.num_cols - 2 * n_border_boxes) + 0.001
                        difficulty = (i - n_border_boxes) / (self.env.cfg.terrain.num_rows - 2 * n_border_boxes) * self.env.cfg.terrain.difficulty_scale
                    else:
                        terrain_type = np.random.random()
                        difficulty = self.env.cfg.terrain.difficulty_scale * np.random.random()
                    if i < n_border_boxes or i >= self.env.cfg.terrain.num_rows - n_border_boxes or j < n_border_boxes or j >= self.env.cfg.terrain.num_cols - n_border_boxes:
                        terrain_type = 0
                        self.terrain_cell_roughnesses[i, j] = 0.0
                        self.terrain_cell_frictions[i, j] = 3.0
                cume_props = np.cumsum(self.env.cfg.terrain.terrain_proportions)

                if self.env.custom_heightmap is not None:
                    self.terrain_cell_center_heights[i, j] = self.env.custom_heightmap[(start_px+end_px)//2, (start_py+end_py)//2]
                    self.height_samples[start_px:end_px, start_py:end_py] = self.env.custom_heightmap[start_px:end_px, start_py:end_py] / self.env.cfg.terrain.vertical_scale
                else:
                    if terrain_type <= cume_props[0]: # flat/rough
                        # self.env.height_samples[start_px:end_px, start_py:end_py] = 0
                        self.terrain_cell_center_heights[i, j] = 0
                    elif terrain_type <= cume_props[1]: # downstairs
                        if self.env.terrain_props is not None:
                            step_run = self.env.terrain_props["step_run"][i, j]
                            step_height = self.env.terrain_props["step_height"][i, j]
                        else:
                            step_run = (self.env.cfg.terrain.max_step_run - self.env.cfg.terrain.min_step_run) * torch.rand(1) + self.env.cfg.terrain.min_step_run
                            step_height = self.env.cfg.terrain.max_step_height * difficulty
                        num_steps = 8
                        for k in range(num_steps):
                            step_px =  int(step_run * k / self.env.cfg.terrain.horizontal_scale)
                            if self.env.terrain.cfg.length_per_env_pixels - 1.0 / self.env.cfg.terrain.horizontal_scale <= step_px * 2:
                                k -= 1
                                break
                            height = k * step_height
                            self.height_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = height / self.env.cfg.terrain.vertical_scale
                            if k < num_steps - 1:
                                self.stair_heights_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = abs(step_height)
                                self.stair_runs_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = step_run
                                self.stair_oris_samples[start_px:start_px+step_px, start_py+step_px:end_py-step_px] = 0
                                self.stair_oris_samples[end_px-step_px:end_px, start_py+step_px:end_py-step_px] = np.pi
                                self.stair_oris_samples[start_px+step_px:end_px-step_px, end_py-step_px:end_py] = -np.pi/2
                                self.stair_oris_samples[start_px+step_px:end_px-step_px, start_py:start_py+step_px] = np.pi/2
                            else:
                                self.stair_heights_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = 0
                                self.stair_runs_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = 0
                                self.stair_oris_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = 0
                        
                        step_px = int(step_run * (k) / self.env.cfg.terrain.horizontal_scale)
                        self.stair_heights_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = 0
                        self.stair_runs_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = 0
                        self.stair_oris_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = 0

                        self.terrain_cell_center_heights[i, j] = height

                    elif terrain_type <= cume_props[2]:  # upstairs
                        if self.env.terrain_props is not None:
                            step_run = self.env.terrain_props["step_run"][i, j]
                            step_height = self.env.terrain_props["step_height"][i, j]
                        else:
                            step_run = (self.env.cfg.terrain.max_step_run - self.env.cfg.terrain.min_step_run) * torch.rand(1) + self.env.cfg.terrain.min_step_run
                            step_height = -self.env.cfg.terrain.max_step_height * difficulty
                        num_steps = 8
                        for k in range(num_steps):
                            step_px = int(step_run * k / self.env.cfg.terrain.horizontal_scale)
                            if self.env.terrain.cfg.length_per_env_pixels - 1.0 / self.env.cfg.terrain.horizontal_scale  <= step_px * 2:
                                k -= 1
                                break
                            height = k * step_height
                            self.height_samples[start_px + step_px:end_px - step_px, start_py + step_px:end_py - step_px] = height / self.env.cfg.terrain.vertical_scale
                            self.stair_heights_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = abs(step_height)
                            self.stair_runs_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = step_run
                            self.stair_oris_samples[start_px:start_px+step_px, start_py+step_px:end_py-step_px] = np.pi
                            self.stair_oris_samples[end_px-step_px:end_px, start_py+step_px:end_py-step_px] = 0
                            self.stair_oris_samples[start_px+step_px:end_px-step_px, end_py-step_px:end_py] = np.pi/2
                            self.stair_oris_samples[start_px+step_px:end_px-step_px, start_py:start_py+step_px] = -np.pi/2

                        step_px = int(step_run * (k) / self.env.cfg.terrain.horizontal_scale)
                        self.stair_heights_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = 0
                        self.stair_runs_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = 0
                        self.stair_oris_samples[start_px+step_px:end_px-step_px, start_py+step_px:end_py-step_px] = 0

                        self.terrain_cell_center_heights[i, j] = height

                    elif terrain_type <= cume_props[3]:  # slope
                        slope = 0.3 #2 * difficulty
                        width = end_px - start_px
                        length = end_py - start_py
                        x = np.arange(0, width)
                        y = np.arange(0, length)

                        center_x = int(width / 2)
                        center_y = int(length / 2)
                        xx, yy = np.meshgrid(x, y, sparse=True)
                        xx = (center_x - np.abs(center_x - xx)) / center_x
                        yy = (center_y - np.abs(center_y - yy)) / center_y
                        xx = xx.reshape(width, 1)
                        yy = yy.reshape(1, length)
                        max_height = int(slope * (self.env.cfg.terrain.horizontal_scale / self.env.cfg.terrain.vertical_scale) * (width / 2))
                        self.height_samples[start_px:end_px, start_py:end_py] = torch.tensor((max_height * xx * yy))

                        platform_size = 1.5

                        platform_size = int(platform_size / self.env.cfg.terrain.horizontal_scale / 2)
                        x1 = width // 2 - platform_size
                        x2 = width // 2 + platform_size
                        y1 = length // 2 - platform_size
                        y2 = length // 2 + platform_size

                        min_h = min(self.height_samples[start_px:end_px, start_py:end_py][x1, y1], 0)
                        max_h = max(self.height_samples[start_px:end_px, start_py:end_py][x1, y1], 0)
                        self.height_samples[start_px:end_px, start_py:end_py] = torch.clip(self.height_samples[start_px:end_px, start_py:end_py], -100, max_h)
                        self.terrain_cell_center_heights[i, j] = max_h * self.env.cfg.terrain.vertical_scale
                    elif terrain_type <= cume_props[4]:  # pit
                        self.height_samples[start_px:end_px, start_py:end_py] = -5.0 / self.env.cfg.terrain.vertical_scale
                        self.terrain_cell_center_heights[i, j] = 0

                # add perlin noise based on roughness
                if self.env.terrain_props is not None:
                    self.terrain_cell_roughnesses[i, j] = self.env.terrain_props["roughness"][i, j]
                roughness = self.terrain_cell_roughnesses[i, j]
                lin_x = np.linspace(0, self.env.cfg.terrain.terrain_length * 4, self.env.terrain.cfg.width_per_env_pixels,
                                    endpoint=False)
                lin_y = np.linspace(0, self.env.cfg.terrain.terrain_width * 4, self.env.terrain.cfg.length_per_env_pixels,
                                    endpoint=False)
                x, y = np.meshgrid(lin_x, lin_y)
                perlin_seed = i * self.env.cfg.terrain.num_cols + j
                self.height_samples[start_px:end_px, start_py:end_py] = self.height_samples[start_px:end_px, start_py:end_py] + torch.tensor(perlin(x, y, seed=perlin_seed), device=self.env.device) * float(roughness) / self.env.cfg.terrain.vertical_scale

                
        self.friction_samples = torch.tensor(self.terrain_cell_frictions).view(self.env.cfg.terrain.num_rows,
                                                                               self.env.cfg.terrain.num_cols).repeat_interleave(
            lengthwise_density, dim=0).repeat_interleave(widthwise_density, dim=1).to(self.env.device)
        self.roughness_samples = torch.tensor(self.terrain_cell_roughnesses).view(self.env.cfg.terrain.num_rows,
                                                                               self.env.cfg.terrain.num_cols).repeat_interleave(
            lengthwise_density, dim=0).repeat_interleave(widthwise_density, dim=1).to(self.env.device)

        self.height_samples[:10, :] = 3.0 / self.env.cfg.terrain.vertical_scale
        self.height_samples[-10:, :] = 3.0 / self.env.cfg.terrain.vertical_scale
        self.height_samples[:, :10] = 3.0 / self.env.cfg.terrain.vertical_scale
        self.height_samples[:, -10:] = 3.0 / self.env.cfg.terrain.vertical_scale


    def initialize(self):
        """ Adds a box terrain to the simulation, sets parameters based on the cfg.
        # """
        from isaacgym import terrain_utils

        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.armature = 0.01
        options.density = 1000
        options.use_mesh_materials = True
        options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        options.override_com = True
        options.override_inertia = True
        options.vhacd_enabled = True
        options.vhacd_params = gymapi.VhacdParams()
        options.vhacd_params.resolution = 3000000
        options.vhacd_params.max_num_vertices_per_ch = 1024
        options.vhacd_params.max_convex_hulls = 64
        options.vhacd_params.concavity = 0.0

        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        env = self.env.envs[0]  # self.gym.create_env(self.sim, env_lower, env_upper, 1)

        collision_group = -1
        collision_filter = -1
        base_asset_root = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../resources/objects/'
        base_asset_file = 'cube.urdf' #'cube_r08.urdf'
        self.base_asset = self.env.gym.load_asset(self.env.sim, base_asset_root, base_asset_file, options)

        texture_files = ['ice_texture.jpg', 'pebble_stone_texture_nature.jpg', 'particle_board_paint_aged.jpg', 'texture_stone_stone_texture_0.jpg', 'texture_wood_brown_1033760.jpg', 'brick_texture.jpg']
        texture_files = ['tiled_snow.jpg', 'tiled_pebble_stone_texture_nature.jpg', 'tiled_grass.jpg', 'tiled_brick_texture.jpg']
        texture_paths = [
            f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../resources/textures/{texture_file}'
            for texture_file in texture_files]
        textures = [self.env.gym.create_texture_from_file(self.env.sim, texture_path) for texture_path in texture_paths]

        self.env.terrain_actors = np.full((self.env.cfg.terrain.num_rows, self.env.cfg.terrain.num_cols), None)

        for i in range(self.env.cfg.terrain.num_rows):
            for j in range(self.env.cfg.terrain.num_cols):
                terrain_px = i * self.env.terrain.cfg.env_width + self.env.terrain.cfg.env_width / 2
                terrain_py = j * self.env.terrain.cfg.env_length + self.env.terrain.cfg.env_length / 2

                min_friction, max_friction = self.env.cfg.domain_rand.ground_friction_range
                static_friction = self.terrain_cell_frictions[i, j]
                dynamic_friction = self.terrain_cell_frictions[i, j]
                restitution = self.env.cfg.terrain.restitution
                segmentation_id = int(
                    (static_friction - min_friction) / (max_friction - min_friction) * 255)
                self.terrain_cell_bounds["x_min"][i, j] = i * self.env.terrain.cfg.env_width
                self.terrain_cell_bounds["y_min"][i, j] = j * self.env.terrain.cfg.env_length
                self.terrain_cell_bounds["x_max"][i, j] = (i + 1) * self.env.terrain.cfg.env_width
                self.terrain_cell_bounds["y_max"][i, j] = (j + 1) * self.env.terrain.cfg.env_length

                pose = gymapi.Transform()
                pose.r = gymapi.Quat(0, 0, 0, 1)
                pose.p = gymapi.Vec3(terrain_px, terrain_py,
                                     -self.env.terrain.cfg.env_width * 0.05 / 2 + self.terrain_cell_heights[i, j])  # , -self.base_depth / 2)

                props = self.env.gym.get_asset_rigid_shape_properties(self.base_asset)
                print(len(props))
                props[0].friction = self.terrain_cell_frictions[i, j]
                props[0].restitution = self.env.cfg.terrain.restitution
                self.env.gym.set_asset_rigid_shape_properties(self.base_asset, props)

                self.env.terrain_actors[i, j] = self.env.gym.create_actor(env, self.base_asset, pose, None, collision_group, collision_filter)
                self.env.gym.set_actor_scale(env, self.env.terrain_actors[i, j], self.env.terrain.cfg.env_width)
                # self.gym.set_rigid_body_color(env, base_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                #                          gymapi.Vec3(1, 1, 1))
                # print(len(textures), segmentation_id, segmentation_id // (255 // len(texture_files) + len(texture_files)))
                texture_idx = segmentation_id // (255 // len(texture_files) + len(texture_files))
                texture = textures[texture_idx % len(textures)]
                self.env.gym.set_rigid_body_texture(env, self.env.terrain_actors[i, j], 0, gymapi.MeshType.MESH_VISUAL, texture)

        lengthwise_density = int(self.env.cfg.terrain.terrain_length / self.env.cfg.terrain.horizontal_scale)
        widthwise_density = int(self.env.cfg.terrain.terrain_width / self.env.cfg.terrain.horizontal_scale)
        self.friction_samples = torch.tensor(self.terrain_cell_frictions).view(self.env.cfg.terrain.num_rows,
                                                                               self.env.cfg.terrain.num_cols).repeat_interleave(
            lengthwise_density, dim=0).repeat_interleave(widthwise_density, dim=1).to(self.env.device)
        self.roughness_samples = torch.tensor(self.terrain_cell_roughnesses).view(self.env.cfg.terrain.num_rows,
                                                                                 self.env.cfg.terrain.num_cols).repeat_interleave(
                lengthwise_density, dim=0).repeat_interleave(widthwise_density, dim=1).to(self.env.device)
