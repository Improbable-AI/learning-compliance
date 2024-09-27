import os
import sys
import glob
import yaml
import wandb
import torch
from typing import Callable, Tuple 
import pickle as pkl

from isaacgym.torch_utils import *
from isaacgym import gymapi, gymutil
from typing import List

sys.path.append('../')

from b1_gym.envs import *
from b1_gym.envs.base.legged_robot_config import Cfg
from b1_gym.envs.b1.b1_plus_z1_config import config_b1_plus_z1
from b1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from b1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from b1_gym_learn.ppo_cse.actor_critic import ActorCritic

def load_policy(run, run_path: str, weights_path: str) -> Callable:
    '''
    1. Loads the latest policy and adaptation module weights in the temporary directory /tmp/legged_data
    2. Initialize these networks with these weights 

    Arguments:
        - runpath:      path of the run in wandb (click on info for a run to retrieve it) 
        - weights_path: local path where the weights are stored locally

    Returns: 
        - The function used to generate actions given the obs history
    '''
    wandb_path = 'tmp/legged_data/'
    # replace file locally if it already exists, root: A string specifying the root directory where the downloaded file should be stored. 
    body_file = run.file(wandb_path + 'body_latest.jit').download(replace=True, root=weights_path) 
    body = torch.jit.load(body_file.name)

    #adaptation_module_file = wandb.restore(weights_path + 'adaptation_module_latest.jit', run_path=run_path)
    adaptation_module_file = run.file(wandb_path + 'adaptation_module_latest.jit').download(replace=True, root=weights_path) 
    adaptation_module = torch.jit.load(adaptation_module_file.name)

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

def load_env(run_path: str, weights_path: str, sim_device: str = 'cuda:0', num_envs: int = 1, headless: bool = False, fix_base: bool = False, teleop: bool = False, interpolate_ee_cmds: bool = True, sample_feasible_commands: bool = False, control_only_z1: bool = False):
    '''
    1. Load the parameters and weights of a wandb run
    2. Initialize the simulation parameters with these weights.
    3. Turn off all the domain randomization parameters.
    4. Create the environment using these params.
    5. Load the policy.

    Arguments:
        - run_path: Path of the run in wandb (click on info for a run to retrieve it) 
        - num_envs: Number of environments to create 
        - headless: Boolean to specify if rendering should be on (False = rendering on)
    
    Returns: 
        - The function used to generate actions given the obs history
    '''
    
    if run_path is not None:
        # test mode
        api = wandb.Api()
        run = api.run(run_path)

        # Default config for all robots
        config_b1_plus_z1(Cfg)

        all_cfg = run.config
        cfg = all_cfg["Cfg"]

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)
                    
    else:
        # play mode
        config_b1_plus_z1(Cfg)


    # # # Turn off DR for evaluation script
    # Cfg.domain_rand.push_robots = False
    # Cfg.domain_rand.push_gripper_stators = False
    # Cfg.domain_rand.push_robot_base = False
    # Cfg.domain_rand.randomize_friction = False
    # Cfg.domain_rand.randomize_gravity = False
    # Cfg.domain_rand.randomize_restitution = False
    # Cfg.domain_rand.randomize_motor_offset = False
    # Cfg.domain_rand.randomize_motor_strength = False
    # Cfg.domain_rand.randomize_friction_indep = False
    # Cfg.domain_rand.randomize_ground_friction = False
    # Cfg.domain_rand.randomize_base_mass = False
    # Cfg.domain_rand.randomize_Kd_factor = False
    # Cfg.domain_rand.randomize_Kp_factor = False
    # Cfg.domain_rand.randomize_joint_friction = False
    # Cfg.domain_rand.randomize_com_displacement = False
    # Cfg.domain_rand.randomize_tile_roughness = True
    # Cfg.domain_rand.tile_roughness_range = [0.0, 0.0]
    # Cfg.domain_rand.ground_friction_range = [2.0, 2.01]
    # Cfg.robot.name = "b1_plus_dismounted_z1"

    # Cfg.noise.noise_level = 0

    # Define env params 
    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = num_envs
    Cfg.env.episode_length_s = 10000
    Cfg.terrain.num_rows = 10
    Cfg.terrain.num_cols = 10
    Cfg.terrain.border_size = 0
    Cfg.terrain.num_border_boxes = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True
    # Cfg.terrain.mesh_type = "boxes_tm"

    Cfg.asset.fix_base_link = fix_base
    Cfg.commands.teleop_occulus = teleop
    Cfg.commands.interpolate_ee_cmds = interpolate_ee_cmds
    Cfg.commands.control_only_z1 = control_only_z1

    Cfg.env.recording_height_px = 720
    Cfg.env.recording_width_px = 1280
    
    Cfg.env.record_video = True

    # Create env
    env = VelocityTrackingEasyEnv(sim_device=sim_device, headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    if run_path is not None:
        # Load policy
        policy = load_policy(run, 
                             run_path=run_path,
                             weights_path=weights_path)
    else:
        # set the dummy policy
        policy = lambda x: torch.zeros((num_envs, 19), device=sim_device)

    return env, policy