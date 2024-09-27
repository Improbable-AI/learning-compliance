'''                                                                  
Evaluates the tracking performance of the IK controller for      
B1 with the Z1 arm for several base velocity commands                      
(v_x, v_y, v_yaw)                                                
                                                                    
1. Sends a velocity command and record the measured velocity    
2. Compute the velocity error (one scalar per vel type)          
3. Plots the error of the measured VS target velocity commands  
    (one point per velocity tuple (v_x, v_y, v_yaw), the color   
        specifies the error magnitude). One plot per vel type       
'''

import os
import sys 
import isaacgym
assert isaacgym
import torch
import numpy as np
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt
from isaacgym.torch_utils import *

from eval_utils import *
from typing import Tuple, List, Optional

sys.path.append('../')

########################################
############# PARAMETERS ###############
########################################
RUN_PATH = "robot-locomotion/b1-loco-z1-manip/runs/voq40aun" 

WEIGHTS_PATH = './eval_model'
NUM_ENVS = 9 #343                                                   # The number of environments define how many velocity commands are evaluated (1 command/env)
NUM_EVAL_STEPS = 2000                                             # Only 1/3 of it is used to compute the error magnitude 
TRANSIENT_LOWER_CLIP = 50
ERROR_TYPE = 'L1' # or L2

NB_VEL_POINTS = [3, 1, 3] #[7, 7, 7]

GAITS = {"pronking": [0, 0, 0],
         "trotting": [0.5, 0, 0],
         "bounding": [0, 0.5, 0],
         "pacing"  : [0, 0, 0.5]}

# Fixed commands
body_height_cmd = 0.0
step_frequency_cmd = 2.0
gait = torch.tensor(GAITS["trotting"]) 
gait_duration_cmd = 0.5
footswing_height_cmd = 0.2
pitch_cmd = 0.0
roll_cmd = 0.0
stance_width_cmd = 0.6
stance_length_cmd = 0.65
end_effector_gripper_command = 0.0

########################################
############# PARAMETERS ###############
########################################

from utils import load_env, load_policy

def play(headless: bool = False):
    
    # Load simulation environment and policy (action = policy(obs_hist))
    env, policy = load_env(run_path = RUN_PATH,
                           weights_path = WEIGHTS_PATH,
                           sim_device = 'cuda:2', 
                           num_envs = NUM_ENVS, 
                           headless = headless)
    
    # add floating cameras
    from b1_gym.sensors.floating_camera_sensor import FloatingCameraSensor
    cameras = []
    for i in range(NUM_ENVS):
        cameras += [FloatingCameraSensor(env, env_idx=i)]
    
    
    # Get first observation
    obs = env.reset()
    actions = torch.zeros((NUM_ENVS, env.cfg.env.num_actions), device=env.device)

    import imageio
    mp4_writer = imageio.get_writer('video.mp4', fps=30) 
    torque_plots = []

    # Send all the commands for NUM_EVAL_STEPS
    for i in tqdm(range(NUM_EVAL_STEPS)):
        with torch.no_grad():
            actions = policy(obs)

        obs, _, _, _ = env.step(actions)

        # render all envs
        env.render()
        camera_images = []
        for k, camera in enumerate(cameras):
            camera.set_position()
            camera_image = camera.get_observation()[:, :, :3]
            camera_image = cv2.resize(camera_image, (480, 360))
            camera_images += [camera_image]

        # tile the images in 4xn grid
        tiled_image = np.concatenate(camera_images, axis=0)
        tiled_torque_plot = np.concatenate(torque_plots, axis=0)
        tiled_image = np.concatenate([tiled_image, tiled_torque_plot], axis=1)


        mp4_writer.append_data(tiled_image)
        
        
    mp4_writer.close()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    overall_eval(headless = False)
