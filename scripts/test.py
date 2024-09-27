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

from typing import Tuple, List, Optional

sys.path.append('./')

########################################
############# PARAMETERS ###############
########################################

NUM_ENVS = 10                                                  
NUM_EVAL_STEPS = 2000   
RECORD_VIDEO = False       

########################################
############# PARAMETERS ###############
########################################

from utils import load_env

def test(headless: bool = False):
    
    # Load simulation environment and policy (action = policy(obs_hist))
    env, policy = load_env(run_path = None,
                           weights_path = None,
                           sim_device = 'cuda:3', 
                        #    sim_device = 'cpu', 
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


    if RECORD_VIDEO:
        import imageio
        mp4_writer = imageio.get_writer('video.mp4', fps=30) 
        torque_plots = []

    # Send all the commands for NUM_EVAL_STEPS
    for i in tqdm(range(NUM_EVAL_STEPS)):
        with torch.no_grad():
            actions = policy(obs)

        obs, _, _, _ = env.step(actions)


        if RECORD_VIDEO:
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

            mp4_writer.append_data(tiled_image)
        
        
    if RECORD_VIDEO:
        mp4_writer.close()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    test(headless = False)
