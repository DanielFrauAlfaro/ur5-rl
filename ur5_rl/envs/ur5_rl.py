#!/usr/bin/env python3

import gymnasium as gym
import pybullet as p
import pyb_utils
import numpy as np
from ur5_rl.resources.ur5 import UR5e as UR5
from ur5_rl.resources.utils import *
from ur5_rl.resources.object import Object
from ur5_rl.resources.table import Table
import matplotlib.pyplot as plt
from math import pi
import math
import time
from cv2 import aruco
import cv2 as cv
import random
from scipy.spatial.transform import Rotation
 
# Gym environment
class UR5Env(gym.Env):
    # Environment Metadata
    metadata = {'render_modes': ['DIRECT', 'GUI'],
                'render_fps': 60}  
  
    def __init__(self, render_mode="DIRECT"):    

        # --- Observation limit values ---
        self._q_limits = [np.array([-1.5, -3.1415, -3.1415, -3.1415, -3.1415, -6.2831]), np.array([1.5, 0.0, 0.0, 3.1415, 3.1415, 6.2831])]
        self._qd_limits = [np.ones(6) * -180, np.ones(6) * 180]
        self._qdd_limits = [np.ones(6) * -5000, np.ones(6) * 5000]
        self._ee_limits = [[-1, -1, -1, -pi, -pi, -pi, 0], [1, 1, 1, pi, pi, pi, 100]]

        self._limits = [self._q_limits,  
                       self._qd_limits,  
                       self._qdd_limits, 
                       self._ee_limits]

        # --- Action limits ---
        # Joint actions
        self.max_action = 0.01
        self._action_limits = [-np.ones(6)*self.max_action, np.ones(6)*self.max_action]
        
        # Appends gripper actions
        self.max_action_g = 2       # Max action G is two because the robot class converts it to integer
        self._action_limits[0] = np.append(self._action_limits[0], -self.max_action_g)
        self._action_limits[1] = np.append(self._action_limits[1],  self.max_action_g)

        # Frame height and width
        self.frame_h = 200
        self.frame_w = 200

        '''
        Box space in action space:
            - Robot joints and gripper: 6 continous robot joints + 1 continous gripper action 

        ''' 
        self.action_space = gym.spaces.box.Box(low=self._action_limits[0],
                               high= self._action_limits[1], dtype=np.float32)
        
        
        '''
        Dictionary of spaces in observation space:
            - Joint Positions: 6 robot end effector position
            - Camera image: grayscale and depth
        '''
        # Dictionary indices
        self._indices = ["ee_position", "gripper_position", "image"]

        self.observation_space = gym.spaces.Dict({
            self._indices[0]: gym.spaces.box.Box(low=np.float32(np.array(self._ee_limits[0])), 
                               high= np.float32(np.array(self._ee_limits[1])), dtype=np.float32),


            self._indices[-1]: gym.spaces.box.Box(low=-1, high=256, shape=(2, self.frame_w, self.frame_h), dtype=np.int16)
        })

        # Time limit of the episode (in seconds)
        self._t_limit = 8
        self._t_act = time.time()



        # Checks if the selected render mode is within the possibles
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Start seed
        self.np_random, __ = gym.utils.seeding.np_random()

        # Client in Pybullet simulation
        self._client = 0
        if render_mode == "DIRECT":
            self._client = p.connect(p.DIRECT)
        elif render_mode == "GUI":
            self._client = p.connect(p.GUI)
        
        # Object coordinates of spawning
        self.obj_pos = [0.2, 0.55, 0.9]

        # Image to be rendered
        self.frame = [np.ones((self.frame_w, self.frame_h), dtype=np.int16), np.ones((self.frame_w, self.frame_h), dtype=np.int16)]


        # Camera Parameters
        self.fov = 10
        self.near_plane = 0.02
        self.far_plane = 5.0
        self.aspect = 1


        # --- Cameras ---
        # Camera positions variables: [[Position], [Orientation]]

        # Coordinates of the cameras
        self.cameras_coord = [[[0.05, 0.95, 1.05], [0.6, 0.0, -pi/2]],                 # External Camera 1
                         [[-0.021, 0.456, 1.179], [3.134 + pi/2, -0.024, 1.58]]]       # Robot camera

        self.std_cam = 0.01
        self.camera_params = set_cam(client=self._client, fov=self.fov, aspect=self.aspect, 
                                     near_val=self.near_plane, far_val=self.far_plane, 
                                     cameras_coord = self.cameras_coord, std = self.std_cam)

        # Distance between object an wrist
        self._dist_obj_wrist = math.inf

        # Reward mask
        self.mask = np.array([-2, 2, 2, 2])

    
    # Computes the whole reward
    def compute_reward(self):
        '''
        Computes the environment reward according to the approximation reward
    and the collision reward

        Returns:
            - The reward (int / float)
        '''

        r = 0

        # Object approximation reward
        r, self._dist_obj_wrist = approx_reward(client = self._client, object = self._object, 
                                                dist_obj_wrist = self._dist_obj_wrist, robot_id = self._ur5.id)

        # Collision reward
        r += collision_reward(client = self._client, collisions_to_check = self.collisions_to_check, mask = self.mask)
            
        return r


    # Computes if the environmen has reach a terminal state
    def get_terminal(self):
        '''
        Checks if the environment has reached a terminal state
            - Terminated: the environment ends normally
            - Truncated: the environment ends with problems
                - End time
                - Out of bounds joints
                - Collision with the table
        
        Returns:
            - Two boolean, the terminated and truncated flag
        '''

        terminated = False
        truncated = (time.time() - self._t_act) > self._t_limit \
                    or out_of_bounds(self._limits, self._ur5) \
                    or check_collision(client = self._client, objects = [self._table.id, self._ur5.id])

        return terminated, truncated


    # Getter for the observations
    def get_observation(self):
        '''
        Obtains the observation in the environment in a dictionary
        '''

        # Gets starting observation of the robot
        observation = self._ur5.get_observation()       # [q, qd, gripper, ee]
        
        # Stores joint velocities
        self.qd = observation[1]

        # Arranges observation vectors into a dictionary
        obs = {}

        # Appends the gripper observation to the end - effector observation
        observation[-1].append(observation[2])

        # Assigns robot observation to the dictionary
        obs[self._indices[0]] = np.array(observation[-1], dtype="float32")

        # Gets camera frames
        self.frame = get_frames(client = self._client, camera_params = self.camera_params,
                                frame_h = self.frame_h, frame_w = self.frame_w, frame = self.frame, )

        # Stores the first frame (external camera)
        obs[self._indices[-1]] = self.frame[0].astype(np.int16)

        return obs


    # Step function
    def step(self, action):
        '''
        Step function. 
            - Applies the action and advances the simulation
            - Computes the rewards
            - Checks for terminal states
            - Gets new observations

        Returns:
            - Observations (dictionary)
            - Rewards (int / float)
            - Terminated and trucanted (bool)
            - Info (dict)
        '''

        # Computes the action
        self._ur5.apply_action_c(action)
        
        # Advances the simulation
        p.stepSimulation()
        
        # --- Debugger ---
        cv.imshow("Station", self.frame[0][0])
        cv.waitKey(1)

        # Computes the rewards after applying the action
        reward = self.compute_reward()

        # Gets the terminal state
        terminated, truncated = self.get_terminal()

        # Get the new state after the action
        obs = self.get_observation()

        # Extra information 
        info = get_info()

        # observations --> obs          --> sensors values
        # reward --> reward             --> task well done
        # terminated --> terminated     --> terminal state, task complete
        # truncated --> truncated       --> time limit reached or observation out of bounds
        # 'info' --> info               --> extra useful information
        return obs, reward, terminated, truncated, info


    # Reset function
    def reset(self, seed=None, options={}):
        '''
        Resets the entire simulation and re - samples positions:
            - Re - samples camera positionining
        '''

        # Reset simulation and gravity establishment
        p.resetSimulation(self._client)
        p.setGravity(0, 0, -10, self._client)


        # --- Camera ---
        # Adds the camera with noise in the positioning
        self.camera_params = set_cam(client=self._client, fov=self.fov, aspect=self.aspect, 
                                     near_val=self.near_plane, far_val=self.far_plane, 
                                     cameras_coord = self.cameras_coord, std = self.std_cam)
                
        
        # --- Create Entities ---
        
        # Random object orientation
        rand_orientation = p.getQuaternionFromEuler(np.random.uniform([-3.1415,-3.1415,-3.1415], [3.1415, 3.1415, 3.1415]), physicsClientId=self._client)

        # Creates a object, a table and the robot
        self._object = Object(self._client, object=0, position=self.obj_pos, orientation=rand_orientation)
        self._ur5 = UR5(self._client)
        self._table = Table(self._client)  

        # Define collisions to keep track of
        self.collisions_to_check = [[self._ur5.id, self._table.id],
                                    [self._object.id, (self._ur5.id, "robotiq_finger_1_link_3")], 
                                    [self._object.id, (self._ur5.id, "robotiq_finger_2_link_3")], 
                                    [self._object.id, (self._ur5.id, "robotiq_finger_middle_link_3")]]      

        # --- Simulation advanced ---
        # Advances the simulation to robot's initial state
        for __ in range(40):
            p.stepSimulation(self._client)
    

        # --- Reseting params 
        # Resets internal values
        __, self._dist_obj_wrist = approx_reward(client = self._client, object = self._object,  # Object - Wrist distance
                                                 dist_obj_wrist = self._dist_obj_wrist, 
                                                 robot_id = self._ur5.id)
        self._t_act = time.time()       # Timer
        __ = self.seed(seed = seed)     # Seed

        # Gets initial state and information
        obs = self.get_observation()
        info = get_info()

        return obs, info


    # Render function
    def render(self, trans=False):
        cv.imshow("Station", self.frame[0][0])
        cv.waitKey(1)


    # Close function: shutdowns the simulation
    def close(self):
        p.disconnect(self._client) 


    # Set seed
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
