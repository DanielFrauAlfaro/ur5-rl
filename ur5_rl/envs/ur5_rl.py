#!/usr/bin/env python3

import gymnasium as gym
import pybullet as p
import pyb_utils
import numpy as np
from ur5_rl.resources.ur5 import UR5e as UR5
from ur5_rl.resources.plane import Plane
from ur5_rl.resources.object import Object
from ur5_rl.resources.table import Table
import matplotlib.pyplot as plt
from math import pi
import math
import time
from cv2 import aruco
import cv2 as cv
import random
from scipy .spatial.transform import Rotation


# Gym environment
class UR5Env(gym.Env):
    # Environment Metadata
    metadata = {'render_modes': ['DIRECT', 'GUI'],
                'render_fps': 60}  
  
    def __init__(self, render_mode="DIRECT"):    

        # Limit values
        self._q_limits = [[-1.5, -3.1415, -3.1415, -3.1415, -3.1415, -6.2831], [1.5, 0.0, 0.0, 3.1415, 3.1415, 6.2831]]
        self._qd_limits = [np.ones(6) * -5000, np.ones(6) * 5000]
        self._qdd_limits = [np.ones(6) * -5000, np.ones(6) * 5000]

        self._ee_limits = [[-1, -1, -1, -pi, -pi, -pi], [1,1,1,pi,pi,pi]]

        self._R_limits =  [np.ones(9) * -10, np.ones(9) * 10]
        self._t_limits = [np.ones(3) * -10, np.ones(3) * 10] 

        # Frame height and width
        self.frame_h = 400
        self.frame_w = 400

        self._limits = [self._q_limits,  
                       self._qd_limits,  
                       self._qdd_limits, 
                       self._ee_limits]

        self.max_action = 0.1
        self._action_limits = [-np.ones(6)*self.max_action, np.ones(6)*self.max_action]

        '''
        Multi-Discrete space in action space:
            - Robot joints and gripper: 6 robot joints

            - 0: do not move
            - 1: increase position of i-joint z axis
            - 2: decrease position of i-joint z axis
        ''' 
        self.action_space = gym.spaces.box.Box(low=self._action_limits[0],
                               high= self._action_limits[1], dtype=np.float32)


        # Dictionary indices
        self._indices = ["q_position", "R_t"]
        self._Rt_indices = ["R", "t"]
        '''
        Dictionary of spaces in observation space:
            - Joint Positions: 6 robot joint
            - R and t matrix between cameras
        '''
        self.observation_space = gym.spaces.Dict({
            self._indices[0]: gym.spaces.box.Box(low=np.float32(np.array(self._q_limits[0])), 
                               high= np.float32(np.array(self._q_limits[1])), dtype=np.float32),

            self._indices[1]: gym.spaces.Dict({
                self._Rt_indices[0]: gym.spaces.box.Box(low=np.float32(np.array(self._R_limits[0])), 
                               high= np.float32(np.array(self._R_limits[1]))),

                self._Rt_indices[1]: gym.spaces.box.Box(low=np.float32(np.array(self._t_limits[0])), 
                               high= np.float32(np.array(self._t_limits[1])))
            })
        })

        # Checks if the selected render mode is within the possibles
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Start seed
        self.np_random, __ = gym.utils.seeding.np_random()

        # Client and UR5 id in Pybullet simulation
        self._client = 0
        if render_mode == "DIRECT":
            self._client = p.connect(p.DIRECT)
        elif render_mode == "GUI":
            self._client = p.connect(p.GUI)
        
        # UR5 object
        self._ur5 = None
        
        # Terminated / Truncated flag
        terminated = False
        truncated = False
        
        # Time limit of the episode (in seconds)
        self._t_limit = 10
        self._t_act = time.time()

        # Object coordinates
        self.obj_pos = [0.2, 0.55, 0.9]

        # Image to be rendered
        self.frame = [np.ones((self.frame_w, self.frame_h)), np.ones((self.frame_w, self.frame_h))]

        # Aruco detectors
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250) # DICT_6X6_250
        parameters =  cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(dictionary, parameters)

        # Camera Parameters
        self.fov = 10
        self.near_plane = 0.02
        self.far_plane = 5.0


        # --- Cameras ---
        # Camera positions variables: [[Position], [Orientation]]

        # Coordinates of the cameras
        self.cameras_coord = [[[0.05, 0.95, 1.05], [0.6, 0.0, -pi/2]],                         # External Camera 1
                         [[-0.021634534277420597, 0.45595843471926517, 1.179405616204087], [3.1339317605594474 + pi/2, -0.02402511411086113, 1.5830796026753562]]]       # Robot camera

        self.std_cam = 0.01
        self.set_cam(cameras_coord = self.cameras_coord, std = self.std_cam)

        # Distance between object an wrist
        self._dist_obj_wrist = math.inf

        # Reward mask
        self.mask = np.array([-100, 2, 2, 2])

        # Vector of warnings
        self.w = []
        
    
    # Adds noise to an array
    def add_noise(self, array, std = 0.5):
        noise = np.random.normal(loc=0, scale=std, size=np.array(array).shape)
        
        return noise

    # Sets the camera with the class parameters and the desiresd coordiantes
    def set_cam(self, cameras_coord, std = 0):
        self.camera_params = []
        self.markers = []

        cameras_coord[0][0] += self.add_noise(cameras_coord[0][0], std = std)
        
        
        # For each camera ...
        for camera in cameras_coord:
            # Obtain rotations
            rot_x = np.array([[1, 0, 0], [0, math.cos(camera[1][0]), -math.sin(camera[1][0])], [0, math.sin(camera[1][0]), math.cos(camera[1][0])]])
            rot_y = np.array([[math.cos(camera[1][1]),0, math.sin(camera[1][1])], [0, 1, 0], [-math.sin(camera[1][1]),0,math.cos(camera[1][1])]])
            rot_z = np.array([[math.cos(camera[1][2]), -math.sin(camera[1][2]), 0], [math.sin(camera[1][2]), math.cos(camera[1][2]), 0], [0, 0, 1]])
            
            rot_mat = np.matmul(np.matmul(rot_x, rot_y), rot_z)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))

            # Computes the view matrix
            view_matrix = p.computeViewMatrix(cameraEyePosition = camera[0], cameraTargetPosition = camera[0] + camera_vec, cameraUpVector = up_vec, physicsClientId = self._client)
            
            # Computes projection matrix
            proj_matrix = p.computeProjectionMatrixFOV(fov = 80, aspect = 1, nearVal = 0.01, farVal = 100, physicsClientId = self._client)
            
            # Convert the tuple to a NumPy array and reshape
            proj_matrix_3x3 = np.array(proj_matrix)
            proj_matrix_3x3 = proj_matrix_3x3.reshape(4, 4)[:-1, :-1]

            # Set a the [3][3] value of the matrix to 1 (is at)
            proj_matrix_3x3[-1][-1] = 1


            # Saves parameters
            self.camera_params.append([view_matrix, proj_matrix, proj_matrix_3x3])
            
            self.markers.append([])

    # Getter for the object position
    def get_object_pos(self, object):
        # ------ Object Position ------
        # Get the position and orientation of the object
        pos, orn = p.getBasePositionAndOrientation(object.id)
        # Convert quaternion to rotation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape((3, 3))

        # Get the directions of the axes in the object's local coordinate system
        y_axis_local = [0,0,1]
        z_axis_local = rotation_matrix[:, 2]
        x_axis_local = np.cross(z_axis_local, y_axis_local)

        y_aux = y_axis_local
        y_axis_local = z_axis_local
        z_axis_local = y_aux
        
        return np.array(pos), np.array(p.getEulerFromQuaternion(orn))

    # Getter for the wrist position
    def get_wrist_pos(self):
        # -------- Wrist Position -------- --> Name:  tool0_ee_link Joint Index: 12 Link Index: b'ee_link'
        # Get the position and orientation of the ee_link
        link_state = p.getLinkState(self._ur5.id, 11, computeLinkVelocity=1, computeForwardKinematics=1)
        pos, orn = link_state[0], link_state[1]

        rotation_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape((3, 3))

        x_axis_local = rotation_matrix[:, 0]
        y_axis_local = rotation_matrix[:, 1]
        z_axis_local = rotation_matrix[:, 2]

        # Euler angles in radians (replace with your actual values)
        roll, pitch, yaw = np.radians(0), np.radians(0), np.radians(45)

        # Create a rotation matrix from Euler angles
        rotation_matrix = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

        # Rotate the vector using the rotation matrix
        x_axis_local = np.dot(rotation_matrix, x_axis_local)
        y_axis_local = np.dot(rotation_matrix, y_axis_local)
        x_axis_local *= -1

        return np.array(pos), np.array(p.getEulerFromQuaternion(orn))

    # Computes the reward according the approximation to the object
    def approx_reward(self, object):
        obj_pos, obj_or = self.get_object_pos(object=object)
        wrist_pos, wrist_or = self.get_wrist_pos()

        distance = np.linalg.norm(wrist_pos - obj_pos)
        reward = 1 if distance < self._dist_obj_wrist else -1

        self._dist_obj_wrist = distance

        return reward
    
    # Check the collision between TWO objects IDs
    def check_collision(self, objects):
        col_detector = pyb_utils.CollisionDetector(self._client, [(objects[0], objects[1])])

        return col_detector.in_collision(margin = 0.0)
        
    # Computes the reward associated with collision reward
    def collision_reward(self, mask = np.array([0,0])):

        checkers = np.array([int(self.check_collision(objects = objects)) for objects in self.collisions_to_check])

        return np.sum(checkers * mask)

    # Computes the whole reward
    def compute_reward(self):
        r = 0

        # Object Approximation
        r += self.approx_reward(object = self._object)

        # Collisions
        r += self.collision_reward(mask = self.mask)
            
        return r


    # Out of bounds check
    def out_of_bounds(self):
        for warning in self.w:
            if "method is not within the observation space" in str(warning.message):
                return True
        
        return False

    # Computes if the environmen has reach a terminal state
    def get_terminal(self):
        terminated = False
        truncated = (time.time() - self._t_act) > self._t_limit \
                    or self.out_of_bounds() \
                    or self.check_collision(objects = [self._table.id, self._ur5.id])

        return terminated, truncated

    # Step function
    def step(self, action):
        # Computes the action
        self._ur5.apply_action(self._ur5.q + action)
        
        # Advances the simulation
        p.stepSimulation()

        # Computes the rewards after applying the action
        reward = self.compute_reward()

        # Gets the terminal state
        terminated, truncated = self.get_terminal()

        # Get the new state after the action
        obs = self.get_observation()

        # Extra information (images)
        info = self.get_info()

        # observations --> obs --> sensors values
        # reward --> reward --> task well done
        # terminated --> terminated --> terminal state, task complete
        # truncated --> truncated --> time limit reached or observation out of bounds
        # 'info' --> info --> extra useful information
        return obs, reward, terminated, truncated, info

    # Reset function
    def reset(self, seed=None, options={}):
        # Reset simulation and gravity establishment
        p.resetSimulation(self._client)
        p.setGravity(0, 0, -20, self._client)

        # Adds the camera with noise in the positioning
        self.set_cam(cameras_coord=self.cameras_coord, std = self.std_cam)

        # self.obj_pos = np.random.normal(self.obj_pos, [0.01, 0.01, 0.01])
        rand_orientation = p.getQuaternionFromEuler(np.random.uniform([-3.1415,-3.1415,-3.1415], [3.1415, 3.1415, 3.1415]), physicsClientId=self._client)

        # Creates a plane and the robot
        self._object = Object(self._client, object=0, position=self.obj_pos, orientation=rand_orientation)
        self._ur5 = UR5(self._client)
        self._table = Table(self._client)  

        # Define collisions
        self.collisions_to_check = [[self._ur5.id, self._table.id],
                                    [self._object.id, (self._ur5.id, "robotiq_finger_1_link_3")], 
                                    [self._object.id, (self._ur5.id, "robotiq_finger_2_link_3")], 
                                    [self._object.id, (self._ur5.id, "robotiq_finger_middle_link_3")]]      

        # Advances the simulation to initial state
        for i in range(40):
            p.stepSimulation(self._client)

        # Resets internal values
        self._dist_obj_wrist = math.inf
        self._t_act = time.time()
        __ = self.seed(seed=seed)

        # Gets initial state and information
        obs = self.get_observation()
        info = self.get_info()

        return obs, info

    # Render function
    def render(self, trans=False):

        cv.imshow("Station", self.frame[0])
        cv.waitKey(1)

    # Close function: shutdowns the simulation
    def close(self):
        p.disconnect(self._client) 

    # Set seed
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

   # Computes the Rotation matrix (R) and Translation (t)
    #   between the two cameras
    def retrieve_R_t(self):
        # For each camera ...
        for idx, camera in enumerate(self.camera_params):
            # Obtains the view

            self.frame[idx] = p.getCameraImage(width = self.frame_w, 
                                     height = self.frame_h, 
                                     viewMatrix = camera[0], 
                                     projectionMatrix = camera[1], 
                                     physicsClientId = self._client)[2]
            
            
            # Generates the RGB representation
            b, g, r, _ = cv.split(self.frame[idx])
            self.frame[idx] = cv.merge([r, g, b])

            # Gray conversion
            gray = cv.cvtColor(self.frame[idx], cv.COLOR_BGR2GRAY)

            # Detects the corners
            markerCorners, _, _ = self.detector.detectMarkers(gray)

            # Concatenates and saves the arrays . There are two sets of arucos            
            combined_array = np.concatenate((markerCorners[0], markerCorners[1]), axis=1)
            self.markers[idx] = combined_array

            # If it has obtained the second image, breaks the loop
            if self.markers[-1] != []: break
        
        # Intrinic parameters of the camera
        K1 = self.camera_params[0][-1]

        # Points rescalation
        points1 = np.vstack([corner for corner in self.markers[0]])
        points2 = np.vstack([corner for corner in self.markers[1]])

        points1_ = np.hstack((points1, np.ones((points1.shape[0], 1))))
        points2_ = np.hstack((points2, np.ones((points2.shape[0], 1))))

        # Point normalization
        normalized_points1 = points1_ @ np.linalg.inv(K1)
        normalized_points2 = points2_ @ np.linalg.inv(K1)

        normalized_points1 = normalized_points1[:,:-1].reshape(-1, 1, 2)
        normalized_points2 = normalized_points2[:,:-1].reshape(-1, 1, 2)

        # Calculate Essential Matrix: coordinates from K1 to K2
        #  The points are obtained from K1
        E, E_ = cv.findEssentialMat(normalized_points1, normalized_points2, K1, method=cv.RANSAC)

        # Recover pose (rotation and translation)
        _, R, t, _ = cv.recoverPose(E, normalized_points1, normalized_points2, K1)
            
        # print(self.R)
        # print(self.t)
        # print("--\n")

        return R.flatten(), t[:,0]

    # Getter for the observations
    def get_observation(self):
        # Gets starting observation
        observation = self._ur5.get_observation()

        # Arranges observation vectors into a dictionary
        obs = {}
        for i in range(len(self._indices[:1])):
            obs[self._indices[i]] = np.array(observation[i], dtype="float32")


        R, t = self.retrieve_R_t() # --> lo que tarda es la obtención de la imagen desde la simulación
        
        obs[self._indices[1]] = {self._Rt_indices[0]: R.astype(np.float32), self._Rt_indices[1]: t.astype(np.float32)}

        return obs

    # Get information from the environment
    def get_info(self):
        return {"frames_ext": cv.cvtColor(self.frame[0], cv.COLOR_BGR2GRAY)}
                # "frames_rob": cv.cvtColor(self.frame[1], cv.COLOR_BGR2GRAY)} --> de momento voy a pasar solamente el frame externo
    
    # Setter de los warnings
    def set_warning(self, w):
        self.w = w
