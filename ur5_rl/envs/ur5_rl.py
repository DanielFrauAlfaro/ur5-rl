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


# Gym environment
class UR5Env(gym.Env):
    # Environment Metadata
    metadata = {'render_modes': ['DIRECT', 'GUI'],
                'render_fps': 60}  
  
    def __init__(self, render_mode="DIRECT"):    

        # Limit values
        self._q_limits = [[-1.5, -3.1415, -3.1415, -3.1415, 0.0, -6.2831], [1.5, 0.0, 0.0, 0.0, 3.1415, 6.2831]]
        self._qd_limits = [[-5000, -5000, -5000, -5000, -5000, -5000], [5000,5000,5000,5000,5000,5000]]
        self._qdd_limits = [[-5000, -5000, -5000, -5000, -5000, -5000], [5000,5000,5000,5000,5000,5000]]

        self._ee_limits = [[-1, -1, -1, -pi, -pi, -pi], [1,1,1,pi,pi,pi]]


        self._limits = [self._q_limits,  
                       self._qd_limits,  
                       self._qdd_limits, 
                       self._ee_limits]

        '''
        Multi-Discrete space in action space:
            - Robot joints and gripper: 6 robot joints

            - 0: do not move
            - 1: increase position of i-joint z axis
            - 2: decrease position of i-joint z axis
        ''' 
        self.action_space = gym.spaces.MultiDiscrete(nvec=[3,3,3,3,3,3])


        # Dictionary indices
        self._indices = ["q_position", "q_velocity", "q_torque", "ee"]

        '''
        Dictionary of spaces in observation space:
            - Joint Positions: 6 robot join
            - Joint velocities: 6 robot joint
            - Joint torque: 6 robot joint
            - End - effector poosition and orientation: XYZ RPY
        '''
        self.observation_space = gym.spaces.Dict({
            self._indices[0]: gym.spaces.box.Box(low=np.float32(np.array(self._q_limits[0])), 
                               high= np.float32(np.array(self._q_limits[1])), dtype=np.float32),

            self._indices[1]: gym.spaces.box.Box(low=np.float32(np.array(self._qd_limits[0])), 
                               high= np.float32(np.array(self._qd_limits[1]))),

            self._indices[2]: gym.spaces.box.Box(low=np.float32(np.array(self._qdd_limits[0])), 
                               high= np.float32(np.array(self._qdd_limits[1]))),

            self._indices[3]: gym.spaces.box.Box(low=np.float32(np.array(self._ee_limits[0])), 
                               high= np.float32(np.array(self._ee_limits[1])))
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

        self._ur5 = None
        
        # Terminated / Truncated flag
        self._terminated = False
        self._truncated = False
        
        # Time limit of the episode (in seconds)
        self._t_limit = 50
        self._t_act = time.time()

        # Image to be rendered
        self._rendered_img = None

        # Constant increment of joint values
        self._incr = 0.1
        self._q_incr = [0, self._incr, -self._incr]


        # Frame height and width
        self.frame_h = 400
        self.frame_w = 400

        # Parameters
        self.fov = 60
        self.near_plane = 0.02
        self.far_plane = 5.0


        # --- Cameras ---
        # Camera positions variables: [[Position], [Orientation]]

        # Coordinates of the cameras
        cameras_coord = [[[0.2, 1.1, 1.1], [0.5, 0.0, -pi/2]],                         # External Camera 1
                         [[-0.021634534277420597, 0.45595843471926517, 1.179405616204087], [3.1339317605594474 + pi/2, -0.02402511411086113, 1.5830796026753562]]]       # Robot camera


        self.camera_params = []
        self.markers = []
        self.R = []
        self.t = []
        self.frames = []

        # For each camera ...
        for camera in cameras_coord[:]:
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
            
            # Saves parameters
            self.camera_params.append([view_matrix, proj_matrix])
            
            self.markers.append([])
            self.frames.append([])


    # Step function
    def step(self, action):
        # Gets an observation of the robot
        observation = self._ur5.get_observation()
        
        # Gets joint positions of the robot and the gripper
        q_act = observation[0]

        # Applies increments according to action
        for i in range(len(q_act)):
            q_act[i] = q_act[i] + self._q_incr[action[i]]


        # Builds the message and sends it to the robot
        q = q_act

        self._ur5.apply_action(q)

        # Advances the simulation
        p.stepSimulation()

        # Gets an observation
        observation = self._ur5.get_observation()



        #######################################################
        # TODO: establecer función de recompensa ##############
        #######################################################
        '''
            Collision
        '''
        # print(self.col_detector.in_collision(margin=0.0))


        '''
            Camera intrinsic parameters
        '''

        
        
        reward = 0



        # If it has reached time limit, truncates the episode
        if (time.time() - self._t_act) > self._t_limit:
            self._truncated = True
        
        # Else, it checks if the observation values are in bounds
        else:
            # Stop condition: values out of bounds
            for i in range(len(self._limits)):
                for j in range(len(observation[i])):
                    if observation[i][j] < self._limits[i][0][j] or observation[i][j] > self._limits[i][1][j]:
                        self._truncated = True
                        break
                if self._truncated:
                    break


        # Arranges observation vectors into a dictionary
        obs = {}
        for i in range(len(self._indices)):
            obs[self._indices[i]] = np.array(observation[i], dtype="float32")

        # Extra information
        info = {"frames_ext": self.frames[0], 
                "frames_rob": self.frames[1],
                "R": self.R,
                "t": self.t}


        # observations --> obs --> sensors values
        # reward --> reward --> task well done
        # self._terminated --> terminated --> terminal state, task complete
        # self._truncated --> truncated --> time limit reached or observation out of bounds
        # 'info' --> info --> extra useful information

        return obs, reward, self._terminated, self._truncated, info


    # Reset function
    def reset(self, seed=None, options={}):
        # Reset simulation and gravity establishment
        p.resetSimulation(self._client)
        p.setGravity(0, 0, -20, self._client)


        # Creates a plane and the robot
        self._object = Object(self._client, object=0, position=[0.2, 0.55, 1.15], orientation=[0, 0, 0, 1])
        self._ur5 = UR5(self._client)
        self._table = Table(self._client)

        # From each body, obtains every joint and stores them in a dictionary
        bodies = [self._ur5, self._object, self._table]

        self.dict_ids ={p.getJointInfo(body.id, joint)[1]: p.getJointInfo(body.id, joint)[0] for body in bodies for joint in range(p.getNumJoints(body.id))}
        self.dict_ids_rev = {value: key for key, value in self.dict_ids.items()}

        '''
            Colisiones relevantes:
                - Colision Robot (cuerpo) - Mesa
                - Colision Dedos Pinza - Objeto

            Se podria usar:
                - contacts = p.getContactPoints(bodyA=body1, bodyB=body2)
            Para obtener la posicion del punto de contacto y lograr que sea por encima del objeto (mas adelante)
        '''
        robot = pyb_utils.Robot(self._ur5.id, client_id=self._client)
        
        self.col_detector = pyb_utils.CollisionDetector(self._client,
                                                    [((self._ur5.id, "wrist_1_link"), (self._ur5.id, "base_link"))])

        # Reset the 'terminated' flag
        self._terminated = False

        # Advances the simulation to initial state
        for i in range(25):
            p.stepSimulation(self._client)

            self.render(trans=True)

        # Gets starting observation
        observation = self._ur5.get_observation()

        # Arranges observation vectors into a dictionary
        obs = {}
        for i in range(len(self._indices)):
            obs[self._indices[i]] = np.array(observation[i], dtype="float32")


        # Resets internal values
        self._truncated = False
        self._terminated = False
        self._t_act = time.time()
        __ = self.seed(seed=seed)

        return obs, {}


    # Render function
    def render(self, trans=False):
        # if it is the first time that 'render' is called ...
        if self._rendered_img is None:
            # ... initializes the image variables
            self.figure, self._rendered_img = plt.subplots(1,len(self.camera_params))
            self._rendered_img = [ax.imshow(np.zeros((self.frame_w, self.frame_h, 3))) for ax in self._rendered_img]

            plt.pause(0.01)


        # Updates the viewed frame for each image
        for idx, camera in enumerate(self.camera_params):
            frame = p.getCameraImage(width = self.frame_w, 
                                     height = self.frame_h, 
                                     viewMatrix = camera[0], 
                                     projectionMatrix = camera[1], 
                                     physicsClientId = self._client)[2]

            self.frames[idx] = frame
            self._rendered_img[idx].set_data(frame)

            b, g, r, a = cv.split(frame)
            frame = cv.merge([b, g, r])
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250) # DICT_6X6_250
            parameters =  cv.aruco.DetectorParameters()
            detector = cv.aruco.ArucoDetector(dictionary, parameters)

            markerCorners, _, _ = detector.detectMarkers(frame)

            
            combined_array = np.concatenate((markerCorners[0], markerCorners[1]), axis=1)

            self.markers[idx] = combined_array

            if not trans:
                break

            # plt.figure()
            # plt.imshow(frame)
            # for i in range(len(markerIds)):
            #     c = markerCorners[i][0]
            #     plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(markerIds[i]))
            #     print(c)
            #     for j in range(4):
            #         plt.plot([c[j, 0]], [c[j, 1]], "o", label = "punto_"+str(j))

            # plt.legend()
            # plt.show()

        plt.draw()
        plt.pause(1/self.metadata['render_fps'])

        K1 = self.camera_params[0][1]
        K2 = self.camera_params[1][1]

        points1 = np.vstack([corner for corner in self.markers[0]])
        points2 = np.vstack([corner for corner in self.markers[1]])
        

        # Convert the tuple to a NumPy array
        K1_array = np.array(K1)
        K1 = K1_array.reshape(4, 4)[:-1, :-1]

        K2_array = np.array(K2)
        K2 = K2_array.reshape(4, 4)[:-1, :-1]

        K1[-1][-1] = 1
        K2[-1][-1] = 1

        points1_ = np.hstack((points1, np.ones((points1.shape[0], 1))))
        points2_ = np.hstack((points2, np.ones((points2.shape[0], 1))))


        normalized_points1 = points1_ @ np.linalg.inv(K1)
        normalized_points2 = points2_ @ np.linalg.inv(K1)

        
        normalized_points1 = normalized_points1[:,:-1].reshape(-1, 1, 2)
        normalized_points2 = normalized_points2[:,:-1].reshape(-1, 1, 2)

        # Calculate Essential Matrix
        E, E_ = cv.findEssentialMat(normalized_points1, normalized_points2, K1, method=cv.RANSAC)


        # # Recover pose (rotation and translation)
        _, self.R, self.t, _ = cv.recoverPose(E, normalized_points1, normalized_points2, K1)

        # print(R)
        # print(t)
        # print("--\n")
     

    # Close function: shutdowns the simulation
    def close(self):
        p.disconnect(self._client) 


    # Setter for the camera position
    def set_cam(self, pos, rpy):
        self._cam_pos = pos

        self._cam_roll = rpy[0]
        self._cam_pitch = rpy[1]
        self._cam_yaw = rpy[2]

    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
