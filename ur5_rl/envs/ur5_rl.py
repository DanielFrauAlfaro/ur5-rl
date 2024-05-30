import gymnasium as gym
import pybullet as p
import numpy as np
from ur5_rl.resources.ur5 import UR5e as UR5
from ur5_rl.resources.utils import *
from ur5_rl.resources.object import Object
from ur5_rl.resources.table import Table
from math import pi
import math
import time
import cv2 as cv
import random

 
# Gym environment
class UR5Env(gym.Env):
    # Environment Metadata
    metadata = {'render_modes': ['DIRECT', 'GUI'],
                'render_fps': 60,
                'show': False}  
  
    def __init__(self, render_mode="DIRECT", show = False):    

        # --- Observation limit values ---
        self._q_limits = [np.array([-1.5, -3.1415, -3.1415, -3.1415, -3.1415, -6.2831]), np.array([1.5, 0.0, 0.0, 3.1415, 3.1415, 6.2831])]
        self._qd_limits = [np.ones(6) * -20, np.ones(6) * 20]
        self._qdd_limits = [np.ones(6) * -5000, np.ones(6) * 5000]
        self._ee_limits = [[-1, -1, -1, -pi, -pi, -pi], [1, 1, 1, pi, pi, pi]]

        self._limits = [self._q_limits,  
                       self._qd_limits,  
                       self._qdd_limits, 
                       self._ee_limits]

        # --- Action limits ---
        # Joint actions scalers
        self.max_action_original = 0.0666
        self.max_action_or_original = 0.12

        self.max_action = self.max_action_original
        self.max_action_or = self.max_action_or_original

        # Yaw scaler
        self.max_action_yaw = 2.5
        self._action_limits = [-np.ones(6), np.ones(6)]

        # Frame height and width
        self.frame_h = 160
        self.frame_w = 160

        '''
        Box space in action space:
            - Robot joints and gripper: 6 continous robot joints

        ''' 
        self.action_space = gym.spaces.box.Box(low=self._action_limits[0],
                               high= self._action_limits[1], dtype=np.float32)
        
        
        '''
        Dictionary of spaces in observation space:
            - Joint Positions: 6 robot end effector position
            - Camera image: grayscale and depth
        '''
        # Dictionary indices
        self._indices = ["ee_position", "image"]

        self.observation_space = gym.spaces.Dict({
            self._indices[0]: gym.spaces.box.Box(low=np.float32(np.array(self._ee_limits[0])), 
                               high= np.float32(np.array(self._ee_limits[1])), dtype=np.float32),


            self._indices[1]: gym.spaces.box.Box(low=0, high=255, shape=(6, self.frame_w, self.frame_h), dtype=np.float16)
        })

        # Time limit of the episode (in steps)
        self._step_limit = 40
        self.global_steps = 0
        self.steps = 0


        # Checks if the selected render mode is within the possibilities
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        # Show variable for render during training
        self.show = show

        # Start seed
        self.np_random, __ = gym.utils.seeding.np_random()
        np.random.seed(0)

        # Client in Pybullet simulation
        self._client = 0
        if render_mode == "DIRECT":
            self._client = p.connect(p.DIRECT)
        elif render_mode == "GUI":
            self._client = p.connect(p.GUI)
        
        # Object initial coordinates for spawning
        self.obj_pos = [0.2, 0.55, 0.9]

        # Images to be rendered
        self.frame = [np.ones((self.frame_w, self.frame_h), dtype=np.int16), 
                      np.ones((self.frame_w, self.frame_h), dtype=np.int16),
                      np.ones((self.frame_w, self.frame_h), dtype=np.int16)]

        # Camera Parameters
        self.fov = 10
        self.near_plane = 0.02
        self.far_plane = 5.0
        self.aspect = 1


        # --- Cameras ---
        # Camera positions variables: [[Position], [Orientation]]
        # Coordinates of the cameras
        self.cameras_coord = [[[0.05, 0.95, 1.05], [0.6, 0.0, -pi/2]],                                  # Frontal
                              [[0.7, 0.55, 1.05], [0.0, 0.0, -pi]],                                     # Side
                              [[0.048, 0.353, 1.1712], [-1.62160814, -0.78472898,  1.57433526]]]        # Wrist


        # Sample STD for camera
        self.std_cam = 0.05

        # Link of the camera in the URDF
        self.camera_link = 22

        # Obtain camera parameters
        self.camera_params = set_cam(client=self._client, fov=self.fov, aspect=self.aspect, 
                                     near_val=self.near_plane, far_val=self.far_plane, 
                                     cameras_coord = self.cameras_coord, std = self.std_cam)

        # Initial distance between object an wrist
        self._dist_obj_wrist = [math.inf, math.inf, math.inf]

        # Reward collision mask
        self.mask = np.array([-0.5, 
                              0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0,
                              0, 0,
                              4, 3, 3])
        
        # Initial gripper position
        self.g = 0

        # Threshold for which there is a collision between two elements
        self.col_thres = 6

    # Resample camera parameters
    def resample_cameras(self):
        '''
        Resample the intrinsic parameters of the cameras
        Input:
            - //

        Output:
            - // 
        '''
        
        # Obtain the position of the camera link
        link_state = p.getLinkState(self._ur5.id, self.camera_link, computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId = self._client)
        pos, orn = link_state[0], link_state[1]

        # Get the rotation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orn, physicsClientId = self._client)).reshape((3, 3))

        # Get vectors
        x_axis_local = rotation_matrix[:,0] / np.linalg.norm(rotation_matrix[:,0])
        y_axis_local = rotation_matrix[:,1] / np.linalg.norm(rotation_matrix[:,1])
        z_axis_local = rotation_matrix[:,2] / np.linalg.norm(rotation_matrix[:,2])

        # Transformation: Y -pi/2 rotation
        x_axis_local, z_axis_local = z_axis_local, x_axis_local     
        x_axis_local, z_axis_local = -x_axis_local, -z_axis_local    

        # New transformation matrix
        rotation_matrix = np.vstack((x_axis_local, y_axis_local, z_axis_local)).T

        # New camera coordinates
        pos = list(pos)
        pos[-1] -= 0.01

        # Sample new camera parameters
        self.cameras_coord[-1][0] = pos
        self.cameras_coord[-1][-1] = rotation_matrix_to_euler_xyz(rotation_matrix)
        self.camera_params = set_cam(client=self._client, fov=self.fov, aspect=self.aspect, 
                                     near_val=self.near_plane, far_val=self.far_plane, 
                                     cameras_coord = self.cameras_coord, std = self.std_cam)


    # Method for grasping
    def grasping(self, g = 5):
        '''
        Takes the gripper to a desired pose from the environment:
        Input:
            - g: position in the range [0,255]

        Output:
            - col (bool): flag detecting collision between the gripper and the object 
                according to the "self.col_thres" and the collision mask
            - self._ur5.g (int): actual gripper position between [0,255]
        '''

        # Applies desired action to the gripper
        self._ur5.apply_action_g(self.g)
        self.g += g

        # Checks for collisions
        col = collision_reward(client = self._client, collisions_to_check = self.collisions_to_check, mask = self.mask) >= self.col_thres

        return col, self._ur5.g


    # Method for getting the error between the object and the robot's end effector
    def get_error(self):
        '''
        Gets the error between the robot's end effector in dual quaternion distance.
        Input:
            - //

        Output:
            - 1/r.item() (float): TODO
            - np.linalg.norm(np.array(obj_pos) - np.array(wrist_pos)) (float): linear distance
            - d_or (float): distance in roation in Euler angles
        '''
        
        # Obtain position and orientation in:
        #    - cartesian position, Euler angles, dual quaternion
        # For the object, both reference systems are returned
        obj_pos, euler, euler_, DQ_obj, DQ_obj_ = get_object_pos(object=self._object, client = self._client)
        wrist_pos, euler_w, DQ_w = get_wrist_pos(client = self._client, robot_id=self._ur5.id)

        # Angular distance using quaternion
        d_p = math.acos(2*np.dot(DQ_w[0:4], DQ_obj[0:4]) ** 2 - 1)
        d_p_ = math.acos(2*np.dot(DQ_w[0:4], DQ_obj_[0:4]) ** 2 - 1)

        # Checks which reference system is the nearest and stores ...
        if d_p < d_p_:
            # Reward
            dist, __, __ = dq_distance(torch.tensor(np.array([DQ_obj])), torch.tensor(np.array([DQ_w])))
            
            # Euler rotation distance
            d_or = np.linalg.norm(np.array(euler_w) - np.array(euler))
        else:
            # Reward
            dist, __, __ = dq_distance(torch.tensor(np.array([DQ_obj_])), torch.tensor(np.array([DQ_w])))
            
            # Euler rotation distance
            d_or = np.linalg.norm(np.array(euler_w) - np.array(euler_))

        return dist.item(), np.linalg.norm(np.array(obj_pos) - np.array(wrist_pos)), d_or


    # Computes the combined reward
    def compute_reward(self):
        '''
        Computes the environment reward according to the approximation reward
            and the collision reward
        Input:
            - //

        Returns:
            - r (float): final reward
        '''

        r = 0

        # Object approximation reward
        r, self._dist_obj_wrist = approx_reward(client = self._client, object = self._object, 
                                                dist_obj_wrist = self._dist_obj_wrist, robot_id = self._ur5.id)

        # Collision reward
        # r += collision_reward(client = self._client, collisions_to_check = self.collisions_to_check, mask = self.mask)
            
        return r


    # Computes if the environmen has reach a terminal state
    def get_terminal(self):
        '''
        Checks if the environment has reached a terminal state
        Input:
            - //
        
        Output:
            - terminated (bool): falg that indicates that the environment ends normally
            - truncated (bool): falg that indicates that the environment ends with problems
                - End time
                - Out of bounds joints
                - Collision with the table
        '''
        
        # Collision reward
        col_r = collision_reward(client = self._client, collisions_to_check = self.collisions_to_check, mask = self.mask)

        # Object and wirst position
        obj_pos, __, __, __, __ = get_object_pos(object=self._object, client = self._client)
        wrist_pos, __, __ = get_wrist_pos(client = self._client, robot_id=self._ur5.id)

        # Flag terminated
        terminated = wrist_pos[-2] <= obj_pos[-2] + 0.0 # \
                    # or col_r > 0.0
                                                                           
        # Flag truncated
        truncated = out_of_bounds(self._limits, self._ur5) \
                    or col_r < 0.0 \
                    or self.steps >= self._step_limit             
                    # or check_collision(client = self._client, objects = [self._table.id, self._ur5.id])

        return terminated, truncated


    # Getter for the observations
    def get_observation(self):
        '''
        Obtains the observation in the environment in a dictionary
        Input:
            - //

        Output:
            - obs (dictionary): contains the observations ("ee_position", "image")
        '''

        # Gets starting observation of the robot
        observation = self._ur5.get_observation()       # [q, qd, gripper, ee]
        
        # Stores joint velocities
        self.qd = observation[1]

        # Arranges observation vectors into a dictionary
        obs = {}

        # Assigns robot observation to the dictionary
        obs[self._indices[0]] = np.array(observation[-1], dtype="float32")

        # Obtains camera frames
        self.frame = get_frames(client = self._client, camera_params = self.camera_params,
                                frame_h = self.frame_h, frame_w = self.frame_w, frame = self.frame, )

        # Stores the normalized camera frames
        normalized_image_0_0 = (self.frame[0][0] - np.min(self.frame[0][0])) / (np.max(self.frame[0][0]) - np.min(self.frame[0][0]))
        normalized_image_1_0 = (self.frame[0][1] - np.min(self.frame[0][1])) / (np.max(self.frame[0][1]) - np.min(self.frame[0][1]))

        normalized_image_0_1 = (self.frame[1][0] - np.min(self.frame[1][0])) / (np.max(self.frame[1][0]) - np.min(self.frame[1][0]))
        normalized_image_1_1 = (self.frame[1][1] - np.min(self.frame[1][1])) / (np.max(self.frame[1][1]) - np.min(self.frame[1][1]))

        normalized_image_0_2 = (self.frame[2][0] - np.min(self.frame[2][0])) / (np.max(self.frame[2][0]) - np.min(self.frame[2][0]))
        normalized_image_1_2 = (self.frame[2][1] - np.min(self.frame[2][1])) / (np.max(self.frame[2][1]) - np.min(self.frame[2][1]))

        # Merges channels in a single image
        merged = cv.merge([normalized_image_0_0, normalized_image_1_0, normalized_image_0_1, normalized_image_1_1, normalized_image_0_2, normalized_image_1_2])
        
        # Transpose
        merged = np.transpose(merged, (2,0,1))
        
        # Store observation to match original typing
        obs[self._indices[-1]] = merged.astype(np.float16)

        return obs


    # Step function
    def step(self, action):
        '''
        Step function. 
            - Applies the action and advances the simulation
            - Computes the rewards
            - Checks for terminal states
            - Gets new observations

        Input:
            - action (np.array): 6D array with the action increments in each dimension
        
        Output:
            - obs (dict): dictionary of observations ("ee_position", "image")
            - reward (float): associated reward respect to the action taken
            - terminated and truncated (bool): end episode flags
            - info (dict): auxiliar information dictionary
        '''

        # Increase number steps
        self.steps += 1
        self.global_steps += 1
        
        # Rescale action
        action[0:3] *= self.max_action
        action[3:]  *= self.max_action_or
        action[-1]  *= self.max_action_yaw        


        # Computes the action
        self._ur5.apply_action_c(action)
        
        # Advances the simulation
        p.stepSimulation(self._client)

        # Resample camera parameters
        self.resample_cameras()
        

        # --- Debugger ---
        if self.show:
            cv.imshow("Station " + str(self._client), self.frame[0][0])
            cv.waitKey(1)


        # Computes the rewards after applying the action
        reward = self.compute_reward()

        # Gets the terminal state
        terminated, truncated = self.get_terminal()

        # Extra penalisation if truncated
        if truncated:
            reward -= 0
            if out_of_bounds(self._limits, self._ur5):
                reward -= 10
        
        # Bonus if terminated
        if reward > 0 and terminated:
            reward += reward*0.5

        # Get the new state after the action
        obs = self.get_observation()

        # Wrist and object position for extra-information
        obj_pos, __, __, __, __ = get_object_pos(object=self._object, client = self._client)
        wrist_pos, __, __ = get_wrist_pos(client = self._client, robot_id=self._ur5.id)

        # Extra information 
        info = {"limit": wrist_pos[-2] <= obj_pos[-2] + 0.0}

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

        Input:
            - seed (int): set a seed for the randomness
            - options (dictionary): extra-options if needed (NOT USED)

        Output:
            - obs (dict): dictionary of observations ("ee_position", "image")
            - info (dict): auxiliar information dictionary
        '''

        # Reset steps
        self.steps = 0
        
        # Reset gripper position
        self.g = 0
            
        # Scalers
        self.max_action = self.max_action_original
        self.max_action_or = self.max_action_or_original

        # Reset simulation and gravity establishment
        p.resetSimulation(self._client)
        p.setGravity(0, 0, -10, self._client)

        # --- Camera ---
        # Samples new camera parameters
        self.camera_params = set_cam(client=self._client, fov=self.fov, aspect=self.aspect, 
                                     near_val=self.near_plane, far_val=self.far_plane, 
                                     cameras_coord = self.cameras_coord, std = self.std_cam)
                
        
        # --- Create Entities ---
        # Random object position and orientation
        pos, orn = np.random.uniform([[0.05, 0.5, 0.85], [-pi, -pi/2, -3.1415]], 
                                     [[0.275,  0.62, 0.85], [pi,  -pi/2,  3.1415]])

        rand_orientation = p.getQuaternionFromEuler(orn, physicsClientId=self._client)
        
                
        # Creates a object, a table and the robot        
        object_chosen = random.randint(0,13) # 0,9
        
        self._object = Object(self._client, object=object_chosen, position=pos, orientation=rand_orientation)
        self._ur5 = UR5(self._client)
        self._table = Table(self._client)  

        # Define collisions to keep track of
        self.collisions_to_check = [[self._ur5.id, self._table.id],
                                    [self._object.id, (self._ur5.id, "robotiq_finger_1_link_3")], 
                                    [self._object.id, (self._ur5.id, "robotiq_finger_2_link_3")], 
                                    [self._object.id, (self._ur5.id, "robotiq_finger_middle_link_3")],
                                    
                                    [self._object.id, (self._ur5.id, "robotiq_finger_1_link_2")], 
                                    [self._object.id, (self._ur5.id, "robotiq_finger_2_link_2")], 
                                    [self._object.id, (self._ur5.id, "robotiq_finger_middle_link_2")], 
                                    
                                    [self._object.id, (self._ur5.id, "robotiq_finger_1_link_1")], 
                                    [self._object.id, (self._ur5.id, "robotiq_finger_2_link_1")], 
                                    [self._object.id, (self._ur5.id, "robotiq_finger_middle_link_1")],
                                    
                                    [self._object.id, (self._ur5.id, "robotiq_tool0")],
                                    [self._object.id, (self._ur5.id, "robotiq_palm")],

                                    [self._object.id, (self._ur5.id, "contact1")], 
                                    [self._object.id, (self._ur5.id, "contact2")], 
                                    [self._object.id, (self._ur5.id, "contact3")]]      

        # --- Simulation advanced ---
        # Advances the simulation to robot's initial state
        for __ in range(60):
            p.stepSimulation(self._client)

        # --- Reseting params ---
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
    def render(self):
        return self.frame[0][0]
        cv.imshow("Station", self.frame[1][0])
        cv.waitKey(1)


    # Close function: shutdowns the simulation
    def close(self):
        p.disconnect(self._client) 
        cv.destroyAllWindows()


    # Set seed
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
