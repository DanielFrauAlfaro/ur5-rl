import numpy as np
import math
import pybullet as p
import pyb_utils
import cv2 as cv
import copy
from dq import *
import torch


# Adds noise to an array
def add_noise(array, std = 0.5):
    '''
       Adds normal noise to an array               
        Input;                                    
           - array (np.array): array to add noise to.         
           - std (float): standard deviation to the noise  
                                                   
       Output:                                    
           - Noised array (np.array)               
    '''

    return np.random.normal(loc=0, scale=std, size=np.array(array).shape)


# Sets the camera with the class parameters and the desiresd coordiantes
def set_cam(client, fov, aspect, near_val, far_val, cameras_coord, std = 0):
    '''
        Obtains the intrinsic and extrinsic parameters of cameras       
        Input:                                                  
           - client: Pybullet client (int)                             
           - fov: field of view (int)                                  
           - aspect: aspect of the cameras (int)                       
           - near_val: nearest value for depth camera (int)            
           - far_val: farthest value for depth camera (int)            
           - cameras_coord: list of arrays of positions for            
       the camera (list of list / arrays) (XYZ - RPY)                  
           - std: standard deviation for the noise                     
                                                                       
       Output:                                                        
           - A list of two matrix: intrinsic and extrinsic parameters  
    '''    

    # PArameters for the cameras
    camera_params = []

    # Adds noise to the camera coordinates (just the first one)
    cameras_coord_aux = copy.deepcopy(cameras_coord)

    # Add noise
    cameras_coord_aux[0][0] += add_noise(cameras_coord_aux[0][0], std = std)
    cameras_coord_aux[1][0] += add_noise(cameras_coord_aux[0][0], std = std)
    

    # For each camera ...
    for camera in cameras_coord_aux:

        # Obtain rotation matrices
        rot_x = np.array([[1, 0, 0], [0, math.cos(camera[1][0]), -math.sin(camera[1][0])], [0, math.sin(camera[1][0]), math.cos(camera[1][0])]])
        rot_y = np.array([[math.cos(camera[1][1]),0, math.sin(camera[1][1])], [0, 1, 0], [-math.sin(camera[1][1]),0,math.cos(camera[1][1])]])
        rot_z = np.array([[math.cos(camera[1][2]), -math.sin(camera[1][2]), 0], [math.sin(camera[1][2]), math.cos(camera[1][2]), 0], [0, 0, 1]])
        
        # Global rotation
        rot_mat = np.matmul(np.matmul(rot_x, rot_y), rot_z)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))

        # Computes the view matrix
        view_matrix = p.computeViewMatrix(cameraEyePosition = camera[0], cameraTargetPosition = camera[0] + camera_vec, cameraUpVector = up_vec, physicsClientId = client)
        
        # Computes projection matrix
        proj_matrix = p.computeProjectionMatrixFOV(fov = 80, aspect = 1, nearVal = 0.01, farVal = 1.4, physicsClientId = client)
        
        # Convert the tuple to a NumPy array and reshape
        proj_matrix_3x3 = np.array(proj_matrix)
        proj_matrix_3x3 = proj_matrix_3x3.reshape(4, 4)[:-1, :-1]

        # Set a the [3][3] value of the matrix to 1 (is at)
        proj_matrix_3x3[-1][-1] = 1

        # Save parameters
        camera_params.append([view_matrix, proj_matrix, proj_matrix_3x3])

    return camera_params


# Shows axis on the world
def print_axis(client, pos, rotation_matrix):
    '''
    Prints axis in the Pybullet simulation:
    Input: 
        - client (int): simulation client
        - pos (np.array, 1x3): position of the axis
        - rotation_matrix (np.array, 3x3): direction of the axis. Each column represents an axis

    Output:
        - //
    '''

    # Length
    axis_length = 0.5

    # Extract axis
    x_axis_local = rotation_matrix[0]
    y_axis_local = rotation_matrix[1]
    z_axis_local = rotation_matrix[2]

    # Obtain the end-point
    line_start = pos
    line_end_x = [pos[0] + axis_length * x_axis_local[0], pos[1] + axis_length * x_axis_local[1], pos[2] + axis_length * x_axis_local[2]]
    line_end_y = [pos[0] + axis_length * y_axis_local[0], pos[1] + axis_length * y_axis_local[1], pos[2] + axis_length * y_axis_local[2]]
    line_end_z = [pos[0] + axis_length * z_axis_local[0], pos[1] + axis_length * z_axis_local[1], pos[2] + axis_length * z_axis_local[2]]

    # Draw lines representing the axes of the object
    p.addUserDebugLine(line_start, line_end_x, [1, 0, 0], lifeTime=0.3, physicsClientId = client)  # X-axis (red)
    p.addUserDebugLine(line_start, line_end_y, [0, 1, 0], lifeTime=0.3, physicsClientId = client)  # Y-axis (green)
    p.addUserDebugLine(line_start, line_end_z, [0, 0, 1], lifeTime=0.3, physicsClientId = client)  # Z-axis (blue)


# Get the rotation Euler angles from the rotation matrix
def rotation_matrix_to_euler_xyz(R):
    '''
    Obtains the Euler angles from a 3x3 rotation matrix
    Input:
        - R (np.array, 3x3): 3x3 rotation matrix

    Output:
        - (np.array) Array of Euler angles
    '''

    theta_y = np.arcsin(-R[0, 2])
    theta_x = np.arctan2(R[1, 2], R[2, 2])
    theta_z = np.arctan2(R[0, 1], R[0, 0])

    return np.array([theta_x, theta_y, theta_z])

# Get quaternion from a Pybullet orientation
def get_quaternion(q):
    '''
    Get the quaternion from a Pybullet quaternion as a np.array
    Input:
        - q (tuple/list/np.array): quaternion from Pybullet [x,y,z,w] 

    Output:
        - (np.array): quaternion [w,x,y,z] 
    '''

    return np.array([q[-1], q[0], q[1], q[2]])


# Get dual quaternion
def get_dualQuaternion(q_r, q_t):
    '''
    Get dual quaternion from a rotation quaternion and a position quaternion
    Input:
        - q_r (np.array): rotation unit quaternion [w,x,y,z]
        - q_t (np.array): traslation unit quaternion [w,x,y,z]

    Output:
        - (np.array): dual quaternion representing rotation and traslation [wr,xr,yr,zr,wt,xt,yt,zt]
    '''

    return np.concatenate((q_r, 0.5*q_mul(torch.tensor(q_t).unsqueeze(dim=0), torch.tensor(q_r).unsqueeze(dim=0)).squeeze(dim=0).numpy()))


# Getter for the object position
def get_object_pos(client, object):
    '''
        Obtains the position and orientantion of a given object class   
        Input:                                                               
           - client (int): Pybullet client                          
           - object (class Object): class that represents an object (must have        
        "id" attribute)                                        
                                             
        Output:                                                        
           - np.array(pos) (np.array): position of the object in cartesian 
           - euler_angles (np.array): euler angles of the original reference system
           - euler_angles_ (np.array): euler angles of the auxiliar reference system
           - DQ (np.array): dual quaternion of the original reference system
           - DQ_ (np.array): dual quaternion of the auxiliar reference system
    '''

    # Get the position and orientation of the object
    pos, orn = p.getBasePositionAndOrientation(object.id, physicsClientId=client)
    pos = list(pos)

    # Convert quaternion to rotation matrix
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orn, physicsClientId = client)).reshape((3, 3))
    
    # Normalise rotation axis
    x_axis_local = rotation_matrix[:,0] / np.linalg.norm(rotation_matrix[:,0])
    y_axis_local = rotation_matrix[:,1] / np.linalg.norm(rotation_matrix[:,1])
    z_axis_local = rotation_matrix[:,2] / np.linalg.norm(rotation_matrix[:,2])

    # Correct cracker_box position
    if "cracker" in object.name:
        pos[-1] += 0.26
    else:
        pos[-1] += 0.215

    # Correct axis
    x_axis_local = np.array([0, 0, -1])
    y_axis_local = np.cross(z_axis_local, x_axis_local)
    
    # Correct boxes axis
    if "box" in object.name:
        z_axis_local, y_axis_local = -y_axis_local, z_axis_local

    z_axis_local_ = -1*z_axis_local
    y_axis_local_ = np.cross(z_axis_local_, x_axis_local)


    # Create new rotation matrices and euler angles for both reference systems
    rotation_matrix = np.vstack((x_axis_local, y_axis_local, z_axis_local)).T
    rotation_matrix_ = np.vstack((x_axis_local, y_axis_local_, z_axis_local_)).T
    euler_angles = rotation_matrix_to_euler_xyz(rotation_matrix)
    euler_angles_ = rotation_matrix_to_euler_xyz(rotation_matrix_)
    
    # print_axis(client = client, pos = pos, rotation_matrix = [x_axis_local, y_axis_local, z_axis_local])      # --> blue (z)
    # print_axis(client = client, pos = pos, rotation_matrix = [x_axis_local, y_axis_local_, z_axis_local_])


    # --- DQ ---
    # Position quaternion
    pos.append(0.0)
    pos_Q = get_quaternion(pos)

    # Rotation quaternion in both reference systems
    orn = p.getQuaternionFromEuler(euler_angles)
    orn_Q = get_quaternion(orn)
    DQ = get_dualQuaternion(q_r=orn_Q, q_t=pos_Q)

    orn_ = p.getQuaternionFromEuler(euler_angles_)
    orn_Q_ = get_quaternion(orn_)
    DQ_ = get_dualQuaternion(q_r=orn_Q_, q_t=pos_Q)

    return np.array(pos), euler_angles, euler_angles_, DQ, DQ_


# Getter for the wrist position
def get_wrist_pos(client, robot_id):
    '''
        Obtains the position and orientantion of a given joint in the robot object class   
        Input:                                                  
           - client (int): Pybullet client                             
           - robot_id (int): ID of the robot in the simulation  
                                             
        Output:                                                        
           - np.array(pos) (np.array): cartesian position of the wrist 
           - euler_angles (np.array): euler angles
           - DQ (np.array): dual quaternion representation of the wrist
    '''

    # Get the position and orientation of the ee_link. Name:  tool0_ee_link Joint Index: 12 Link Index: b'ee_link'
    link_state = p.getLinkState(robot_id, 11, computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId = client)
    pos, orn = link_state[0], link_state[1]
    pos = list(pos)

    # Get rotation matrix
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orn, physicsClientId = client)).reshape((3, 3))

    # Normalise axis
    x_axis_local = rotation_matrix[:,0] / np.linalg.norm(rotation_matrix[:,0])
    y_axis_local = rotation_matrix[:,1] / np.linalg.norm(rotation_matrix[:,1])
    z_axis_local = rotation_matrix[:,2] / np.linalg.norm(rotation_matrix[:,2])

    # Transform axis
    x_axis_local = x_axis_local + y_axis_local
    x_axis_local /= np.linalg.norm(x_axis_local)
    y_axis_local = np.cross(z_axis_local, x_axis_local)

    x_axis_local, z_axis_local = z_axis_local, -x_axis_local
    

    # print_axis(client = client, pos = pos, rotation_matrix = [x_axis_local, y_axis_local, z_axis_local]) # --> blue (z)
    
    # Create new matrix and euler angles
    rotation_matrix = np.vstack((x_axis_local, y_axis_local, z_axis_local)).T
    euler_angles = rotation_matrix_to_euler_xyz(rotation_matrix)


    # --- DQ ---
    # Position quaternion
    pos.append(0.0)
    pos_Q = get_quaternion(pos)

    # Rotation quaternion
    orn = p.getQuaternionFromEuler(euler_angles)
    orn_Q = get_quaternion(orn)

    # Dual quaternion
    DQ = get_dualQuaternion(q_r=orn_Q, q_t=pos_Q)
    
    return np.array(pos), euler_angles, DQ 


# Computes the reward according the approximation to the object
def approx_reward_EUC(client, object, dist_obj_wrist, robot_id):
    '''
       Computes the reward due to approximation to the object  
                                                                       
           - client: Pybullet client (int)                             
           - object: class that represents an object (must have        
        "id" attribute) (object)
           - dist_obj_wrist: previous distance between the wrist and the object (float)
           - robot_id: id of the robot in the simulation (int)                          
                                             
       Returns:                                                        
           - The new reward for approximation (int, -1 ó 1)
           - The new distance between the object and the wrist
    '''

    
    # Obtains the object and wrist positions
    obj_pos, DQ_obj, DQ_obj_ = get_object_pos(object=object, client = client)
    wrist_pos, DQ_w = get_wrist_pos(client = client, robot_id=robot_id)


    d = np.linalg.norm(wrist_pos[:-1] - obj_pos[:-1])
    theta = [abs(np.dot(r_w, r_obj)/(np.linalg.norm(r_w)*np.linalg.norm(r_obj))) for r_w, r_obj in zip(DQ_w, DQ_obj)]
    theta_ = [abs(np.dot(r_w, r_obj)/(np.linalg.norm(r_w)*np.linalg.norm(r_obj))) for r_w, r_obj in zip(DQ_w, DQ_obj_)]

    theta = max(sum(theta), sum(theta_)) / len(theta)

    d *= 4.5 / math.pi
    theta = (1/theta - 1) / math.pi
    
    r = 1 / (d + theta)
    # d = d.item()
    # theta = theta.item()

    distance = [r, d, theta]

    approx_list = [i < j for i,j in zip(distance, dist_obj_wrist)]
    not_approx = False in approx_list[1:]

    reward = np.tanh(-r)*3 if not_approx else np.tanh(r)*3
    
    # if not approx_list[0]:
    #     if approx_list[-1] or theta < 0.09:
    #         reward += np.tanh(r)

    #     if approx_list[1]:
    #         reward += np.tanh(r)*0.33

    if approx_list[-1]:# or theta < 0.015:
        reward += np.tanh(r)

    if d < 0.07 and theta < 0.015:
        print("AAA")
        reward = r

    
    
    # Updates distance
    dist_obj_wrist = distance

    return reward, dist_obj_wrist


# Computes the reward according the approximation to the object in DQ
def approx_reward(client, object, dist_obj_wrist, robot_id):
    '''
        Computes the reward due to approximation to the object using DQ 
        Input:                                                
           - client (int): Pybullet client                             
           - object (class Object): class that represents an object (must have        
        "id" attribute) 
           - dist_obj_wrist (float): previous distance between the wrist and the object
           - robot_id (id): ID of the robot in the simulation                          
                                             
        Output:                                                        
           - reward (float): reward associated with the distance
           - dist_obj_wrist (float): new distance 
    '''

    # Obtains the object and wrist positions
    obj_pos, __, __, DQ_obj, DQ_obj_ = get_object_pos(object=object, client = client)
    wrist_pos, __, DQ_w = get_wrist_pos(client = client, robot_id=robot_id)

    # Angular distance using quaternion
    d_p = math.acos(2*np.dot(DQ_w[0:4], DQ_obj[0:4]) ** 2 - 1)
    d_p_ = math.acos(2*np.dot(DQ_w[0:4], DQ_obj_[0:4]) ** 2 - 1)

    # Checks which reference system is closer and computes the reward
    if d_p < d_p_:
        r, d, theta = dq_distance(torch.tensor(np.array([DQ_obj])), torch.tensor(np.array([DQ_w])))
    else:
        r, d, theta = dq_distance(torch.tensor(np.array([DQ_obj_])), torch.tensor(np.array([DQ_w])))

    # Extract items and list them
    r = r.item()
    d = d.item()
    theta = theta.item()
    distance = [r, d, theta]

    # Checks if the robot has gotten closer to the object in all dimensions (global, position, orientation)
    approx_list = [i < j for i,j in zip(distance, dist_obj_wrist)]
    not_approx = False in approx_list[1:]

    # Compute the reward
    reward = np.tanh(-r)*3 if not_approx else np.tanh(r)*3
    
    # Bonus for orientation approximation
    if approx_list[-1] or theta < 0.08:
        reward += np.tanh(r)

    # Bouns for getting close
    if d < 0.08 and theta < 0.08:
        reward = r
    
    # Update distance
    dist_obj_wrist = distance

    return reward, dist_obj_wrist


# Check the collision between TWO objects IDs
def check_collision(client, objects):
    '''
        Computes the collisions between a list of two objects   
        Input:                                                    
           - client (int): Pybullet client                           
           - objects (tuple/list/np.array): ID of the objects in the simulation                                
                                             
        Output:                                                        
           - (bool): flag indicating wether they are collisioning or not (boolean)
    '''

    # Collision detector object for both objects
    col_detector = pyb_utils.CollisionDetector(client, [(objects[0], objects[1])])

    # Detects collision with a certain margin
    return col_detector.in_collision(margin = 0.0005)
    

# Computes the reward associated with collision reward
def collision_reward(client, collisions_to_check, mask = np.array([0,0])):
    '''
        Computes the complete reward related to the object collision   
        Input:                                                       
           - client: Pybullet client (int)                             
           - collisions_to_check (list): list of lists with the IDs of the object
        to check collisions
           - mask (np.array): mask of rewards associated with each collision                         
                                             
        Output:                                                        
           - (float): Complete collisionn reward
    '''

    # Obtains the collisions as integers
    checkers = np.array([int(check_collision(client=client, objects = objects)) for objects in collisions_to_check])

    # Applies the mask to the collisions
    return np.sum(checkers * mask)


# Check if the robot is out of bounds
def out_of_bounds(limits, robot):
    '''
        Checks if the robot is out of bounds articularlly
        Input:                                                            
           - limits (list): list of list of limits. Must have in index 0 and 1
        the position and velocities upper and lower bounds for each joint                             
           - object (class UR5e): robot class object                                       
                                             
        Output:                                                        
           - (bool): If the robot is out of bounds returns "True", other way returns "False"
    '''

    # Obtains the joint position of the robot
    qs = [robot.q, robot.qd]
    
    # Iterates the first two limits
    for idx, limit in enumerate(limits[:2]):
        if True in list(limit[0] > qs[idx]) or  \
           True in list(limit[1] < qs[idx]):
            print("Limites ", idx)
            return True

    return False


# Gets the frames
def get_frames(client, camera_params, frame_h, frame_w, frame):
    '''
        Obtains the frames for the cameras in the environment   
        Input:                                                               
           - client (int): Pybullet client                             
           - camera_params (list of np.array 3x3): intrinsec and extrinsic matrices for the cameras
            - frame_h (int): frame height 
            - frame_w (int): frame width 
            - frame (list of np.array, 3xHxW): current list of frames                                      
                                             
        Output:                                                        
           - frame (list of np.array, 3xHxW): list of images captured from each camera
    '''

    # For each camera set of matrices ...
    for idx, camera in enumerate(camera_params):
        # Obtains the view
        camera_info = p.getCameraImage(width = frame_w, 
                                 height = frame_h, 
                                 viewMatrix = camera[0], 
                                 projectionMatrix = camera[1], 
                                 physicsClientId = client)
        
        # RGB buffer
        rgb_buffer = camera_info[2]

        # Gray conversion
        rgb_buffer = cv.cvtColor(rgb_buffer, cv.COLOR_BGR2GRAY)

        # Processed depth image
        depth_buffer = np.reshape(camera_info[3], (frame_h, frame_w))
        
        depth_image = (depth_buffer - np.min(depth_buffer)) / (np.max(depth_buffer) - np.min(depth_buffer))  # Normalize to [0, 1]
        depth_frame = (depth_image * 255).astype(np.uint8)

        # Merges both frames and transpose it --> (channels, gray, depth)
        frame[idx] = cv.merge([rgb_buffer, depth_frame])
        frame[idx] = np.transpose(frame[idx], (2,0,1))

    return frame

# Get information from the environment
def get_info():
    return {}
    