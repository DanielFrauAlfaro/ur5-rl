import numpy as np
import math
import pybullet as p
import pyb_utils
from scipy.spatial.transform import Rotation
import cv2 as cv
import time
import copy
# import dqrobotics
from dq import *
import torch

# Adds noise to an array
def add_noise(array, std = 0.5):
    '''
       Adds normal noise to an array               
                                                   
           - array: array to add noise to.         
           - std: Standard deviation to the noise  
                                                   
       Returns:                                    
           - Noised array (np.array)               
    '''

    return np.random.normal(loc=0, scale=std, size=np.array(array).shape)

# Sets the camera with the class parameters and the desiresd coordiantes
def set_cam(client, fov, aspect, near_val, far_val, cameras_coord, std = 0, first = True):
    '''
       Obtains the intrinsic and extrinsic parameters of cameras       
                                                                       
           - client: Pybullet client (int)                             
           - fov: field of view (int)                                  
           - aspect: aspect of the cameras (int)                       
           - near_val: nearest value for depth camera (int)            
           - far_val: farthest value for depth camera (int)            
           - cameras_coord: list of arrays of positions for            
       the camera (list of list / arrays) (XYZ - RPY)                  
           - std: standard deviation for the noise                     
                                                                       
       Returns:                                                        
           - A list of two matrix: intrinsic and extrinsic parameters  
    '''    

    # PArameters for the cameras
    camera_params = []

    # Adds noise to the camera coordinates (just the first one)
    cameras_coord_aux = copy.deepcopy(cameras_coord)
    if True:
        cameras_coord_aux[0][0] += add_noise(cameras_coord_aux[0][0], std = std)
        cameras_coord_aux[1][0] += add_noise(cameras_coord_aux[0][0], std = std)
    
    # For each camera ...
    for camera in cameras_coord_aux:
        # Obtain rotations
        rot_x = np.array([[1, 0, 0], [0, math.cos(camera[1][0]), -math.sin(camera[1][0])], [0, math.sin(camera[1][0]), math.cos(camera[1][0])]])
        rot_y = np.array([[math.cos(camera[1][1]),0, math.sin(camera[1][1])], [0, 1, 0], [-math.sin(camera[1][1]),0,math.cos(camera[1][1])]])
        rot_z = np.array([[math.cos(camera[1][2]), -math.sin(camera[1][2]), 0], [math.sin(camera[1][2]), math.cos(camera[1][2]), 0], [0, 0, 1]])
        
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

        # Saves parameters
        camera_params.append([view_matrix, proj_matrix, proj_matrix_3x3])

    return camera_params


# Shows axis on the world
def print_axis(client, pos, rotation_matrix):
    axis_length = 0.1
    x_axis, y_axis, z_axis = [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]

    # Get the directions of the axes in the object's local coordinate system
    # y_axis_local = [0,0,1]
    # z_axis_local = rotation_matrix[:, 2]
    # x_axis_local = np.cross(z_axis_local, y_axis_local)

    # y_aux = y_axis_local
    # y_axis_local = z_axis_local
    # z_axis_local = y_aux

    x_axis_local = rotation_matrix[0]
    y_axis_local = rotation_matrix[1]
    z_axis_local = rotation_matrix[2]

    

    # Draw lines representing the axes of the object
    line_start = pos
    line_end_x = [pos[0] + 0.5 * x_axis_local[0], pos[1] + 0.5 * x_axis_local[1], pos[2] + 0.5 * x_axis_local[2]]
    line_end_y = [pos[0] + 0.5 * y_axis_local[0], pos[1] + 0.5 * y_axis_local[1], pos[2] + 0.5 * y_axis_local[2]]
    line_end_z = [pos[0] + 0.5 * z_axis_local[0], pos[1] + 0.5 * z_axis_local[1], pos[2] + 0.5 * z_axis_local[2]]

    p.addUserDebugLine(line_start, line_end_x, [1, 0, 0], lifeTime=0.3, physicsClientId = client)  # X-axis (red)
    p.addUserDebugLine(line_start, line_end_y, [0, 1, 0], lifeTime=0.3, physicsClientId = client)  # Y-axis (green)
    p.addUserDebugLine(line_start, line_end_z, [0, 0, 1], lifeTime=0.3, physicsClientId = client)  # Z-axis (blue)


# Get the rotation Euler angles from the rotation matrix
def rotation_matrix_to_euler_xyz(R):
    '''
    Obtains the Euler angles from a 3x3 rotation matrix

        - R: 3x3 rotation matrix (lists // numpy matrix)

    Returns:
        - Array of Euler angles (numpy array)
    '''

    theta_y = np.arcsin(-R[0, 2])
    theta_x = np.arctan2(R[1, 2], R[2, 2])
    theta_z = np.arctan2(R[0, 1], R[0, 0])

    return np.array([theta_x, theta_y, theta_z])


def get_quaternion(q):
    return np.array([q[-1], q[0], q[1], q[2]])

def get_dualQuaternion(q_r, q_t):
    return np.concatenate((q_r, 0.5*q_mul(torch.tensor(q_t).unsqueeze(dim=0), torch.tensor(q_r).unsqueeze(dim=0)).squeeze(dim=0).numpy()))



# Getter for the object position
def get_object_pos(client, object):
    '''
       Obtains the position and orientantion of a given object class   
                                                                       
           - client: Pybullet client (int)                             
           - object: class that represents an object (must have        
        "id" attribute) (object)                                       
                                             
       Returns:                                                        
           - Two arrays, the position (XYZ) and orientation (RPY)
    '''

    # Get the position and orientation of the object
    pos, orn = p.getBasePositionAndOrientation(object.id, physicsClientId=client)
    # Convert quaternion to rotation matrix
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orn, physicsClientId = client)).reshape((3, 3))
    
    x_axis_local = rotation_matrix[:,0] / np.linalg.norm(rotation_matrix[:,0])
    y_axis_local = rotation_matrix[:,1] / np.linalg.norm(rotation_matrix[:,1])
    z_axis_local = rotation_matrix[:,2] / np.linalg.norm(rotation_matrix[:,2])

    # y_axis_local[0] = math.cos(math.pi / 2.0) * z_axis_local[0] - math.sin(math.pi / 2.0) * z_axis_local[1]
    # y_axis_local[1] = math.sin(math.pi / 2.0) * z_axis_local[0] + math.cos(math.pi / 2.0) * z_axis_local[1]
    # y_axis_local[2] = z_axis_local[2]

    # x_axis_local = np.cross(z_axis_local, y_axis_local)
    # x_axis_local /= np.linalg.norm(x_axis_local)


    # roll, pitch, yaw = np.radians(0), np.radians(0), np.radians(0)
    # my_rotation_matrix = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

    # new_matrix = rotation_matrix.transpose()*my_rotation_matrix*rotation_matrix
    # y_axis_local = np.matmul(new_matrix, z_axis_local)
    # y_axis_local /= np.linalg.norm(y_axis_local)

    # x_axis_local = new_matrix[:,0] / np.linalg.norm(new_matrix[:,0])
    # y_axis_local = new_matrix[:,1] / np.linalg.norm(new_matrix[:,1])
    # z_axis_local = new_matrix[:,2] / np.linalg.norm(new_matrix[:,2])


    # auto new_rot = matAux.transpose()*rotpos90x*rotneg45z*matAux;
    # endLinkX = new_rot*endLinkX;
    # endLinkY = new_rot*endLinkY;
    # endLinkZ = new_rot*endLinkZ;

    pos = list(pos)

    if "cracker" in object.name:
        pos[-1] += 0.26
    else:
        pos[-1] += 0.215 # * x_axis_local

    x_axis_local = np.array([0, 0, -1])
    y_axis_local = np.cross(z_axis_local, x_axis_local)

    if "box" in object.name:
        z_axis_local, y_axis_local = -y_axis_local, z_axis_local

    z_axis_local_ = -1*z_axis_local
    y_axis_local_ = np.cross(z_axis_local_, x_axis_local)

    


    rotation_matrix = np.vstack((x_axis_local, y_axis_local, z_axis_local)).T
    rotation_matrix_ = np.vstack((x_axis_local, y_axis_local_, z_axis_local_)).T
    euler_angles = rotation_matrix_to_euler_xyz(rotation_matrix)
    euler_angles_ = rotation_matrix_to_euler_xyz(rotation_matrix_)
    
    # print_axis(client = client, pos = pos, rotation_matrix = [x_axis_local, y_axis_local, z_axis_local])
    # print_axis(client = client, pos = pos, rotation_matrix = [x_axis_local, y_axis_local_, z_axis_local_])

    pos = list(pos)
    pos.append(0.0)
    pos_Q = get_quaternion(pos)
    orn = p.getQuaternionFromEuler(euler_angles)
    orn_Q = get_quaternion(orn)
    DQ = get_dualQuaternion(q_r=orn_Q, q_t=pos_Q)


    orn_ = p.getQuaternionFromEuler(euler_angles_)
    orn_Q_ = get_quaternion(orn_)
    DQ_ = get_dualQuaternion(q_r=orn_Q_, q_t=pos_Q)

     # --> blue (z)
    
    return np.array(pos), euler_angles, euler_angles_, DQ, DQ_# rotation_matrix# euler_angles # x_axis_local  + y_axis_local + z_axis_local# z_axis_local#np.array(p.getEulerFromQuaternion(orn, physicsClientId = client))


# Getter for the wrist position
def get_wrist_pos(client, robot_id):
    '''
       Obtains the position and orientantion of a given joint in the robot object class   
                                                                       
           - client: Pybullet client (int)                             
           - robot_id: id of the toot in the simulation (int)   
                                             
       Returns:                                                        
           - Two arrays, the position (XYZ) and orientation (RPY)
    '''
    # -------- Wrist Position -------- --> Name:  tool0_ee_link Joint Index: 12 Link Index: b'ee_link'
    # Get the position and orientation of the ee_link
    link_state = p.getLinkState(robot_id, 11, computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId = client)
    pos, orn = link_state[0], link_state[1]
    # pos[-1] += 0.01

    # ----- Extra code for changing axis ------
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orn, physicsClientId = client)).reshape((3, 3))

    x_axis_local = rotation_matrix[:,0] / np.linalg.norm(rotation_matrix[:,0])
    y_axis_local = rotation_matrix[:,1] / np.linalg.norm(rotation_matrix[:,1])
    z_axis_local = rotation_matrix[:,2] / np.linalg.norm(rotation_matrix[:,2])

    x_axis_local = x_axis_local + y_axis_local
    x_axis_local /= np.linalg.norm(x_axis_local)
    y_axis_local = np.cross(z_axis_local, x_axis_local)

    x_axis_local, z_axis_local = z_axis_local, -x_axis_local
    
    # print_axis(client = client, pos = pos, rotation_matrix = [x_axis_local, y_axis_local, z_axis_local]) # --> blue (z)
    
    rotation_matrix = np.vstack((x_axis_local, y_axis_local, z_axis_local)).T
    euler_angles = rotation_matrix_to_euler_xyz(rotation_matrix)

    pos = list(pos)
    pos.append(0.0)
    pos_Q = get_quaternion(pos)

    orn = p.getQuaternionFromEuler(euler_angles)
    orn_Q = get_quaternion(orn)

    DQ = get_dualQuaternion(q_r=orn_Q, q_t=pos_Q)
    
    # my_rotation = rotation_matrix
    # # Euler angles in radians (replace with your actual values)
    # roll, pitch, yaw = np.radians(0), np.radians(0), np.radians(45)

    # # Create a rotation matrix from Euler angles
    # rotation_matrix = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

    # # Rotate the vector using the rotation matrix
    # x_axis_local = np.dot(rotation_matrix, x_axis_local)
    # y_axis_local = np.dot(rotation_matrix, y_axis_local)
    # x_axis_local *= -1

    # y_axis_local, z_axis_local = z_axis_local, y_axis_local
    # y_axis_local, x_axis_local = x_axis_local, y_axis_local
    # y_axis_local = np.cross(z_axis_local, x_axis_local)

    # rotation_matrix = np.vstack((x_axis_local, y_axis_local, z_axis_local)).T
    # euler_angles = rotation_matrix_to_euler_xyz(rotation_matrix)

    return np.array(pos), euler_angles, DQ #rotation_matrix# euler_angles #x_axis_local  + y_axis_local + z_axis_local# z_axis_local, y_axis_local# np.array(p.getEulerFromQuaternion(orn, physicsClientId=client))


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

# Computes the reward according the approximation to the object
def approx_reward(client, object, dist_obj_wrist, robot_id):
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
    obj_pos, __, __, DQ_obj, DQ_obj_ = get_object_pos(object=object, client = client)
    wrist_pos, __, DQ_w = get_wrist_pos(client = client, robot_id=robot_id)

    # Primary parts
    # p_w = dqrobotics.P(DQ_w)
    # p_obj = dqrobotics.P(DQ_obj)
    # p_obj_ = dqrobotics.P(DQ_obj_)

    # # Dual parts
    # d_w = dqrobotics.D(DQ_w)
    # d_obj = dqrobotics.D(DQ_obj)
    # d_obj_ = dqrobotics.D(DQ_obj_)

    # # Vectors
    # # --- Dual quaternion vectors ---
    # w_DQ_vec = dqrobotics.vec8(DQ_w)
    # obj_DQ_vec = dqrobotics.vec8(DQ_obj)
    # obj_DQ_vec_ = dqrobotics.vec8(DQ_obj_)

    # # --- Primary parts vector ---
    # p_w_vec = dqrobotics.vec4(p_w)
    # p_obj_vec = dqrobotics.vec4(p_obj)
    # p_obj_vec_ = dqrobotics.vec4(p_obj_)

    # # --- Dual parts vector ---
    # d_w_vec = dqrobotics.vec4(d_w)
    # d_obj_vec = dqrobotics.vec4(d_obj)
    # d_obj_vec_ = dqrobotics.vec4(d_obj_)

    # Angular distance using quaternion
    d_p = math.acos(2*np.dot(DQ_w[0:4], DQ_obj[0:4]) ** 2 - 1)
    d_p_ = math.acos(2*np.dot(DQ_w[0:4], DQ_obj_[0:4]) ** 2 - 1)


    if d_p < d_p_:
        r, d, theta = dq_distance(torch.tensor(np.array([DQ_obj])), torch.tensor(np.array([DQ_w])))
    else:
        r, d, theta = dq_distance(torch.tensor(np.array([DQ_obj_])), torch.tensor(np.array([DQ_w])))


    r = r.item()
    d = d.item()
    theta = theta.item()

    distance = [r, d, theta]

    approx_list = [i < j for i,j in zip(distance, dist_obj_wrist)]
    not_approx = False in approx_list[1:]

    reward = np.tanh(-r)*3 if not_approx else np.tanh(r)*3
    
    # if not approx_list[0]:
    #     if approx_list[-1] or theta < 0.09:
    #         reward += np.tanh(r)

    #     if approx_list[1]:
    #         reward += np.tanh(r)*0.33

    if approx_list[-1] or theta < 0.08:
        reward += np.tanh(r)

    if d < 0.08 and theta < 0.08:
        # print("AAA")
        reward = r
    
    
    # Updates distance
    dist_obj_wrist = distance

    return reward, dist_obj_wrist



# Check the collision between TWO objects IDs
def check_collision(client, objects):
    '''
       Computes the collisions between a list of two objects   
                                                                       
           - client: Pybullet client (int)                             
           - objects: id of the objects in the simulation (int)                                  
                                             
       Returns:                                                        
           - A boolean indicating wether they are collisioning or not (boolean)
    '''

    # Collision detector object for both objects
    col_detector = pyb_utils.CollisionDetector(client, [(objects[0], objects[1])])

    # Detects collision with a certain margin (0.0)
    return col_detector.in_collision(margin = 0.0005)
    

# Computes the reward associated with collision reward
def collision_reward(client, collisions_to_check, mask = np.array([0,0])):
    '''
       Computes the complete reward related to the object collision   
                                                                       
           - client: Pybullet client (int)                             
           - collisions_to_check: list of list with the ids of the object
        to check collisions (list of lists of int)
           - mask: mask of rewards associated with each collision (np.array)                        
                                             
       Returns:                                                        
           - Complete collisionn reward (int / float)
    '''

    # Obtains the collisions as integers
    checkers = np.array([int(check_collision(client=client, objects = objects)) for objects in collisions_to_check])

    # Applies the mask to the collisions
    return np.sum(checkers * mask)


# Check if the robot is out of bounds
def out_of_bounds(limits, robot):
    '''
       Checks if the robot is out of bounds articularlly
                                                                       
           - limits: list of list of limits. Must have in index 0 and 1
        the position and velocities upper and lower bounds for each joint                             
           - object: robot class object (object)                                       
                                             
       Returns:                                                        
           -If the robot is out of bounds (bool)
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
                                                                       
           - client: Pybullet client (int)                             
           - camera_params: intrinsec and extrinsic matrices for the cameras
        (list of two matrices)
            - frame_h: frame height (int)
            - frame_w: frame width (int)
            - frame: current list of frames                                      
                                             
       Returns:                                                        
           - Two arrays, the position (XYZ) and orientation (RPY)
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

        # Computes images for the first one (external)
        # break
    
    return frame

# Get information from the environment
def get_info():
    return {}
    