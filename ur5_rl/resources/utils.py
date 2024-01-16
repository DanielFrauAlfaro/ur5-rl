import numpy as np
import math
import pybullet as p
import pyb_utils
from scipy.spatial.transform import Rotation
import cv2 as cv
import time

# Adds noise to an array
def add_noise(array, std = 0.5):
    return np.random.normal(loc=0, scale=std, size=np.array(array).shape)

# Sets the camera with the class parameters and the desiresd coordiantes
def set_cam(client, fov, aspect, near_val, far_val, cameras_coord, std = 0):
    camera_params = []
    markers = []

    cameras_coord[0][0] += add_noise(cameras_coord[0][0], std = std)
    
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
        view_matrix = p.computeViewMatrix(cameraEyePosition = camera[0], cameraTargetPosition = camera[0] + camera_vec, cameraUpVector = up_vec, physicsClientId = client)
        
        # Computes projection matrix
        proj_matrix = p.computeProjectionMatrixFOV(fov = 80, aspect = 1, nearVal = 0.01, farVal = 1, physicsClientId = client)
        
        # Convert the tuple to a NumPy array and reshape
        proj_matrix_3x3 = np.array(proj_matrix)
        proj_matrix_3x3 = proj_matrix_3x3.reshape(4, 4)[:-1, :-1]

        # Set a the [3][3] value of the matrix to 1 (is at)
        proj_matrix_3x3[-1][-1] = 1


        # Saves parameters
        camera_params.append([view_matrix, proj_matrix, proj_matrix_3x3])
        
        markers.append([])

    return camera_params, markers

# Getter for the object position
def get_object_pos(client, object):
    # ------ Object Position ------
    # Get the position and orientation of the object
    pos, orn = p.getBasePositionAndOrientation(object.id, physicsClientId=client)
    # Convert quaternion to rotation matrix
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orn, physicsClientId = client)).reshape((3, 3))

    # Get the directions of the axes in the object's local coordinate system
    y_axis_local = [0,0,1]
    z_axis_local = rotation_matrix[:, 2]
    x_axis_local = np.cross(z_axis_local, y_axis_local)

    y_aux = y_axis_local
    y_axis_local = z_axis_local
    z_axis_local = y_aux
    
    return np.array(pos), np.array(p.getEulerFromQuaternion(orn, physicsClientId = client))

# Getter for the wrist position
def get_wrist_pos(client, robot_id):
    # -------- Wrist Position -------- --> Name:  tool0_ee_link Joint Index: 12 Link Index: b'ee_link'
    # Get the position and orientation of the ee_link
    link_state = p.getLinkState(robot_id, 11, computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId = client)
    pos, orn = link_state[0], link_state[1]

    rotation_matrix = np.array(p.getMatrixFromQuaternion(orn, physicsClientId = client)).reshape((3, 3))

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

    return np.array(pos), np.array(p.getEulerFromQuaternion(orn, physicsClientId=client))

# Computes the reward according the approximation to the object
def approx_reward(client, object, dist_obj_wrist, robot_id):
    obj_pos, obj_or = get_object_pos(object=object, client = client)
    wrist_pos, wrist_or = get_wrist_pos(client = client, robot_id=robot_id)

    distance = np.linalg.norm(wrist_pos - obj_pos)
    reward = 1 if distance < dist_obj_wrist else -1

    dist_obj_wrist = distance

    return reward, dist_obj_wrist

# Check the collision between TWO objects IDs
def check_collision(client, objects):
    col_detector = pyb_utils.CollisionDetector(client, [(objects[0], objects[1])])

    return col_detector.in_collision(margin = 0.0)
    
# Computes the reward associated with collision reward
def collision_reward(client, collisions_to_check, mask = np.array([0,0])):

    checkers = np.array([int(check_collision(client=client, objects = objects)) for objects in collisions_to_check])

    return np.sum(checkers * mask)


def out_of_bounds(w):
    for warning in w:
        if "method is not within the observation space" in str(warning.message):
            return True
    
    return False

# Get information from the environment
def get_info(frame):
    return {}
    

def get_frames(client, camera_params, frame_h, frame_w, frame):

    # For each camera ...
    for idx, camera in enumerate(camera_params):
        # Obtains the view

        camera_info = p.getCameraImage(width = frame_w, 
                                 height = frame_h, 
                                 viewMatrix = camera[0], 
                                 projectionMatrix = camera[1], 
                                 physicsClientId = client)
        
        rgb_buffer = camera_info[2]

        depth_buffer = np.reshape(camera_info[3], (frame_h, frame_w))

        # Visualize the depth image with a colormap
        depth_image = (depth_buffer - np.min(depth_buffer)) / (np.max(depth_buffer) - np.min(depth_buffer))  # Normalize to [0, 1]

        # Apply a colormap (e.g., 'viridis')
        depth_frame = (depth_image * 255).astype(np.uint8)

        # Gray conversion
        rgb_buffer = cv.cvtColor(rgb_buffer, cv.COLOR_BGR2GRAY)

        frame[idx] = cv.merge([rgb_buffer, depth_frame])
        frame[idx] = np.transpose(frame[idx], (2,0,1))
        break

        

    return frame