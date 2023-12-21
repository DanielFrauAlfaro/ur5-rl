import pybullet as p
from math import pi, cos, sin
import numpy as np

p.connect(p.GUI)
#Load your simulation scene and objects as needed

# Set the desired camera resolution
width = 200
height = 200

# Set the camera parameters
fov = 60  # Field of view in degrees
near_plane = 0.02  # Near clipping plane
far_plane = 5.0  # Far clipping plane

# Set the camera pose (position and orientation)
camera_position = [0.2, 1.1, 1.1]
orientation = [0.5, 0.0, -pi/2]

rot_x = np.array([[1, 0, 0], [0, cos(orientation[0]), -sin(orientation[0])], [0, sin(orientation[0]), cos(orientation[0])]])
rot_y = np.array([[cos(orientation[1]),0, sin(orientation[1])], [0, 1, 0], [-sin(orientation[1]),0,cos(orientation[1])]])
rot_z = np.array([[cos(orientation[2]), -sin(orientation[2]), 0], [sin(orientation[2]), cos(orientation[2]), 0], [0, 0, 1]])

rot_mat = np.matmul(np.matmul(rot_x, rot_y), rot_z)
camera_vec = np.matmul(rot_mat, [1, 0, 0])
camera_up_vector = np.matmul(rot_mat, np.array([0, 0, 1]))

camera_target = [0, 0, 0]

# Set the camera view matrix
view_matrix = p.computeViewMatrix(
    cameraEyePosition=camera_position,
    cameraTargetPosition=camera_position + camera_vec,
    cameraUpVector=camera_up_vector
)

# Set the camera projection matrix
proj_matrix = p.computeProjectionMatrixFOV(
    fov=fov,
    aspect=float(width) / height,
    nearVal=near_plane,
    farVal=far_plane
)

# Update the visualization
# p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

# Get the camera information
camera_info = p.getDebugVisualizerCamera()

# Extract the intrinsic parameters
intrinsic_matrix = camera_info



print(intrinsic_matrix)
print("\n\n\n")

# Disconnect from PyBullet
p.disconnect()