import numpy as np
import torch
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Quaternion, Pose, Point


pos, orn = np.random.uniform([[0.05, 0.5, 0.85], [-np.pi, -np.pi/2, -3.1415]], 
                             [[0.275,  0.62, 0.85], [np.pi,  -np.pi/2,  3.1415]])

quat_tf = quaternion_from_euler(orn[0], orn[1], orn[2])

q_msg = Quaternion(quat_tf[0], quat_tf[1], quat_tf[2], quat_tf[3])

point = Point(pos[0], pos[1], pos[2])
pose = Pose()
pose.position = point
pose.orientation = q_msg
print(pose)