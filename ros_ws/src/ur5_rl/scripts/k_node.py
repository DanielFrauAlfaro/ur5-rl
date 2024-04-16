#! /usr/bin/python3

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from spatialmath import SE3
import roboticstoolbox as rtb
from math import pi
import numpy as np
import copy
import time



# Internal UR5 DH model
ur5 = rtb.DHRobot([
    rtb.RevoluteDH(d=0.1625, alpha=pi/2.0),
    rtb.RevoluteDH(a=-0.425),
    rtb.RevoluteDH(a = -0.3922),
    rtb.RevoluteDH(d = 0.1333, alpha=pi/2.0),
    rtb.RevoluteDH(d = 0.0997, alpha=-pi/2.0),
    rtb.RevoluteDH(d = 0.0996)
], name="UR5e")
ur5.base = SE3.RPY(0,0,pi / 2)

q_init = [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0]
q = [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0]
ee_init = np.array([0.1332997, 0.49, 0.48, 3.14, 0.0, 0.0])
ee = copy.deepcopy(ee_init)
starting_flag = False

def get_q_from_msg(msg, list_of_joints):
    return [msg.position[msg.name.index(joint)] for joint in list_of_joints if joint in msg.name]

def state_cb(data):
    global q, starting_flag

    q = data
    if not starting_flag: starting_flag = True

def ee_cb(data):
    global ee

    ee += data.data

    ee[0:3] *= 0.0666
    ee[3:]  *= 0.12
    ee[-1]  *= 2.5


# Module that implements the robot's forward kinematics
def DK_module(q, list_of_joints):
    global ur5

    # DK module
    q_ = get_q_from_msg(q, list_of_joints)
    T = ur5.fkine(q_, order='yxz')
    ee_pos = T.t.tolist()
    ee_or = T.rpy('yxz').tolist()

    return ee_pos + ee_or


# Module that implemets the robot's inverse kinematics
def IK_module(ee, q_prev):
    global ur5

    x = ee[0]                                 
    y = ee[1]
    z = ee[2]
        
    roll = ee[3]
    pitch = ee[4]
    yaw = ee[5]

    # Builds up homogeneus matrix
    T = SE3(x, y, z)
    T_ = SE3.RPY(roll, pitch, yaw, order='zyx')

    T = T * T_

    # Computes inverse kinematics
    return ur5.ik_LM(T,q0 = q_prev).q


# Kinematics Module
def K_module(pubs, list_of_joints):
    global q, ee

    q_prev = get_q_from_msg(q, list_of_joints)
    ee_msg = Float32MultiArray()
    joint_msg = Float64MultiArray()
    
    rate = rospy.Rate(10)
    t = 0
    t_limit = 40
    while not rospy.is_shutdown():
        
        # Direct Kinematics Module
        ee_msg.data = DK_module(q, list_of_joints)
        pubs["DK"].publish(ee_msg)
        

        # Inverse Kinematics Module

        # joint_msg.data = IK_module(ee, q_prev)
        # pubs["IK"].publish(joint_msg)

        q_prev = q
        t += 1
        if t >= 40:
            t = 0
            move_to_home(pubs)

        rate.sleep()    


def move_to_home(pubs):
    global ee_init, q_init

    joint_msg = Float64MultiArray()
    joint_msg.data = q_init
    for i in range(1): 
        pubs["IK"].publish(joint_msg)


if __name__ == "__main__":
    try:
        # Node
        rospy.init_node("DK_module")

        # List of joint names
        list_of_joints = rospy.get_param("joint_list").split()
        k_to_controller_topic = rospy.get_param("k_to_controller_topic")
        controller_to_k_topic = rospy.get_param("controller_to_k_topic")


        # -- Publishers --
        pubs ={}
        pubs["DK"] = rospy.Publisher(k_to_controller_topic, Float32MultiArray, queue_size=10)
        pubs["IK"] = rospy.Publisher("/ur5_controllers/command", Float64MultiArray, queue_size = 10)

        # -- Subscribers --
        rospy.Subscriber("/joint_states", JointState, state_cb)
        rospy.Subscriber(controller_to_k_topic, Float32MultiArray, ee_cb)
        
        time.sleep(4)
        # Wait until there is data being sent
        while not starting_flag: pass
        move_to_home(pubs)
        


        # Kinematics module
        K_module(pubs, list_of_joints)

    except rospy.ROSInterruptException:
        pass