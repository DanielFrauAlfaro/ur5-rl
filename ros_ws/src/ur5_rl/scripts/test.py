#! /usr/bin/python3

import rospy
from stable_baselines3 import SAC
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from ur5_rl.msg import ManageObjectAction, ManageObjectGoal
from cv_bridge import CvBridge
import numpy as np
import cv2
import time
import copy
import actionlib
from spatialmath import SE3
import roboticstoolbox as rtb
from math import pi



frame_h = 160
frame_w = 160

bridge = CvBridge()
images = {}
x = {}  

images["frontal"] = [np.ones((frame_w, frame_h), dtype=np.int16), 
                     np.ones((frame_w, frame_h), dtype=np.int16)]

images["side"] = [np.ones((frame_w, frame_h), dtype=np.int16), 
                  np.ones((frame_w, frame_h), dtype=np.int16)]

images["robot"] = [np.ones((frame_w, frame_h), dtype=np.int16), 
                   np.ones((frame_w, frame_h), dtype=np.int16)]

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

q_init = [-0.004, -1.549, -1.547, -pi/2.0, pi/2.0, pi/2.0]
q = [-0.004, -1.549, -1.547, -pi/2.0, pi/2.0, pi/2.0]
ee_init = np.array([0.1332997, 0.49, 0.48, 3.14, 0.0, 0.0])
ee = copy.deepcopy(ee_init)
starting_flag = False

list_of_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def get_q_from_msg(msg, list_of_joints):
    return [msg.position[msg.name.index(joint)] for joint in list_of_joints if joint in msg.name]

def state_cb(data):
    global q, starting_flag

    q = data
    
    if not starting_flag: starting_flag = True


def image_cb(data):
    global bridge, images, x, frame_w, frame_h

    if "color" in data.header.frame_id:
        if "camera" in data.header.frame_id:
            images["robot"][0] = bridge.imgmsg_to_cv2(data)

            # Convert color image to grayscale
            images["robot"][0] = cv2.cvtColor(images["robot"][0], cv2.COLOR_BGR2GRAY)

            # Resize grayscale image to 160x160
            images["robot"][0] = cv2.resize(images["robot"][0], (frame_h, frame_w))
            images["robot"][0] = (images["robot"][0] - np.min(images["robot"][0])) / float(max(1.0, (np.max(images["robot"][0]) - np.min(images["robot"][0]))))


        elif "frontal" in data.header.frame_id:
            images["frontal"][0] = bridge.imgmsg_to_cv2(data)

            # Convert color image to grayscale
            images["frontal"][0] = cv2.cvtColor(images["frontal"][0], cv2.COLOR_BGR2GRAY)

            # Resize grayscale image to 160x160
            images["frontal"][0] = cv2.resize(images["frontal"][0], (frame_h, frame_w))
            images["frontal"][0] = (images["frontal"][0] - np.min(images["frontal"][0])) / float(max(1.0, (np.max(images["frontal"][0]) - np.min(images["frontal"][0])))) 

        else:
            images["side"][0] = bridge.imgmsg_to_cv2(data)

            # Convert color image to grayscale
            images["side"][0] = cv2.cvtColor(images["side"][0], cv2.COLOR_BGR2GRAY)

            # Resize grayscale image to 160x160
            images["side"][0] = cv2.resize(images["side"][0], (frame_h, frame_w))
            images["side"][0] = (images["side"][0] - np.min(images["side"][0])) / float(max(1.0, (np.max(images["side"][0]) - np.min(images["side"][0]))))


    elif "depth" in data.header.frame_id:
        if "camera" in data.header.frame_id:
            images["robot"][1] = bridge.imgmsg_to_cv2(data)
            
            
            images["robot"][1] = cv2.resize(images["robot"][1], (frame_h, frame_w))
            images["robot"][1] = (images["robot"][1] - np.min(images["robot"][1])) / float(max(1.0, (np.max(images["robot"][1]) - np.min(images["robot"][1]))))
            images["robot"][1] = (images["robot"][1] * 255).astype(np.uint8)
            
            images["robot"][1] = (images["robot"][1] - np.min(images["robot"][1])) / float(max(1.0, (np.max(images["robot"][1]) - np.min(images["robot"][1]))))  


        elif "frontal" in data.header.frame_id:
            images["frontal"][1] = bridge.imgmsg_to_cv2(data)
            
            # Resize grayscale image to 160x160
            images["frontal"][1] = cv2.resize(images["frontal"][1], (frame_h, frame_w))
            images["frontal"][1] = (images["frontal"][1] - np.min(images["frontal"][1])) / float(max(1.0,(np.max(images["frontal"][1]) - np.min(images["frontal"][1]))))
            images["frontal"][1] = (images["frontal"][1] * 255).astype(np.uint8)
            
            images["frontal"][1] = (images["frontal"][1] - np.min(images["frontal"][1])) / float(max(1.0, (np.max(images["frontal"][1]) - np.min(images["frontal"][1])))) 
    
        else:
            images["side"][1] = bridge.imgmsg_to_cv2(data)

            # Resize grayscale image to 160x160
            images["side"][1] = cv2.resize(images["side"][1], (frame_h, frame_w))
            images["side"][1] = (images["side"][1] - np.min(images["side"][1])) / float(max(1.0,(np.max(images["side"][1]) - np.min(images["side"][1]))))
            images["side"][1] = (images["side"][1] * 255).astype(np.uint8)
            
            images["side"][1] = (images["side"][1] - np.min(images["side"][1])) / float(max(1.0, (np.max(images["side"][1]) - np.min(images["side"][1]))))



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
def IK_module(ee, q_prev, list_of_joints):
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
    q_prev = get_q_from_msg(msg = q_prev, list_of_joints=list_of_joints)

    q = ur5.ik_LM(T,q0 = q_prev)
    return q[0]

def move_to_home(pubs):
    global ee_init, q_init

    joint_msg = Float64MultiArray()
    joint_msg.data = q_init
    pubs["IK"].publish(joint_msg)

    # print("SE MANDE EL MENSAJE DE HOME\n\n")

    time.sleep(2)

def spawn_object(spawn_client, spawned = True):

    if spawned:
        goal = ManageObjectGoal()
        goal.spawn = False

        spawn_client.send_goal(goal)
        spawn_client.wait_for_server(rospy.Duration.from_sec(5.0))

    goal = ManageObjectGoal()
    goal.spawn = True

    spawn_client.send_goal(goal)
    spawn_client.wait_for_server(rospy.Duration.from_sec(5.0))

    spawned = True

def update_obs():
    global images, q

    x = {}
    flag = True
    try:   
        # print(np.unique(images["side"][1]))
        merged = cv2.merge([images["frontal"][0], images["frontal"][1],
                        images["side"][0], images["side"][1],
                        images["robot"][0], images["robot"][1]])
        merged = np.transpose(merged, (2,0,1))
        x["image"] = merged
    except:
        flag = False
        

    x["ee_position"] = DK_module(q, list_of_joints)

    return x, flag


def rl_controller(pubs, model_path):
    global x

    # Model loading
    model = SAC.load(model_path)      
    model.policy.eval()

    rate = rospy.Rate(10)

    q_prev = q
    joint_msg = Float64MultiArray()

    spawn_client = actionlib.SimpleActionClient('/manage_object', ManageObjectAction)
    spawn_client.wait_for_server()
    move_to_home(pubs)
    
    t = 0
    t_limit = 40
    spawn_object(spawn_client=spawn_client, spawned=False)

    while not rospy.is_shutdown():
        x, flag = update_obs()

        if flag:
            action, __   = model.predict(x, deterministic = True)
            
            action[0:3] *= 0.0666 * 0.25
            action[3:]  *= 0.12 * 0.25
            action[-1]  *= 2.5 * 0.25
            print(action)
            action += x["ee_position"]
            print(action)
            
            # print(action[2])
            # print("-----")
            joint_msg.data = IK_module(action, q_prev, list_of_joints)
            pubs["IK"].publish(joint_msg)
                

        q_prev = q
        t += 1

        # print(joint_msg)
        print("---")

        if t >= t_limit or x["ee_position"][2] < 0.36:
            
            
            # spawn_object(spawn_client=spawn_client)
            # time.sleep(3)

            move_to_home(pubs)
            print("START?")
            __ = input()
            t = 0


        rate.sleep()






if __name__ == "__main__":
    try:
        rospy.init_node("rl_controller")

        model_path = rospy.get_param("model_path")
        k_to_controller_topic = rospy.get_param("k_to_controller_topic")
        controller_to_k_topic = rospy.get_param("controller_to_k_topic")
        
        # -- Publishers --
        pubs ={}
        pubs["DK"] = rospy.Publisher(k_to_controller_topic, Float32MultiArray, queue_size=10)
        pubs["IK"] = rospy.Publisher("/ur5_controllers/command", Float64MultiArray, queue_size = 10)

        # -- Subscribers
        # rospy.Subscriber(k_to_controller_topic, Float32MultiArray, ee_cb)
        rospy.Subscriber("/joint_states", JointState, state_cb)

        rospy.Subscriber("/camera_robot/color/image_raw", Image, image_cb)
        rospy.Subscriber("/camera_frontal/color/image_raw", Image, image_cb)
        rospy.Subscriber("/camera_side/color/image_raw", Image, image_cb)
        rospy.Subscriber("/camera_robot/depth/image_raw", Image, image_cb)
        rospy.Subscriber("/camera_frontal/depth/image_raw", Image, image_cb)
        rospy.Subscriber("/camera_side/depth/image_raw", Image, image_cb)
        
        while not starting_flag: pass
        move_to_home(pubs)
        time.sleep(5)

        rl_controller(pubs, model_path)


    except rospy.ROSInterruptException:
        pass

