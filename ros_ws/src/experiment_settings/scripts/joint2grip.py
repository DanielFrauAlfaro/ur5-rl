#! /usr/bin/python3

import rospy
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState

joint_names = ["robotiq_finger_1_joint_1", "robotiq_finger_1_joint_2", "robotiq_finger_1_joint_3"]
                       
# "robotiq_finger_2_joint_1", "robotiq_finger_2_joint_2", "robotiq_finger_2_joint_3", "robotiq_finger_middle_joint_1", "robotiq_finger_middle_joint_2", "robotiq_finger_middle_joint_3"]

m1 = 1.2218 /  140.0
m2 = 1.5708 / 100.0

pub = []

def cb(data):
    global m1, m2, joint_names, pub

    joints = []
    
    for i in joint_names:
        index = data.name.index(i)

        joints.push_back(data.position[index])

    g = int(joints[0]/m1)    
    
    pub[0].publsih(g)
            
    

if __name__ == "__main__":
    rospy.init_node("joint2grip")

    pub.push_back(rospy.Publisher("/aurova/grip_state", Int32))
    rospy.Subscriber("/joint_states", JointState, cb)
    
    rospy.spin()