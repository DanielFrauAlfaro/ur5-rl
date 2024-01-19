#!/usr/bin/env python3

import pybullet as p
import pybullet_data
import collections
import matplotlib.pyplot as plt
import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb
from math import pi
import os

ur5 = rtb.DHRobot([
            rtb.RevoluteDH(d=0.1625, alpha=pi/2.0),
            rtb.RevoluteDH(a=-0.425),
            rtb.RevoluteDH(a = -0.3922),
            rtb.RevoluteDH(d = 0.1333, alpha=pi/2.0),
            rtb.RevoluteDH(d = 0.0997, alpha=-pi/2.0),
            rtb.RevoluteDH(d = 0.0996)
        ], name="UR5e")
    
ur5.base = SE3.RPY(0,0,pi / 2)      # Rotate robot base so it matches Gazebo model
# ur5.tool = SE3(0.0, 0.0, 0.03)

# Define GUI elements
def user_interface(): # [ 0.49586404 -0.11770215  0.48789221]    //      [-2.38737814e+00 -3.04939623e-05 -3.14158786e+00]  --> Robotic Toolbox
                      # (0.13295353284875722, 0.4919942088645094, 1.1878922101017253)
                      # [0.0, -1.5708, -1.5708, -1.5708, 1.5708, -0.785 + pi]

    shoulder_pan_gui = p.addUserDebugParameter('Shoulder pan', -1.5, 1.5, 0)
    shoulder_lift_gui = p.addUserDebugParameter('Shoulder lift', -3.14, 0.0, -1.5708)
    elbow_gui = p.addUserDebugParameter('Elbow', -3.14, 0.0, -1.5708)
    wrist_1_gui = p.addUserDebugParameter('Wrist 1', -3.14, 0.0, -1.5708)
    wrist_2_gui = p.addUserDebugParameter('Wrist 2', 0.0, 3.14, 1.5708)
    wrist_3_gui = p.addUserDebugParameter('Wrist 3', -3.1415, 3.1415, -0.785 + pi)

    gripper_1 = p.addUserDebugParameter('Finger 1', 0, 1.2, 0)
    gripper_2 = p.addUserDebugParameter('Finger 2', 0, 1.2, 0)
    gripper_mid = p.addUserDebugParameter('Finger Middle', 0, 1.2, 0)

    gripper_palm = p.addUserDebugParameter('Palm', 1, 0, 1)

    x_gui = p.addUserDebugParameter('X', -1.5, 1.5, 0)
    y_gui = p.addUserDebugParameter('Y', -1.5, 1.5, 0.0)
    z_gui = p.addUserDebugParameter('Z', -1.5, 1.5, 0.0)
    roll_gui = p.addUserDebugParameter('Roll', -1.5, 1.5, 0)
    pitch_gui = p.addUserDebugParameter('Pitch', -1.5, 1.5, 0)
    yaw_gui = p.addUserDebugParameter('Yaw', -1.5, 1.5, 0)

    return [shoulder_pan_gui, shoulder_lift_gui, elbow_gui, wrist_1_gui, wrist_2_gui, wrist_3_gui, gripper_1, gripper_2, gripper_mid, gripper_palm,
            x_gui, y_gui, z_gui, roll_gui, pitch_gui, yaw_gui]

maxForce = 100
v = 100

# Set robot values
def set_joints(ur5_id, j):

    # pos = p.calculateInverseKinematics(ur5_id, 6, (j[0], j[1], j[2]), (j[3], j[4], j[5]))

    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=1 + 2, controlMode=p.POSITION_CONTROL, targetPosition=j[0])
    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=2 + 2, controlMode=p.POSITION_CONTROL, targetPosition=j[1])
    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=3 + 2, controlMode=p.POSITION_CONTROL, targetPosition=j[2])
    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=4 + 2, controlMode=p.POSITION_CONTROL, targetPosition=j[3])
    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=5 + 2, controlMode=p.POSITION_CONTROL, targetPosition=j[4])
    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=6 + 2, controlMode=p.POSITION_CONTROL, targetPosition=j[5])

    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=11, controlMode=p.POSITION_CONTROL, targetPosition=j[6])
    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=15, controlMode=p.POSITION_CONTROL, targetPosition=j[7])
    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=19, controlMode=p.POSITION_CONTROL, targetPosition=j[8])

    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=10, controlMode=p.POSITION_CONTROL, targetPosition=j[9])

def set_cart(ur5_id, pos, robot):


    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=1 + 2, controlMode=p.POSITION_CONTROL, targetPosition=j[0])
    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=2 + 2, controlMode=p.POSITION_CONTROL, targetPosition=j[1])
    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=3 + 2, controlMode=p.POSITION_CONTROL, targetPosition=j[2])
    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=4 + 2, controlMode=p.POSITION_CONTROL, targetPosition=j[3])
    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=5 + 2, controlMode=p.POSITION_CONTROL, targetPosition=j[4])
    p.setJointMotorControl2(bodyUniqueId=ur5_id, jointIndex=6 + 2, controlMode=p.POSITION_CONTROL, targetPosition=j[5])



# Read GUI elements
def read_gui(gui_joints):
    j1 = p.readUserDebugParameter(gui_joints[0])
    j2 = p.readUserDebugParameter(gui_joints[1])
    j3 = p.readUserDebugParameter(gui_joints[2])
    j4 = p.readUserDebugParameter(gui_joints[3])
    j5 = p.readUserDebugParameter(gui_joints[4])
    j6 = p.readUserDebugParameter(gui_joints[5])
    
    g1 = p.readUserDebugParameter(gui_joints[6])
    g2 = p.readUserDebugParameter(gui_joints[7])
    gm = p.readUserDebugParameter(gui_joints[8])

    gp = p.readUserDebugParameter(gui_joints[9])

    return [j1, j2, j3, j4, j5, j6, g1, g2, gm, gp]


# Spawn environment
def spawn_environment(id):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    p.setGravity(0, 0, -10, physicsClientId=id) # Earth gravity

    ur5_id = p.loadURDF(fileName=os.path.dirname(__file__) + '/ur5_rl/resources/models/models/ur5_env.urdf',
                              basePosition=[0, 0, 0], 
                              physicsClientId=client, 
                              useFixedBase=1)
    return ur5_id


# Process all model joints: obtain data of each joint
def setup_robot(robotID):
    # Name of the gripper and UR5 joint (all joints are included on the same body)
    gripperJoints = ["robotiq_finger_1_joint_1", "robotiq_finger_2_joint_1", "robotiq_finger_middle_joint_1"]
    gripperMimicJoints = ["robotiq_finger_1_joint_1", "robotiq_finger_1_joint_2", "robotiq_finger_1_joint_3", 
                            "robotiq_finger_2_joint_1", "robotiq_finger_2_joint_2", "robotiq_finger_2_joint_3", 
                            "robotiq_finger_middle_joint_1", "robotiq_finger_middle_joint_2", "robotiq_finger_middle_joint_3"]
    controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
                    "elbow_joint", "wrist_1_joint",
                    "wrist_2_joint", "wrist_3_joint"]
    palmJoints = ["palm_finger_1_joint", "palm_finger_2_joint"]
    
    # Get info from joints and enable joint torque sensor
    joints = []
    ur5_joints_id = []
    gripper_joints_id = []
    gripper_mimic_joints_id = []
    palm_joint_id = []
    numJoints = p.getNumJoints(robotID)
    jointInfo = collections.namedtuple("jointInfo",["id","name","type",'damping','friction',"lowerLimit","upperLimit","maxForce","maxVelocity","controllable"])
    
    list_attr = {}

    # Iterates for each joint
    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = info[2]
        jointDamping = info[6]
        jointFriction = info[7]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = (jointType != p.JOINT_FIXED)
        
        list_attr[jointName] = jointID

        print("Name: ", jointName, "Joint Index:", i, "Link Index:", info[12])
        print("--")


        # If a joint is controllable ...
        if controllable:
            # ... enables torque sensors, ...
            # p.enableJointForceTorqueSensor(bodyUniqueId=id, 
            #                                jointIndex=jointID, 
            #                                enableSensor=1,
            #                                physicsClientId=client)

            # ... saves its properties, ...
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                        jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            joints.append(info)

            # ... saves the IDs of the UR5 joints, ...
            if jointName in controlJoints:
                ur5_joints_id.append(jointID)
            
            # ... saves the IDs of the Gripper joints and ...
            if jointName in gripperJoints:
                gripper_joints_id.append(jointID)

            if jointName in gripperMimicJoints:
                gripper_mimic_joints_id.append(jointID)

            # saves the ID of the Palm joint
            if jointName in palmJoints:
                palm_joint_id.append(jointID)
    
    #print(joints)
    return ur5_joints_id, controllable

    


# Main
if __name__ == "__main__":
    
    


    client = p.connect(p.GUI)

    gui_joints = user_interface()
    
    ur5_id = spawn_environment(client)

    n_joints = p.getNumJoints(ur5_id)

    joints, __ = setup_robot(ur5_id)

    # mimic_finger_1 = gripperControl(ur5_id, joints, '1')
    # mimic_finger_2 = gripperControl(ur5_id, joints, '2')
    # mimic_finger_middle = gripperControl(ur5_id, joints, 'middle')

    # mimic_palm = palmControl(ur5_id, joints)

    palm = 0
    prev_palm = 0
    state_palm = 0

    while True:
        
        j = read_gui(gui_joints)
        palm = j[-1]

        if palm != prev_palm:
            state_palm = state_palm + 1

            if state_palm > 2:
                state_palm = 0

        if state_palm == 0:
            j[-1] = -0.19
        elif state_palm == 1:
            j[-1] = 0
        else:
            j[-1] = 0.17

        set_joints(ur5_id, j)
        prev_palm = palm

        q = []

        # UR5 joint values
        for i in joints:
            aux = p.getJointState(bodyUniqueId=ur5_id, 
                                jointIndex=i,
                                physicsClientId=client)

            q.append(aux[0])


        # render(client, ur5_id)
        T = ur5.fkine(q, order='zyx')

        print("Robotic Toolboox T: ", T.t)
        print("Robotic Toolbox R: ", T.rpy(order='zyx'))
        print("--")
        # print(T)

        state = p.getLinkState(bodyUniqueId = ur5_id, linkIndex = 11, computeForwardKinematics = True)
        print("Pybullet API T: ", state[0])
        print("Pybullet aPI R: ", p.getEulerFromQuaternion(state[1]))
        print("--\n\n")

        p.stepSimulation()