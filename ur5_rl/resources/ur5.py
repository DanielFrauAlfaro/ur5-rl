#!/usr/bin/env python3

import pybullet as p
import collections
import os
from spatialmath import SE3
import roboticstoolbox as rtb
from math import pi
import numpy as np

# Class for the UR5
class UR5e:
    def __init__(self, client):
        
        # Internal UR5 DH model
        self.__ur5 = rtb.DHRobot([
            rtb.RevoluteDH(d=0.1625, alpha=pi/2.0),
            rtb.RevoluteDH(a=-0.425),
            rtb.RevoluteDH(a = -0.3922),
            rtb.RevoluteDH(d = 0.1333, alpha=pi/2.0),
            rtb.RevoluteDH(d = 0.0997, alpha=-pi/2.0),
            rtb.RevoluteDH(d = 0.0996)
        ], name="UR5e")
    
        self.__ur5.base = SE3.RPY(0,0,-pi)      # Rotate robot base so it matches Gazebo model
        self.__ur5.tool = SE3(0.0, 0.0, 0.03)

        # Client ID of the Pybullet server
        self.client = client

        # Load the UR5 URDF
        self.id = p.loadURDF(fileName=os.path.dirname(__file__) + '/models/models/ur5_env.urdf',
                              basePosition=[0, 0, 0], 
                              physicsClientId=client, 
                              useFixedBase=1)

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
        self.joints = []
        self.ur5_joints_id = []
        self.gripper_joints_id = []
        self.gripper_mimic_joints_id = []
        self.palm_joint_id = []
        numJoints = p.getNumJoints(self.id)
        jointInfo = collections.namedtuple("jointInfo",["id","name","type",'damping','friction',"lowerLimit","upperLimit","maxForce","maxVelocity","controllable"])
        
        list_attr = {}

        # Iterates for each joint
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
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

            # print("Name: ", jointName, "Joint Index:", i, "Link Index:", info[12])
            # print("--")


            # If a joint is controllable ...
            if controllable:
                # ... enables torque sensors, ...
                p.enableJointForceTorqueSensor(bodyUniqueId=self.id, 
                                               jointIndex=jointID, 
                                               enableSensor=1,
                                               physicsClientId=self.client)

                # ... saves its properties, ...
                info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
                self.joints.append(info)

                # ... saves the IDs of the UR5 joints, ...
                if jointName in controlJoints:
                    self.ur5_joints_id.append(jointID)
                
                # ... saves the IDs of the Gripper joints and ...
                if jointName in gripperJoints:
                    self.gripper_joints_id.append(jointID)

                if jointName in gripperMimicJoints:
                    self.gripper_mimic_joints_id.append(jointID)

                # saves the ID of the Palm joint
                if jointName in palmJoints:
                    self.palm_joint_id.append(jointID)



        # Starting joint positions for the robot and the gripper
        self.q = [0.0, -1.5708, -1.5708, -1.5708, 1.5708, -0.785 + pi]

        # Brings the robot to a starting position
        self.apply_action(self.q)



    # Return the client and robot's IDs
    def get_ids(self):
        return self.client, self.id
    
    def apply_action_c(self, action):
        x = action[0]                                 
        y = action[1]
        z = action[2]
            
        roll = action[3]
        pitch = action[4]
        yaw = action[5]

        # Builds up homogeneus matrix
        T = SE3(x, y, z)
        T_ = SE3.RPY(roll, pitch, yaw, order='xyz')

        self.T = T * T_

        # Computes inverse kinematics
        q = self.__ur5.ikine_LM(self.T,q0 = self.q)       # Inversa: obtiene las posiciones articulares a través de la posición


        # Applies the joint action
        self.apply_action(q.q)

    # Moves the robot to a desired position
    def apply_action(self, action):
        # Decodes the action in robot joint position, gripper position and palm state
        q = action

        # Assigns the action to the internal values of the robot
        # self.q = q

        # UR5 control
        p.setJointMotorControlArray(bodyUniqueId=self.id, 
                                    jointIndices=self.ur5_joints_id, 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=q,
                                    physicsClientId=self.client)


    
    # Returns observation of the robot state
    def get_observation(self):
        self.q = []
        qd = []
        q_t = []


        # UR5 joint values
        for i in self.ur5_joints_id:
            aux = p.getJointState(bodyUniqueId=self.id, 
                                jointIndex=i,
                                physicsClientId=self.client)

            self.q.append(aux[0])
            qd.append(aux[1])
            q_t.append(aux[2][-1])

        # End effector position and orientation
        T = self.__ur5.fkine(self.q, order='xyz')
        ee_pos = T.t
        ee_or = T.eul('xyz')

        ee = np.array([ee_pos[0], ee_pos[1], ee_pos[2], ee_or[0], ee_or[1], ee_or[2]])
        
        
        # Builds and returns the message
        observation = [self.q, qd, q_t,  ee]

        return observation
            
        


        
