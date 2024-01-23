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

        # --- Parameters
        self.client = client    # Client ID of the Pybullet server
        

        # Internal UR5 DH model
        self.__ur5 = rtb.DHRobot([
            rtb.RevoluteDH(d=0.1625, alpha=pi/2.0),
            rtb.RevoluteDH(a=-0.425),
            rtb.RevoluteDH(a = -0.3922),
            rtb.RevoluteDH(d = 0.1333, alpha=pi/2.0),
            rtb.RevoluteDH(d = 0.0997, alpha=-pi/2.0),
            rtb.RevoluteDH(d = 0.0996)
        ], name="UR5e")
    

        # Load the UR5 URDF
        self.id = p.loadURDF(fileName=os.path.dirname(__file__) + '/models/models/ur5_env.urdf',
                              basePosition=[0, 0, 0], 
                              physicsClientId=client, 
                              useFixedBase=1)

        # Name of the gripper and UR5 joint (all joints are included on the same body)
        gripperJoints = ["robotiq_finger_1_joint_1", "robotiq_finger_2_joint_1", "robotiq_finger_middle_joint_1"]

        controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint"]
        
        # Get info from joints and enable joint torque sensor
        self.ur5_joints_id = []
        self.gripper_joints_id = []

        numJoints = p.getNumJoints(self.id)
        jointInfo = collections.namedtuple("jointInfo",["id","name","type",'damping','friction',"lowerLimit","upperLimit","maxForce","maxVelocity","controllable"])
        
        list_attr = {}

        # Iterates over each joint
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

    
            # If a joint is controllable ...
            if controllable:
                # print("Name: ", jointName, "Joint Index:", i, "Link Index:", info[12])
                # print("--")

                # ... enables torque sensors, ...
                p.enableJointForceTorqueSensor(bodyUniqueId=self.id, 
                                               jointIndex=jointID, 
                                               enableSensor=1,
                                               physicsClientId=self.client)

                # ... saves its properties, ...
                info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)

                # ... saves the IDs of the UR5 joints, ...
                if jointName in controlJoints:
                    self.ur5_joints_id.append(jointID)
                
                # ... saves the IDs of the Gripper joints and ...
                if jointName in gripperJoints:
                    self.gripper_joints_id.append(jointID)


        # Starting joint positions and velocities for the robot joints
        self.q = [-0.004, -1.549, -1.547, -pi/2.0, pi/2.0, pi/2.0]
        self.qd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Starting end effector's position
        self.ee = [0.1332997, 0.49190053, 0.48789219, -3.14, 0.0, -2.35658978]

        # Gripper parameters
        self.m1 = 1.2218 / 140
        self.max_closure = 100
        self.g = 1

        # Brings the robot and gripper to a starting position
        self.apply_action(self.q)
        self.apply_action_g(self.g)



    # Return the client and robot's IDs
    def get_ids(self):
        return self.client, self.id
    
    # Applies cartesian action
    def apply_action_c(self, action):
        '''
        Given a cartesian action, applies the new position to the robot,
    along with the gripper action.

            - action: array of XYZ - RPY - G position actions, where G is the gripper
        action [0, 255] (list of length 7 --> 6 + 1)
        '''
        
        # Converts the action to a
        action = self.ee + action

        # Obtains XYZ and RPY positions
        x = action[0]                                 
        y = action[1]
        z = action[2]
            
        roll = action[3]
        pitch = action[4]
        yaw = action[5]

        # Builds up the homogeneus transformation matrix for XYZ and RPY
        T = SE3(x, y, z)
        T_ = SE3.RPY(roll, pitch, yaw, order='zyx')

        # Computes the complete transformation
        self.T = T * T_

        # Computes inverse kinematics
        q = self.__ur5.ik_LM(self.T,q0 = self.q)

        # Applies the joint action (joint and gripper)
        self.apply_action(q[0])
        self.apply_action_g(int(action[-1]))


    # Moves the robot to a desired position
    def apply_action(self, action):
        '''
        Applies a joint action

            - action: a set of joint positions according to the ids.
        [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3] (list)
        '''

        p.setJointMotorControlArray(bodyUniqueId=self.id, 
                                    jointIndices=self.ur5_joints_id, 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=action,
                                    physicsClientId=self.client)
    
    # Appliy action to the gripper
    def apply_action_g(self, action):
        '''
        Applies a gripper action

            - action: gripper action [0, 255] (int)
        '''

        # Computes pseudo - inverse kinematics. Transforms "g" action to closure angles 
        action = min(self.m1 * action, self.max_closure)
        action = np.ones(3) * action

        p.setJointMotorControlArray(bodyUniqueId=self.id, 
                                    jointIndices=self.gripper_joints_id, 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=action,
                                    physicsClientId=self.client)


    # Returns observation of the robot state
    def get_observation(self):
        '''
        Gets the observation space of the robot

        Returns:
            - A list of observations [q, qd, gripper, ee]
        '''

        self.q = []
        self.qd = []
        q_t = []
        g = []

        # UR5 joint values
        for i in self.ur5_joints_id:
            aux = p.getJointState(bodyUniqueId=self.id, 
                                jointIndex=i,
                                physicsClientId=self.client)

            self.q.append(aux[0])
            self.qd.append(aux[1])
            q_t.append(aux[2][-1])


        # Gripper Joint Values
        for i in self.gripper_joints_id:
            aux = p.getJointState(bodyUniqueId=self.id, 
                                jointIndex=i,
                                physicsClientId=self.client)

            g.append(aux[0])
        
        # Obtains [0, 255] representation
        self.g = min(min(g) / self.m1, self.max_closure)

        # End effector position and orientation
        T = self.__ur5.fkine(self.q, order='zyx')
        ee_pos = T.t
        ee_or = T.eul('zyx')
        self.ee = [ee_pos[0], ee_pos[1], ee_pos[2], ee_or[0], ee_or[1], ee_or[2]]
        
        # Builds and returns the message
        observation = [self.q, self.qd, self.g,  self.ee]

        return observation
            
        


        
