#!/usr/bin/env python3

import pybullet as p
import os

# Class for importing objects according to a number
class Table:
    def __init__(self, client, position=[-0.16, 1.0, 0.74], orientation=[3.14, 0, 0, 1]):
        # Load URDF model of selected object: pybullet_URDF_objects/' + spawnables[object] + '/model.urdf',

        euler = [-1.57,0,-1.57]
        orientation = p.getQuaternionFromEuler(euler)

        self.id = p.loadURDF(fileName=os.path.dirname(__file__) + '/models/models/table.urdf',
                                 basePosition=position,
                                 baseOrientation=orientation, 
                                 physicsClientId=client,
                                 useFixedBase=1)