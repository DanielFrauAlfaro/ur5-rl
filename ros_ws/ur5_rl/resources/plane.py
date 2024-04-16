#!/usr/bin/env python3

import pybullet as p
import os


class Plane:
    def __init__(self, client):
        # Import land plane
        p.loadURDF(fileName=os.path.dirname(__file__) + '/models/urdf/simpleplane.urdf',
                              basePosition=[0, 0, 0], 
                              physicsClientId=client,
                              useFixedBase=1)
        
        # Import table
        '''p.loadURDF(fileName=os.path.dirname(__file__) + '/models/urdf/table.urdf',
                              basePosition=[1, 0, 0.3], 
                              physicsClientId=client,
                              useFixedBase=1)'''