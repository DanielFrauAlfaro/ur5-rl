#!/usr/bin/env python3

import pybullet as p
import os

# Class for importing objects according to a number
'''
Class for spawning the object in the simulation.
    - Parameters:
        - client (int): Pybullet client
        - object (int): number of the object to be spawned in the internal class spawnables
        - position (list/np.array, 1x3): cartesian position of the object
        - orientation (list/np.array, 1x3): orientation of the object in Euler angles

'''
class Object:
    def __init__(self, client, objects):
        
        # List of object directories
        self.spawnables = tuple(objects)

        self.client = client
        
        
    def spawn(self, object_chosen = 0, rand_position=[0, 1, 0.5], rand_orientation=[0, 0, 0, 1]):

        self.name = self.spawnables[object_chosen]

        # Load URDF model of selected object
        self.id = p.loadURDF(fileName=os.path.dirname(__file__) + '/models/pybullet_URDF_objects/' + self.spawnables[object_chosen] + '/model.urdf',
                                 basePosition = rand_position,
                                 baseOrientation = rand_orientation, 
                                 physicsClientId = self.client)
        
    