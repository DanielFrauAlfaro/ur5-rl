#!/usr/bin/env python3

import pybullet as p
import os

# Class for importing objects according to a number
class Object:
    def __init__(self, client, object=0, position=[0, 1, 0.5], orientation=[0, 0, 0, 1]):

        
        # List of object directories
        spawnables = ("002_master_chef_can", "005_tomato_soup_can", 
                      "sugar_box", "cracker_box", 
                      "006_mustard_bottle", 
                      "017_orange", "cleanser", 
                      "conditioner", "magic_clean",                  "repellent", "potato_chip_1", "021_bleach_cleanser", "pen_container_1", "orion_pie")
        
        self.name = spawnables[object]

        # Load URDF model of selected object: pybullet_URDF_objects/' + spawnables[object] + '/model.urdf',
        self.id = p.loadURDF(fileName=os.path.dirname(__file__) + '/models/pybullet_URDF_objects/' + spawnables[object] + '/model.urdf',
                                 basePosition=position,
                                 baseOrientation=orientation, 
                                 physicsClientId=client)
        

    