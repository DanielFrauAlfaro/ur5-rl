#!/usr/bin/env python3

import pybullet as p
import os

# Class for importing objects according to a number
class Object:
    def __init__(self, client, object=0, position=[0, 1, 0.5], orientation=[0, 0, 0, 1]):
        # List of object directories
        spawnables = ("002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "011_banana", "012_strawberry", "013_apple", "014_lemon", "015_peach", "016_pear", "017_orange", "018_plum", "021_bleach_cleanser")

        # Load URDF model of selected object
        p.loadURDF(fileName=os.path.dirname(__file__) + '/models/pybullet_URDF_objects/' + spawnables[object] + '/model.urdf',
                              basePosition=position,
                              baseOrientation=orientation, 
                              physicsClientId=client)