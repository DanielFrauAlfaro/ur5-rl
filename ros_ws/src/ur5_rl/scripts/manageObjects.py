#!/usr/bin/env python3

import rospy
import numpy as np
import actionlib
from ur5_rl.msg import ManageObjectResult, ManageObjectAction
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnModel



class MyActionServer:
    def __init__(self):
        self.server = actionlib.SimpleActionServer('/manage_object', ManageObjectAction, self.execute, False)
        self.server.start()

    def execute(self, goal):
        rospy.loginfo("Received goal: %d", goal.input)
        
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        model_xml = "/daniel/Desktop/ur5-rl/ros_ws/src/ur5_rl/urdf/002_master_chef_can/model.urdf"
        model_name = "spawned_object"
        pose = Pose()
        

        pose.position.x = 0.1
        pose.position.y = 0.55
        pose.position.z = 1
        pose.orientation.x = -0.17641110570880375
        pose.orientation.y =-0.6847474876059038
        pose.orientation.z = -0.17641110570880367
        pose.orientation.w = 0.6847474876059036

        spawn_model(
            model_name=model_name,
            model_xml=open(model_xml, 'r').read(),
            robot_namespace='ur5_rl',
            initial_pose=pose,
            reference_frame='world'
        )

        result = ManageObjectResult()
        result.res = True
        self.server.set_succeeded(result)

if __name__ == '__main__':
    rospy.init_node('objects_server')
    server = MyActionServer()
    rospy.spin()
