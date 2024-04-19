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
        self.object_name = "spawned_object"
        self.spawned = False
        self.spawner = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.deleter = rospy.ServiceProxy('/gazebo/delete_model', SpawnModel)
        self.server.start()

    def execute(self, goal):
        rospy.loginfo("Received goal: %d", goal.input)
        
        if goal.input == 1:
            
            model_xml = "/daniel/Desktop/ur5-rl/ros_ws/src/ur5_rl/urdf/002_master_chef_can/model.urdf"
            
            pose = Pose()
            
            pose.position.x = 0.1
            pose.position.y = 0.55
            pose.position.z = 1
            pose.orientation.x = -0.17641110570880375
            pose.orientation.y =-0.6847474876059038
            pose.orientation.z = -0.17641110570880367
            pose.orientation.w = 0.6847474876059036

            self.spawner(
                model_name=self.object_name,
                model_xml=open(model_xml, 'r').read(),
                robot_namespace='ur5_rl',
                initial_pose=pose,
                reference_frame='world'
            )

            result = ManageObjectResult()
            result.res = True
            self.server.set_succeeded(result)

        else:
            self.deleter(model_name = self.object_name)
            
            result = ManageObjectResult()
            result.res = True
            self.server.set_succeeded(result)


if __name__ == '__main__':
    rospy.init_node('objects_server')
    server = MyActionServer()
    rospy.spin()
