#!/usr/bin/env python3

import rospy
import actionlib
from ur5_rl.msg import ManageObjectsFeedback, ManageObjectsResult, ManageObjectsGoal, ManageObjectsAction
from gazebo_msgs.srv import SpawnModel



class MyActionServer:
    def __init__(self):
        self.server = actionlib.SimpleActionServer('/manage_object', ManageObjectsAction, self.execute, False)
        self.server.start()

    def execute(self, goal):
        rospy.loginfo("Received goal: %d", goal.input)
        
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        model_xml = "/home/daniel/Desktop/ur5-rl/ros_ws/src/ur5_rl/urdf/002_master_chef_can/model.urdf"
        model_name = "spawned_object"
        pose = 0, 0, 0, 0, 0, 0
        spawn_model(model_name, model_xml, "object", pose, "world")
        result = ManageObjectsResult(success=True)
        self.server.set_succeeded(result)

if __name__ == '__main__':
    rospy.init_node('objects_server')
    server = MyActionServer()
    rospy.spin()
