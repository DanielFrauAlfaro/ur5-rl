#!/usr/bin/env python3

import rospy
import numpy as np
import actionlib
from ur5_rl.msg import ManageObjectResult, ManageObjectAction
from geometry_msgs.msg import Pose, Quaternion, Point
from gazebo_msgs.srv import SpawnModel, DeleteModel
from tf.transformations import quaternion_from_euler
import random



class MyActionServer:
    def __init__(self):
        self.server = actionlib.SimpleActionServer('/manage_object', ManageObjectAction, self.execute, False)
        self.object_name = "spawned_object"
        self.spawned = False
        self.spawner = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.deleter = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.server.start()
        self.spawnables_list = ("002_master_chef_can", "005_tomato_soup_can", "sugar_box", "cracker_box", 
                                "006_mustard_bottle", "017_orange", "cleanser", "conditioner", "magic_clean",
            
                                "repellent", "potato_chip_1", "021_bleach_cleanser", "pen_container_1", "orion_pie")
        self.spawned = False



    def execute(self, goal):
        # rospy.loginfo("Received goal: %d", goal.spawn)
        
        if goal.spawn:
            if not self.spawned:
            
                object_chosen = random.randint(0,9) * 0
                model_xml = "/daniel/Desktop/ur5-rl/ros_ws/src/ur5_rl/urdf/" + self.spawnables_list[object_chosen] + "/model.urdf"
                # print(f"--- Spawning model {object_chosen}")


                pose = Pose()

                pos, orn = np.random.uniform([[0.05, 0.5, 0.85], [-np.pi, -np.pi/2, -3.1415]], 
                                            [[0.275,  0.62, 0.85], [np.pi,  -np.pi/2,  3.1415]])

                point = Point(pos[0], pos[1], pos[2])

                quat_tf = quaternion_from_euler(orn[0], orn[1], orn[2])
                q_msg = Quaternion(quat_tf[0], quat_tf[1], quat_tf[2], quat_tf[3])
                
                pose.orientation = q_msg
                pose.position = point

                self.spawner(
                    model_name=self.object_name,
                    model_xml=open(model_xml, 'r').read(),
                    robot_namespace='ur5_rl',
                    initial_pose=pose,
                    reference_frame='world'
                )

                result = ManageObjectResult()
                result.res = True
                self.spawned = True
                self.server.set_succeeded(result)

            else:
                # print("--- Spawner Service: SPAWNING ERROR: There is an object in the simulation")

                result = ManageObjectResult()
                result.res = False
                self.server.set_succeeded(result)


        else:
            if self.spawned:
                self.deleter(model_name = self.object_name)
                
                result = ManageObjectResult()
                result.res = True
                self.spawned = False
                self.server.set_succeeded(result)

            else:
                # print("--- Spawner Service: DELETING ERROR: There is NOT any object in the simulation")

                result = ManageObjectResult()
                result.res = False
                self.server.set_succeeded(result)


if __name__ == '__main__':
    rospy.init_node('objects_server')
    server = MyActionServer()
    rospy.spin()
