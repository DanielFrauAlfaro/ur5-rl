#!/usr/bin/env python3

import rospy

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from tf.transformations import euler_from_quaternion, quaternion_from_euler

if __name__ == "__main__":
    rospy.init_node("test")
    robot = moveit_commander.RobotCommander()
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    display_trajectory_publisher = rospy.Publisher("/move_group/display_planned_path",moveit_msgs.msg.DisplayTrajectory,queue_size=20,)

    pose_goal = move_group.get_current_pose().pose

    
    print(pose_goal)




    pose_goal.position.x = -0.604859
    pose_goal.position.y = 0.216956
    pose_goal.position.z = 0.127153 + 0.22
    





    pose_goal.orientation.x = -0.661474
    pose_goal.orientation.y = 0.262644
    pose_goal.orientation.z = 0.682835
    pose_goal.orientation.w = 0.164936

    (r, p, y) = euler_from_quaternion([ pose_goal.orientation.x,  pose_goal.orientation.y,  pose_goal.orientation.z , pose_goal.orientation.w])
    print(r,p,y)
    r += pi+ pi/2
    aux_p = p
    p = -y
    y = aux_p
    (x, y_, z, w) = quaternion_from_euler(r,p,y)
    print(x, y_, z, w)
    pose_goal.orientation.x = x
    pose_goal.orientation.y = y_
    pose_goal.orientation.z = z
    pose_goal.orientation.w = w

    move_group.set_planning_time(10.0)

    move_group.set_pose_target(pose_goal)
    plan = move_group.plan()

    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    # Publish
    for i in range(0, 10):
        display_trajectory_publisher.publish(display_trajectory)

    # opcion = input("\n\nTrayectoria correcta? \nSeleccione 's' para aceptar. ")

    # if opcion == "s":
    #     success = move_group.go(wait=True)
        
    #     # Calling `stop()` ensures that there is no residual movement
    #     move_group.stop()
    
    # # It is always good to clear your targets after planning with poses.
    # # Note: there is no equivalent function for clear_joint_value_targets().
    move_group.clear_pose_targets()

    '''
Sin trans
0.216956
0.604859
0.127153


-0.661474
0.262644
0.682835
0.164936

    '''
'''
Con trans
0.216956
0.604859
0.127153


-0.51061
0.495784
0.567739
0.413691


'''
