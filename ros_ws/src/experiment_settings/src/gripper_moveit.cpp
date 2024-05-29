#include <iostream>
#include <ros/ros.h>
#include "std_msgs/Int32.h"
#include "std_msgs/Float64MultiArray.h"
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <thread>

ros::Publisher end_mov; 

void cb_gripper(const std_msgs::Float64MultiArray::ConstPtr msg)
{   
    std::string PLANNING_GROUP_GRIPPER = "gripper";
    moveit::planning_interface::MoveGroupInterface move_group_interface_gripper(PLANNING_GROUP_GRIPPER);

    std::cout<<"\n\n ------- Movement Request ------- \n\n";
    moveit::planning_interface::MoveGroupInterface::Plan my_plan_arm;
    
    std::vector<double> joint_values = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    int dim = msg->data.size();

    for(int i = 0; i<dim ; i++)
    {
        // 3F gripper
        if(dim == 3)
        {
            joint_values[i*dim] = msg->data[i];
            joint_values[i*dim + 2] = -0.61085;
        }

        // 4F gripper
        else
        {
            // TODO
            std::cout<<"\n\n\n  TODO: moveit con la allegro \n\n\n";
        }
    }

    std::cout<<"\n\n ------- Planning -------\n\n";
    move_group_interface_gripper.setJointValueTarget(joint_values);
    std::cout<<"\n\n\n ------- Plan arm -------\n";
    move_group_interface_gripper.plan(my_plan_arm);


    std::cout<<"\n\n ------- Executing -------\n\n";
    move_group_interface_gripper.execute(my_plan_arm);

    std_msgs::Int32 msg_end_mov;
    msg_end_mov.data = 1;

    std::cout<<"\n\n ------- Ending process ------- \n\n";
    for(int k = 0; k<2; k++)
    {
        ros::spinOnce();
        end_mov.publish(msg_end_mov);
    }
}

int main(int argc, char* argv[])
{
    
    ros::init(argc, argv, "gripper_3f_moveit");
    ros::NodeHandle nh("~");
    
    ros::Rate r(10);

    std::cout<<"AAAAAAAA\n\n\n\n\n";

    ros::Subscriber sub = nh.subscribe("/aurova/gripper_values", 1, cb_gripper);

    end_mov = nh.advertise<std_msgs::Int32>("/aurova/end_mov", 1);

    ros::spin();

    return 0;
}