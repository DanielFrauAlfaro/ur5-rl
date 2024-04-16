#include <iostream>
#include <ros/ros.h>
#include <typeinfo>

#include "moveit_msgs/ExecuteTrajectoryActionGoal.h"
#include "sensor_msgs/JointState.h"
#include "std_msgs/String.h"
#include<unistd.h>

ros::Publisher pub;
sensor_msgs::JointState allegro_ref;

// Callback for Moveit! execution trajectory
void moveit_exec(const moveit_msgs::ExecuteTrajectoryActionGoal::ConstPtr& msg)
{
    // If the move_group selected is the Allegro hand ...
    if(std::find(std::begin(msg->goal.trajectory.joint_trajectory.joint_names), std::end(msg->goal.trajectory.joint_trajectory.joint_names), "joint_0") != std::end(msg->goal.trajectory.joint_trajectory.joint_names))
    {
        // Number of points and number of joints
        int n_points = msg->goal.trajectory.joint_trajectory.points.size();
        int n_joints = msg->goal.trajectory.joint_trajectory.joint_names.size();

        // Time interval
        unsigned int prev_time = 0.0;
        unsigned int t = 0.0;

        // Iterates points
        for(int i = 0; i < n_points; i++)
        {
            // Iterates joints (0 - 15 joints for Allegro)
            for(int j = 0; j < n_joints; j++)
            {
                // Gets the position of "_" character and the length of the name
                int found = msg->goal.trajectory.joint_trajectory.joint_names[j].find("_", 4);
                int len = msg->goal.trajectory.joint_trajectory.joint_names[j].length();
                
                // Converts the last one or two characters to integers ("joint_12" --> 12 (int))
                int n = std::stoi(msg->goal.trajectory.joint_trajectory.joint_names[j].substr(found+1, len));

                // Builds the message
                allegro_ref.position[n] = msg->goal.trajectory.joint_trajectory.points[i].positions[j];
                allegro_ref.velocity[n] = msg->goal.trajectory.joint_trajectory.points[i].velocities[j];
            }

            // Publish to control topic of the Allegro Hand
            pub.publish(allegro_ref);

            // Loop delay according the moveit planification
            t = msg->goal.trajectory.joint_trajectory.points[i].time_from_start.sec * 1000000 + msg->goal.trajectory.joint_trajectory.points[i].time_from_start.nsec * 0.001;
            usleep((t - prev_time));
            prev_time = t;
        }
    }
}

int main(int argc, char* argv[]){

    // This must be called before anything else ROS-related
    ros::init(argc, argv, "bridge");

    // Create a ROS node handle
    ros::NodeHandle nh("node_handler");
    ros::Rate loop_rate(10);

    // Subscribe to the execute trajectory from Moveit!
    ros::Subscriber sub_exec = nh.subscribe("/execute_trajectory/goal", 1, moveit_exec);
    
    // Advertise to the joint command topics of the Allegro
    pub = nh.advertise<sensor_msgs::JointState>("/allegroHand_0/joint_cmd", 1);
    ros::Publisher pub_lib_cmd = nh.advertise<std_msgs::String>("/allegroHand_0/lib_cmd", 1);

    // Builds message for the Allegro
    allegro_ref.name = {"joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "joint_8", "joint_9", "joint_10", "joint_11", "joint_12", "joint_13", "joint_14", "joint_15"};
    allegro_ref.position = {0.054257972111446115, -0.20809914819077, 0.795042366920901, 0.7356181756152973, 0.0406002603936231, -0.2176680902679441, 0.761706234737664, 0.839460142892501, 0.04786486085199019, -0.134700620722323415, 0.881007587233535, 0.8570546693443598, 0.9487283112871126, 0.414104128145736, 0.293728525560962, 0.75062584902150477};
    allegro_ref.velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Publishes home position to the "lib_cmd" topic for 5 seconds
    int it = 0;

    std_msgs::String home;
    home.data = "home";

    while(it < 5)
    {
        pub.publish(allegro_ref);
        //pub_lib_cmd.publish(home);
        ros::Duration(1).sleep(); 
        it++;
    }

    // Spin
    ros::spin();
    
    return 0;
}