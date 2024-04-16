#include <iostream>
#include <ros/ros.h>
#include <typeinfo>

#include "moveit_msgs/ExecuteTrajectoryActionGoal.h"
#include "sensor_msgs/JointState.h"
#include "std_msgs/String.h"
#include<unistd.h>

ros::Publisher pub;
sensor_msgs::JointState joint_states;
unsigned int sec, ini;
double nsec;

void retrieve_msg_data(const sensor_msgs::JointState::ConstPtr& msg, int n_joints)
{
    //int n_joints = msg->name.size();

    for(int i = 0; i < n_joints; i++)
    {
        auto it = std::find(std::begin(joint_states.name), std::end(joint_states.name), msg->name[i]);
        auto index = it - joint_states.name.begin();

        joint_states.position[index] = msg->position[i];
        //joint_states.velocity[index] = msg->velocity[i];
        //joint_states.effort[index] = msg->effort[i];
    }
}

void ur_states_cb(const sensor_msgs::JointState::ConstPtr& msg)
{
    retrieve_msg_data(msg, 6);
}

void allegro_states_cb(const sensor_msgs::JointState::ConstPtr& msg)
{
    retrieve_msg_data(msg, 16);
}

void timer_cb(const ros::TimerEvent& event)
{
    float whole;
    joint_states.header.stamp = ros::Time::now();

    pub.publish(joint_states);
}

int main(int argc, char* argv[])
{
    // This must be called before anything else ROS-related
    ros::init(argc, argv, "my_joint_states");

    // Create a ROS node handle
    ros::NodeHandle nh("node_handler");
    ros::Rate loop_rate(10);

    joint_states.name = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "joint_8", "joint_9", "joint_10", "joint_11", "joint_12", "joint_13", "joint_14", "joint_15"};
    joint_states.position = {-1.5708, -1.5708, -1.5708, -1.5708, 1.5708, -0.78537, 0.035257972111446115, -0.1566409914819077, 0.815042366920901, 0.8346181756152973, -0.04506002603936231, -0.1876680902679441, 0.821706234737664, 0.8259460142892501, -0.08786486085199019, -0.054700620722323415, 0.9091007587233535, 0.8320546693443598, 0.8487283112871126, 0.4074104128145736, 0.253728525560962, 0.8062584902150477};
    joint_states.velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    joint_states.effort = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};


    std::string states_topic;
    nh.getParam("states_topic", states_topic);

    ini = ros::Time::now().toSec();

    // Subscriber to UR5 joint states topic
    ros::Subscriber ur_joint_states = nh.subscribe("/ur5e/joint_states", 1, ur_states_cb);
    
    // Subscriber to Allegro Hand joint states topic
    ros::Subscriber allegro_joint_states = nh.subscribe("/allegroHand_0/joint_states", 1, allegro_states_cb);

    // Publisher of "/joint_states" topic
    pub = nh.advertise<sensor_msgs::JointState>("/joint_states", 1);

    // Timer callback
    ros::Timer timer = nh.createTimer(ros::Duration(0.1), timer_cb);

    ros::spin();

    /*
        Obtener el parámetro del nombre del topic_states del UR5        } --> Establecer el callback
        Obtener el parámetro del nombre del topic_states de la Allegro  } --> Establecer el callback

        Almacenar los valores de los callbacks
        Hacer un callback por timer y publicar los valores desde ahí en el topic combinado /joint_states
    */


    return 0;
}