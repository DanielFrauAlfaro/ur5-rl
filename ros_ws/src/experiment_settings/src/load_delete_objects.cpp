#include <experiment_settings/Object.h>
#include <fstream>
#include <gazebo_msgs/DeleteModel.h>
#include <gazebo_msgs/DeleteModelRequest.h>
#include <gazebo_msgs/DeleteModelResponse.h>
#include <gazebo_msgs/SpawnModel.h>
#include <gazebo_msgs/SpawnModelRequest.h>
#include <gazebo_msgs/SpawnModelResponse.h>
#include <iostream>
#include <ros/ros.h>
#include "std_msgs/Int32.h"
#include <string>
#include <tf/tf.h>

experiment_settings::Object mensaje;
ros::Subscriber sub;

void readMessageCallback(const experiment_settings::Object::ConstPtr& msg){
  
  if (msg->chosenObject.size() > 0){
    mensaje = *msg;    
    sub.shutdown();
  }
}

void addCallback(const std_msgs::Int32::ConstPtr& msg){

}

void removeCallback(const std_msgs::Int32::ConstPtr& msg){

}


int main(int argc, char *argv[]){

  // This must be called before anything else ROS-related
  ros::init(argc, argv, "load_delete");

  // Create a ROS node handle
  ros::NodeHandle nh("~");
  ros::Rate loop_rate(10);

  // Get parameters from launch file
  std::string saveFilesPath;
  nh.getParam("save_files_path", saveFilesPath);
  
  // Read message (all of it)
  ros::Subscriber sub = nh.subscribe("/aurova/objects/data", 1, readMessageCallback);  
    
  int add_delete_object = 0;
  int actual_object = 0;
  
  while(mensaje.chosenObject.size() == 0){
    ros::spinOnce();
  }
  
  int number_objects = mensaje.chosenObject.size();
  
  while(ros::ok()){
  
    if (add_delete_object == 0){
      
      ros::Subscriber sub1 = nh.subscribe("/aurova/objects/add", 1, addCallback);
      std_msgs::Int32 addObject = *ros::topic::waitForMessage<std_msgs::Int32>("/aurova/objects/add", nh);
      ROS_INFO("RECEIVED ORDER: adding.");
      
      tf::Quaternion q;
      q.setRPY(mensaje.orientation[actual_object].x, mensaje.orientation[actual_object].y, mensaje.orientation[actual_object].z);
      
      geometry_msgs::Pose pose;
      pose.position.x = mensaje.pose[actual_object].x;
      pose.position.y = mensaje.pose[actual_object].y;
      pose.position.z = mensaje.pose[actual_object].z;
      pose.orientation.x = q.x();
      pose.orientation.y = q.y();
      pose.orientation.z = q.z();
      pose.orientation.w = q.w();
      
      gazebo_msgs::SpawnModel spawn;
      spawn.request.model_name = "object";
      
      std::ifstream ifs;
      std::ostringstream s;
      std::string completeFile;
      ifs.open(mensaje.chosenObject[actual_object]);
      s << ifs.rdbuf();

      completeFile = s.str();
      //std::cout<<completeFile<<std::endl;
      ifs.close();
      
      spawn.request.model_name="object";
      spawn.request.model_xml=completeFile;
      spawn.request.robot_namespace="experiment_settings";
      spawn.request.initial_pose=pose;
      spawn.request.reference_frame="world";      
      std::string topicGazebo = "/gazebo/spawn_sdf_model";
      
      // Save actual object to file
      std::fstream my_file;
      my_file.open(saveFilesPath+std::to_string(actual_object)+"_info.txt", std::ios::out);
      if (!my_file){
        ROS_INFO("ERROR: file was not created.");
      }
      else{
        my_file<<"RELEASED OBJECT IN: \n";
        my_file<<"\t Object: "<<mensaje.chosenObject[actual_object]<<"\n";
        my_file<<"\t Pose: "<<mensaje.pose[actual_object].x<<", "<<mensaje.pose[actual_object].y<<", "<<mensaje.pose[actual_object].z<<"\n";
        my_file<<"\t Orientation: \n";
        my_file<<"\t\t Quaternion ->"<<q.x()<<", "<<q.y()<<", "<<q.z()<<", "<<q.w()<<"\n";
        my_file<<"\t\t RPY ->"<<mensaje.orientation[actual_object].x<<", "<<mensaje.orientation[actual_object].y<<", "<<mensaje.orientation[actual_object].z<<"\n";
        my_file.close();
      }      
      
      ros::ServiceClient spawn_object = nh.serviceClient<gazebo_msgs::SpawnModel>(topicGazebo);
      spawn_object.waitForExistence();
      
       if (!spawn_object.call(spawn)) {
        ROS_ERROR("Failed to call service %s",topicGazebo.c_str());
       }
       
       ROS_INFO("Result: %s, code %u",spawn.response.status_message.c_str(), spawn.response.success);
      add_delete_object = 1;
      sub1.shutdown();
    }
    else{

      ros::Subscriber sub2 = nh.subscribe("/aurova/objects/remove", 1, removeCallback);
      std_msgs::Int32 removeObject =  *ros::topic::waitForMessage<std_msgs::Int32>("/aurova/objects/remove", nh);
      
      ROS_INFO("RECEIVED ORDER: removing.");
      
      gazebo_msgs::DeleteModel deleteModel;
      deleteModel.request.model_name = "object";
      std::string topicGazeboDelete = "/gazebo/delete_model";
      ros::ServiceClient delete_object = nh.serviceClient<gazebo_msgs::DeleteModel>(topicGazeboDelete);
      delete_object.waitForExistence();
      
      if (!delete_object.call(deleteModel)) {
        ROS_ERROR("Failed to call service %s",topicGazeboDelete.c_str());
       }
       
       ROS_INFO("Result: %s, code %u",deleteModel.response.status_message.c_str(), deleteModel.response.success);

      add_delete_object = 0;
      sub2.shutdown();

      if(removeObject.data == 1){
        actual_object = actual_object + 1;
      }
    }
    
    if (number_objects == actual_object){
      ros::shutdown();
    }
  }
    
  return 0;
}
