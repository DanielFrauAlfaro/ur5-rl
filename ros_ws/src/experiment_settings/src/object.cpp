#include <cstdlib>
#include <ctime>
#include <experiment_settings/Object.h>
#include <filesystem>
#include <geometry_msgs/Point.h>
#include <iostream>
#include <math.h>
#include <ros/ros.h>
#include <string>
#include <typeinfo>

int main(int argc, char *argv[]){

  // This must be called before anything else ROS-related
  ros::init(argc, argv, "object_pose");

  // Create a ROS node handle
  ros::NodeHandle nh("~");

  // Get parameters from launch file
  int numObjects, setSeed;
  std::string objectsDir;
  nh.getParam("num_objects", numObjects);
  nh.getParam("objects_dir", objectsDir);
  nh.getParam("set_seed", setSeed);
  
  // Set seed
  std::cout<<"\n\n\n\n Set seed: "<<setSeed<<"\n\n\n\n"<<std::endl;
  if (setSeed == 1){
    srand(0);
    ROS_INFO("Seed has been set.");
  }
  else{
    srand(time(nullptr));
  }
  
  // Advertise topic
  ros::Publisher publisher = nh.advertise<experiment_settings::Object>("/aurova/objects/data", 1);
  ros::Rate loop_rate(10);
  
  // Get objects to work
  std::vector<std::string> objects, chosenObjects;

  for(const std::filesystem::__cxx11::directory_entry& file: std::filesystem::directory_iterator(objectsDir)){
    objects.push_back(file.path().string());   
  }
  
  for(int i=0; i<numObjects; i++){
    int random = rand() % objects.size();
    chosenObjects.push_back(objects[random]+"/model.sdf");
    std::cout<<objects[random]+"/model.sdf"<<std::endl;
  }


  // float x_min = 0.30;
  // float x_max = 0.40;
  // float y_min = 0.35;
  // float y_max = 0.60;
  // float z_min = 0.95;
  // float z_max = 0.95;

  // Get positions  
  float x_min = 0.26;
  float x_max = 0.22;
  float y_min = 0.52;
  float y_max = 0.58;
  float z_min = 0.8;
  float z_max = 0.8;
  std::vector<float> x_coord, y_coord, z_coord;

  for(int i=0; i<numObjects; i++){
    float random = ((float)rand() /RAND_MAX) * (x_max-x_min) + x_min;
    x_coord.push_back(random);
    random = ((float)rand() /RAND_MAX) * (y_max-y_min) + y_min;
    y_coord.push_back(random);
    random = ((float)rand() /RAND_MAX) * (z_max-z_min) + z_min;
    z_coord.push_back(random);
  }
  
  // Get orientations
  float roll_min = 0;
  float roll_max = 2*M_PI;
  float pitch_min = 0;
  float pitch_max = 2*M_PI;
  float yaw_min = 0;
  float yaw_max = 2*M_PI;
  std::vector<float> roll_coord, pitch_coord, yaw_coord;
  float min=5.0, max=-5.0;
  
  for(int i=0; i<numObjects; i++){
    float random = ((float)rand() /RAND_MAX) * (roll_max-roll_min) + roll_min;
    roll_coord.push_back(random);
    random = ((float)rand() /RAND_MAX) * (pitch_max-pitch_min) + pitch_min;
    pitch_coord.push_back(random);
    random = ((float)rand() /RAND_MAX) * (yaw_max-yaw_min) + yaw_min;
    yaw_coord.push_back(random);
  }
  
  // Fill message with data
  experiment_settings::Object customMessage;
  
  customMessage.header.stamp = ros::Time::now();
  customMessage.header.frame_id = "object_pose";
  
  for(int i=0; i<numObjects; i++){
    customMessage.chosenObject.push_back(chosenObjects[i]);
    
    geometry_msgs::Point point;
    point.x = x_coord[i];
    point.y = y_coord[i];
    point.z = z_coord[i];   
    customMessage.pose.push_back(point);
    
    point.x = roll_coord[i];
    point.y = pitch_coord[i];
    point.z = yaw_coord[i];    
    customMessage.orientation.push_back(point);
  }

  while(ros::ok()){
    publisher.publish(customMessage);
    ros::Duration(0.5).sleep();  
  }
  
  return 0;
}
