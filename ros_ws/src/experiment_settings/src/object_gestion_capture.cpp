#include <cv_bridge/cv_bridge.h>
#include <gazebo_msgs/GetModelState.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Int32.h"
#include <tf/tf.h>

#include <thread>

ros::Subscriber sub, sub1, sub2, sub3, sub4;
bool endMoveit, endGrasping, capturedRGB, capturedD, capturedPC;
sensor_msgs::Image::ConstPtr rgbImg, depthImg;
sensor_msgs::PointCloud2::ConstPtr pcImg;

void moveitReadyCallback(const std_msgs::Int32::ConstPtr& msg){

  if (msg->data == 1){
    sub.shutdown();
    ros::Duration(4.0).sleep(); 
    endMoveit = true;  
  }
}

void captureRGBCallback(const sensor_msgs::Image::ConstPtr& msg){

  if (msg->height != 0){
    rgbImg = msg;
    sub1.shutdown();
    capturedRGB = true; 
    //std::cout<<"RGB_: "<<rgbImg->height<<std::endl; 
  }
}

void captureDCallback(const sensor_msgs::Image::ConstPtr& msg){

  if (msg->height != 0){
    depthImg = msg;
    sub2.shutdown();
    capturedD = true;  
    //std::cout<<"D_: "<<depthImg->height<<std::endl; 
  }
}

void capturePCCallback(const sensor_msgs::PointCloud2ConstPtr& msg){

  if (msg->height != 0){
    pcImg = msg;
    sub3.shutdown();
    capturedPC = true;  
    //std::cout<<"PC_: "<<pcImg->height<<std::endl; 
  }
}

void moveitGraspFinishedCallback(const std_msgs::Int32::ConstPtr& msg){

  if (msg->data == 1){
    sub4.shutdown();
    endGrasping = true;  
  }
}

int main(int argc, char *argv[]){

  // This must be called before anything else ROS-related
  ros::init(argc, argv, "gestion_capture_analysis");

  // Create a ROS node handle
  ros::NodeHandle nh("~");
  ros::Rate loop_rate(10);

  // Get parameters from launch file
  std::string saveFilesPath;
  std::string mode;

  nh.getParam("save_files_path", saveFilesPath);
  nh.getParam("mode", mode);

  int currentAction = 0;
  bool exit = false;
  int actual_object = 0;
  
  sub = nh.subscribe("/aurova/moveit_ready", 1, moveitReadyCallback);
  endMoveit = false;
  while (endMoveit == false){
    ros::spinOnce();  
  }
  
  while(ros::ok()){
  
    // First step: ask user to add or delete an object
    do{
      ROS_INFO(" ");
      ROS_INFO("Program of spawning objects in gazebo.");
      ROS_INFO("For loading a random object press \"l\"");
      ROS_INFO("For deleting the object press \"r\".");
      ROS_INFO("For keeping the same objec press \"c\".");
      ROS_INFO(" ");
    
      char option;
      std::cin>>option;
      exit = false;
    
      if ((option=='l') and (currentAction==0)){
        exit = true;
        currentAction = 1;
        
        //std::cout<<"OPTION L. Exit: "<<exit<<" & currentAction: "<<currentAction<<std::endl;
        ROS_INFO("Loading object number %d", actual_object);
        ros::Publisher add_object_pub = nh.advertise<std_msgs::Int32>("/aurova/objects/add", 1);
        std_msgs::Int32 msg;
        msg.data = 1;
        
        int loop = 0;
        while(loop < 2){      
          ros::spinOnce();
          add_object_pub.publish(msg);
          ros::Duration(0.5).sleep(); 
          loop = loop + 1;
        }
      }
      else if((option=='r') and (currentAction==1)){
        exit = true;
        currentAction = 0;
        
        //std::cout<<"OPTION R. Exit: "<<exit<<" & currentAction: "<<currentAction<<std::endl;
        ROS_INFO("Removing object number %d", actual_object);
        ros::Publisher delete_object_pub = nh.advertise<std_msgs::Int32>("/aurova/objects/remove", 1);
        std_msgs::Int32 msg;
        msg.data = 1;
        
        int loop = 0;
        while(loop < 2){      
          ros::spinOnce();
          delete_object_pub.publish(msg);
          ros::Duration(0.5).sleep(); 
          loop = loop + 1;
        }
        actual_object = actual_object+1;
      }
      else if((option=='c') and (currentAction==1)){
        exit=true;
        currentAction = 1;
        ROS_INFO("Using the same object as previous experiment");
      }
      else{
        ROS_INFO("Option not valid. Try again!");
        //std::cout<<"OPTION R. Exit: "<<exit<<std::endl;
      }
    }while(exit==false);
    
    //std::cout<<"Out of loop 1"<<std::endl;
    
    bool takePicture = false;
    // Second step: ask user if he is happy with the spawned object. If so, take picture
    if (currentAction == 1){
      do{
        ROS_INFO(" ");
        ROS_INFO("Object has been loaded. Can it be considered for running an experiment?");
        ROS_INFO("If so, press \"y\"");
        ROS_INFO("If not, press \"n\".");
        ROS_INFO("Press \"r\" for reset");
        ROS_INFO(" ");

    
        char option;
        std::cin>>option;
        exit = false;
    
        if (option=='y'){
          exit = true;
          takePicture = true;
          
          //std::cout<<"OPTION Y. Exit: "<<exit<<" & takePicture: "<<takePicture<<std::endl;
          ROS_INFO("Taking picture to object number %d", actual_object);
          
          // Get object pose and save it to file
          if(mode == "simulation")
          {
            std::string getModelStateName = "/gazebo/get_model_state";
            ros::ServiceClient getModelState = nh.serviceClient<gazebo_msgs::GetModelState>(getModelStateName);
            gazebo_msgs::GetModelState objstate;
            objstate.request.model_name = "object";
            objstate.request.relative_entity_name = "world";
            getModelState.waitForExistence();
        
            if (!getModelState.call(objstate)) {
              ROS_ERROR("Failed to call service %s",getModelStateName.c_str());
            }
        
            ROS_INFO("Result: %s, code %u",objstate.response.status_message.c_str(), objstate.response.success);
            
            std::fstream my_file;
            my_file.open(saveFilesPath+std::to_string(actual_object)+"_info.txt", std::ios::app);
            if (!my_file){
              ROS_INFO("ERROR: file was not created.");
            }
            else{
              my_file<<"CURRENT OBJECT IN: \n";
              my_file<<"\t Pose: "<<objstate.response.pose.position.x<<", "<<objstate.response.pose.position.y<<", "<<objstate.response.pose.position.z<<"\n";
              my_file<<"\t Orientation: \n";
              my_file<<"\t\t Quaternion ->"<<objstate.response.pose.orientation.x<<", "<<objstate.response.pose.orientation.y<<", "<<objstate.response.pose.orientation.z<<", "<<objstate.response.pose.orientation.w<<"\n";
              
              tf::Quaternion q(objstate.response.pose.orientation.x,
                              objstate.response.pose.orientation.y,
                              objstate.response.pose.orientation.z,
                              objstate.response.pose.orientation.w);
              tf::Matrix3x3 m(q);
              double roll, pitch, yaw;
              m.getRPY(roll, pitch, yaw);
                  
              my_file<<"\t\t RPY ->"<<roll<<", "<<pitch<<", "<<yaw<<"\n";
              my_file.close();
            }  
          }
          

          std::string name_color_raw = "/camera/color/image_raw";
          std::string name_depth_raw = "/camera/depth/image_raw";
          std::string name_depth_points = "/camera/depth/color/points";

          // TODO: poner aqui los nombres de los topics de la camara real
          if(mode == "real")
          {
            name_color_raw = "/camera/color/image_raw";
            name_depth_raw = "/camera/depth/image_rect_raw";
            name_depth_points = "/camera/depth/color/points";
          }
          

          sub1 = nh.subscribe(name_color_raw, 1, captureRGBCallback);
          sub2 = nh.subscribe(name_depth_raw, 1, captureDCallback);
          sub3 = nh.subscribe(name_depth_points, 1, capturePCCallback);
          capturedRGB = false, capturedD = false, capturedPC = false;
          while ((capturedRGB != true) or (capturedD != true) or (capturedPC != true)){
            ros::spinOnce();  
          }
          
          ROS_INFO("Captured images of object number %d. Please, remove the object.", actual_object);
          
          cv::Mat frameRGB, frameD;
          //std::cout<<rgbImg->height<<std::endl;
          //std::cout<<depthImg->height<<std::endl;
          frameRGB = cv_bridge::toCvShare(rgbImg, "bgr8")->image;      
          frameD = cv_bridge::toCvShare(depthImg, "16UC1")->image;
          
          //cv::Size s1 = frameRGB.size();
          //cv::Size s2 = frameD.size();
          
          //std::cout<<"Image1: "<<s1.height<<"x"<<s1.width<<std::endl;
          //std::cout<<"Image2: "<<s2.height<<"x"<<s2.width<<std::endl;
          
          cv::imwrite(saveFilesPath+std::to_string(actual_object)+"_color.png", frameRGB);
          cv::imwrite(saveFilesPath+std::to_string(actual_object)+"_depth.png", frameD);
          
          
          pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::fromROSMsg(*pcImg, *pcl_cloud);
          
          //std::cout <<"PC: "<<pcl_cloud->size()<<std::endl;
          pcl::io::savePCDFileASCII (saveFilesPath+std::to_string(actual_object)+"_PC.pcd", *pcl_cloud);
          
          ROS_INFO("Captured images saved to %s", saveFilesPath.c_str());     
          
          // Extra code for controlling when everything is ready and pass RGBD image
          ros::Publisher publishRGBD = nh.advertise<sensor_msgs::PointCloud2>("/aurova/image/RGBD", 1);
    
          int loop = 0;
          while(loop < 2){      
            ros::spinOnce();
            publishRGBD.publish(*pcImg);
            ros::Duration(0.5).sleep(); 
            loop = loop + 1;
          } 
          
          // Topic que indique que se acaba el proceso de grasping y que lo bloquee todo en caso contrario
          sub4 = nh.subscribe("/aurova/moveit_grasp_finished", 1, moveitGraspFinishedCallback);
          endGrasping = false;
          while (endGrasping == false){
            ros::spinOnce();  
          }
                    
        }
        else if(option=='n'){
          exit = true;
          takePicture = false;
        
          //std::cout<<"OPTION N. Exit: "<<exit<<" & takePicture: "<<takePicture<<std::endl;
          ROS_INFO("Rejected to take images of object number %d. Please remove the object.", actual_object);
          
        }
        
        else{
          ROS_INFO("Option not valid. Try again!");
          //std::cout<<"OPTION R2. Exit: "<<exit<<std::endl;
        }
      }while(exit==false);
    }    
    
    endGrasping = false;
    //std::cout<<"Out of loop 2"<<std::endl;
    ros::spinOnce();
  }
      
  return 0;
}
