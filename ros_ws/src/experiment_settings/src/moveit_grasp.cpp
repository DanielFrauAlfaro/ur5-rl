#include <cmath>
#include "experiment_settings/Grasp.h"
#include "experiment_settings/GraspEvoContacts.h"
#include "experiment_settings/GraspEvoPose.h"
#include <filesystem>
#include <gazebo_msgs/ContactsState.h>
#include <geograspevo/GeoGraspEvo.h>
#include <iostream>
#include "message_filters/subscriber.h"
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <ros/ros.h>
#include "std_msgs/Int32.h"
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include "visualization_msgs/Marker.h"

#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <experiment_settings/IKAction.h>


/*#include <chrono>*/
#include <thread>
/*#include <unistd.h> */

ros::Subscriber sub;
bool graspResult, transformReceived, contact, contact1_b, contact2_b, contact3_b, contact1_prev, contact2_prev, contact3_prev;
experiment_settings::Grasp::ConstPtr graspResultMsg;

void graspResultCallback(const experiment_settings::Grasp::ConstPtr& msg){
  
  if (msg->bestGrasp.graspContactPoints.height != 0){
    graspResultMsg = msg;
    sub.shutdown();
    graspResult = true; 
  }
}

// Get current pose of joints in cartesian and articular space
void getCurrentPose(moveit::planning_interface::MoveGroupInterface *interface, geometry_msgs::PoseStamped *current_pose, std::vector<double> *joint_values, std::vector<std::string> *joint_names){

  *current_pose = interface->getCurrentPose();
  *joint_values = interface->getCurrentJointValues();
  *joint_names = interface->getJoints();
}

void translateMessage(GraspEvoContacts *contacts, float *ranking, GraspEvoPose *pose, Eigen::Matrix4f *transformation){

  // Ranking
  *ranking = graspResultMsg->ranking;
  
  // Grasp points
  experiment_settings::GraspEvoContacts msgContacts;
  msgContacts = graspResultMsg->bestGrasp;
  pcl::PointCloud<pcl::PointXYZ>::Ptr graspPoints(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(msgContacts.graspContactPoints, *graspPoints);
  std::vector <pcl::PointXYZ> translatedPoints;
  for(int i=0; i<graspPoints->points.size(); i++){
    translatedPoints.push_back(graspPoints->points[i]);
  }
  (*contacts).graspContactPoints = translatedPoints;
  
  // Middle point and pose
  experiment_settings::GraspEvoPose msgPose;
  msgPose = graspResultMsg->bestPose;
  std::vector <Eigen::Vector3f> translatedPosePoints;
  for(int i=0; i<msgPose.graspPosePoints.size(); i++){
    Eigen::Vector3f point;
    point[0] = msgPose.graspPosePoints[i].x;
    point[1] = msgPose.graspPosePoints[i].y;
    point[2] = msgPose.graspPosePoints[i].z;    
    translatedPosePoints.push_back(point);
  }  
  (*pose).graspPosePoints = translatedPosePoints;
  
  tf::Quaternion qtf(msgPose.midPointPose.orientation.x, msgPose.midPointPose.orientation.y, msgPose.midPointPose.orientation.z, msgPose.midPointPose.orientation.w);
  Eigen::Quaternionf qtf1(qtf.w(), qtf.x(), qtf.y(), qtf.z());
  Eigen::Matrix3f rotMatrix = qtf1.toRotationMatrix();
  *transformation << rotMatrix(0,0), rotMatrix(0,1), rotMatrix(0,2), msgPose.midPointPose.position.x,
                     rotMatrix(1,0), rotMatrix(1,1), rotMatrix(1,2), msgPose.midPointPose.position.y, 
                     rotMatrix(2,0), rotMatrix(2,1), rotMatrix(2,2), msgPose.midPointPose.position.z, 
                                  0,              0,              0,                               1;
  (*pose).midPointPose = *transformation;
}


Eigen::Vector3d transformPoint(const tf::Stamped<tf::Point> & tfPointIn, const std::string & sourceFrame, const std::string & targetFrame){

  tf::TransformListener transformer;
  tf::Stamped<tf::Point> tfPointOut;

  transformer.waitForTransform(targetFrame, sourceFrame, ros::Time(0), ros::Duration(3.0));
  transformer.transformPoint(targetFrame, tfPointIn, tfPointOut);

  Eigen::Vector3d outputPoint(tfPointOut.getX(), tfPointOut.getY(), tfPointOut.getZ());

  return outputPoint;
}

Eigen::Vector3d transformVector(const tf::Stamped<tf::Vector3> & tfVectorIn, const std::string & sourceFrame, const std::string & targetFrame) {
  tf::TransformListener transformer;
  tf::Stamped<tf::Vector3> tfVectorOut;

  transformer.waitForTransform(targetFrame, sourceFrame, ros::Time(0), ros::Duration(3.0));
  transformer.transformVector(targetFrame, tfVectorIn, tfVectorOut);

  Eigen::Vector3d outputVector(tfVectorOut.getX(), tfVectorOut.getY(), tfVectorOut.getZ());

  return outputVector;
}

void drawDataRviz(GraspEvoContacts contacts, GraspEvoPose pose, ros::Publisher vis_pub, Eigen::Matrix4f transformation, std::vector<Eigen::Vector3d> *graspAndMiddlePointsWorldFrame, int *identifierMarkerRviz, std::vector<visualization_msgs::Marker> *markerRvizVector){

  // Grasping points
  for(int i=0; i<contacts.graspContactPoints.size(); i++){
    tf::Stamped<tf::Point> point;
    point.setX(contacts.graspContactPoints[i].x);
    point.setY(contacts.graspContactPoints[i].y);
    point.setZ(contacts.graspContactPoints[i].z);
    point.frame_id_ = "camera_depth_optical_frame";
    
    Eigen::Vector3d pointWorld = transformPoint(point, "camera_depth_optical_frame", "base_link");
    
    visualization_msgs::Marker marker;
    marker.header.frame_id = "base_link";
    marker.header.stamp = ros::Time();
    marker.id = *identifierMarkerRviz;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = pointWorld[0];
    marker.pose.position.y = pointWorld[1];
    marker.pose.position.z = pointWorld[2];
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.01;
    marker.scale.y = 0.01;
    marker.scale.z = 0.01;
    marker.color.a = 1.0; // Don't forget to set the alpha!
    if (i==0){
      marker.color.r = 1.0f;
    }
    else if(i==1){
      marker.color.g = 1.0f;
    }
    else{
      marker.color.b = 1.0f;
    }
    marker.lifetime = ros::Duration(1000000);

    vis_pub.publish(marker);
    
    (*graspAndMiddlePointsWorldFrame).push_back(pointWorld);
    (*markerRvizVector).push_back(marker);
    *identifierMarkerRviz = *identifierMarkerRviz + 1;
  }
  
  // Mid point
  tf::Stamped<tf::Point> point1;
  point1.setX(pose.midPointPose.translation().x());
  point1.setY(pose.midPointPose.translation().y());
  point1.setZ(pose.midPointPose.translation().z());
  point1.frame_id_ = "camera_depth_optical_frame";


  // tf::Stamped<tf::Point> point_baseX;
  // point_baseX.setX(1.0);
  // point_baseX.setY(0.0);
  // point_baseX.setZ(0.0);
  // point_baseX.frame_id_ = "camera_depth_optical_frame";
  // tf::Stamped<tf::Point> point_baseY;
  // point_baseY.setX(0.0);
  // point_baseY.setY(1.0);
  // point_baseY.setZ(0.0);
  // point_baseY.frame_id_ = "camera_depth_optical_frame";
  // tf::Stamped<tf::Point> point_baseZ;
  // point_baseZ.setX(0.0);
  // point_baseZ.setY(0.0);
  // point_baseZ.setZ(1.0);
  // point_baseZ.frame_id_ = "camera_depth_optical_frame";
  
// Transformadas de [[1, 0, 0], [0, 1, 0], [0, 0, 1]] en el "optical_depth_frame" a "base_link"
// 0.492294
// 0.288742
// 0.497312

// -0.507597
// -0.710832
// 0.463157

// -0.472495
// 0.288763
// -0.536636
  Eigen::Vector3d pointWorld1 = transformPoint(point1, "camera_depth_optical_frame", "base_link");
  visualization_msgs::Marker marker1;
  marker1.header.frame_id = "base_link";
  marker1.header.stamp = ros::Time();
  marker1.id = *identifierMarkerRviz;
  marker1.type = visualization_msgs::Marker::SPHERE;
  marker1.action = visualization_msgs::Marker::ADD;
  marker1.pose.position.x = pointWorld1[0];
  marker1.pose.position.y = pointWorld1[1];
  marker1.pose.position.z = pointWorld1[2];
  marker1.pose.orientation.x = 0.0;
  marker1.pose.orientation.y = 0.0;
  marker1.pose.orientation.z = 0.0;
  marker1.pose.orientation.w = 1.0;
  marker1.scale.x = 0.01;
  marker1.scale.y = 0.01;
  marker1.scale.z = 0.01;
  marker1.color.a = 1.0; // Don't forget to set the alpha!
  marker1.color.r = 1.0f;
  marker1.color.g = 1.0f;
  marker1.color.b = 0.0f;
  marker1.lifetime = ros::Duration(1000000);

  vis_pub.publish(marker1);
  
  (*graspAndMiddlePointsWorldFrame).push_back(pointWorld1);
  (*markerRvizVector).push_back(marker1);
  *identifierMarkerRviz = *identifierMarkerRviz + 1;
  

  // Axis in mid point
  /*for(int i=0;i<3;i++){
    tf::Stamped<tf::Vector3> axe;
    axe.setX(transformation(i,0));
    axe.setY(transformation(i,1));
    axe.setZ(transformation(i,2));
    axe.frame_id_ = "camera_depth_optical_frame";
  
    Eigen::Vector3d axeNew = transformVector(axe, "camera_depth_optical_frame", "base_link");
  
    visualization_msgs::Marker axis1;
    axis1.header.frame_id = "base_link";
    axis1.header.stamp = ros::Time();
    axis1.id = *identifierMarkerRviz;
    axis1.type = visualization_msgs::Marker::ARROW;
    axis1.action = visualization_msgs::Marker::ADD;
    geometry_msgs::Point p,p1;
    p.x = pointWorld1[0];
    p.y = pointWorld1[1];
    p.z = pointWorld1[2];
    axis1.points.push_back(p);
    p1.x = pointWorld1[0]+(axeNew[0]/10.0);
    p1.y = pointWorld1[1]+(axeNew[1]/10.0);
    p1.z = pointWorld1[2]+(axeNew[2]/10.0);
    axis1.points.push_back(p1);
    axis1.scale.x = 0.01;
    axis1.scale.y = 0.01;
    axis1.scale.z = 0.01;
    axis1.color.a = 1.0; // Don't forget to set the alpha!
    if (i==0){
      axis1.color.r = 1.0f;
    }
    else if(i==1){
      axis1.color.g = 1.0f;    
    }
    else{
      axis1.color.b = 1.0f;    
    }

    axis1.lifetime = ros::Duration(1000000);

    vis_pub.publish(axis1);
    (*markerRvizVector).push_back(axis1);
    *identifierMarkerRviz = *identifierMarkerRviz + 1;
  }*/
}

void print4x4Matrix (const Eigen::Matrix4d & matrix){
  printf ("Rotation matrix :\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
  printf ("Translation vector :\n");
  printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}

void task1(ros::NodeHandle nh, ros::Rate loop_rate, tf::Transform transformFrames, tf::TransformBroadcaster br){
    
  while(nh.ok() && transformReceived==false){
    br.sendTransform(tf::StampedTransform(transformFrames, ros::Time::now(), "base_link", "objectAxis")); 
    /*br.sendTransform(tf::StampedTransform(transformFrames[1], ros::Time::now(), "base_link", "robotiq_finger_2_halfLink"));
    br.sendTransform(tf::StampedTransform(transformFrames[2], ros::Time::now(), "base_link", "robotiq_finger_middle_halfLink")); */
    
    ros::spinOnce();   
    loop_rate.sleep(); 
  }  
}

/*void task2(pcl::PointCloud<pcl::PointXYZ> Final, std::vector<Eigen::Vector3d> *pointWorld){

  tf::Stamped<tf::Point> point1;
  Eigen::Vector3d singlePointWorld;
  for(int i=0;i<Final.points.size();i++){

    point1.setX(Final.points[i].x);
    point1.setY(Final.points[i].y);
    point1.setZ(Final.points[i].z);
    
    if (i==0){
      point1.frame_id_ = "robotiq_finger_1_halfLink";
    }
    if (i==1){
      point1.frame_id_ = "robotiq_finger_2_halfLink";
    }
    if (i==2){
      point1.frame_id_ = "robotiq_finger_middle_halfLink";
    }
    
    singlePointWorld = transformPoint(point1, point1.frame_id_, "base_link"); 
    (*pointWorld).push_back(singlePointWorld);   
    std::cout<<"Point "<<i<<point1<<" and new is "<<singlePointWorld<<std::endl;   
  }      
  transformReceived = true;
}*/

void moveCloseObject(std::vector<Eigen::Vector3d> graspAndMiddlePointsWorldFrame, ros::NodeHandle nh, ros::Rate loop_rate, ros::Publisher vis_pub, moveit::planning_interface::MoveGroupInterface *move_group_interface_arm, moveit::planning_interface::MoveGroupInterface::Plan *my_plan_arm, int *identifierMarkerRviz, std::vector<visualization_msgs::Marker> *markerRvizVector, char *inputChar, std::string saveFilesPath, int actualObject, float *gripper_value){

    std::vector<geometry_msgs::Pose> desiredPoints;
    std::vector<std::vector<double>> points;
    
    std::cout<<"Points to work with: "<<std::endl;
    for(int i=0;i<graspAndMiddlePointsWorldFrame.size();i++){
      std::vector<double> point {graspAndMiddlePointsWorldFrame[i][0], graspAndMiddlePointsWorldFrame[i][1], graspAndMiddlePointsWorldFrame[i][2]};
      
      std::cout<<"###########\n\n";
      std::cout<<point[0]<<" "<<point[1]<<" "<<point[2]<<std::endl;
      std::cout<<"###########\n\n";

      points.push_back(point);
    }
    std::cout<<std::endl;

    // Nuevo punto medio
    std::vector<double> new_mid_1 = {(points[1][0] + points[2][0])/2.0, (points[1][1] + points[2][1])/2.0, (points[1][2] + points[2][2])/2.0};
    std::vector<double> new_mid_2 = {(new_mid_1[0] + points[0][0])/2.0, (new_mid_1[1] + points[0][1])/2.0, (new_mid_1[2] + points[0][2])/2.0};
    
    // graspAndMiddlePointsWorldFrame[3][0] = new_mid_2[0];
    // graspAndMiddlePointsWorldFrame[3][1] = new_mid_2[1];
    // graspAndMiddlePointsWorldFrame[3][2] = new_mid_2[2];

    // Calcula los tres vectores posibles
    std::vector<double> vector1 {points[1][0] - points[0][0], points[1][1] - points[0][1], points[1][2] - points[0][2]};
    std::vector<double> vector2 {points[2][0] - points[0][0], points[2][1] - points[0][1], points[2][2] - points[0][2]};
    std::vector<double> vector3 {points[2][0] - points[1][0], points[2][1] - points[1][1], points[2][2] - points[1][2]};

    // Obtiene sus módulos
    float mod1 = sqrt(pow(vector1[0], 2) + pow(vector1[1], 2) + pow(vector1[2], 2));
    float mod2 = sqrt(pow(vector2[0], 2) + pow(vector2[1], 2) + pow(vector2[2], 2));
    float mod3 = sqrt(pow(vector3[0], 2) + pow(vector3[1], 2) + pow(vector3[2], 2));

    // Redondea los módulos
    mod1 = round(mod1 * 10000) / 10000.0;
    mod2 = round(mod2 * 10000) / 10000.0;
    mod3 = round(mod3 * 10000) / 10000.0;

    std::cout<<"V1: "<<vector1[0]<<" "<<vector1[1]<<" "<<vector1[2]<<std::endl;
    std::cout<<"V2: "<<vector2[0]<<" "<<vector2[1]<<" "<<vector2[2]<<std::endl;
    std::cout<<"V3: "<<vector3[0]<<" "<<vector3[1]<<" "<<vector3[2]<<std::endl<<std::endl<<std::endl;

    // std::cout<<"MOD 1: "<<mod1<<std::endl;
    // std::cout<<"MOD 2: "<<mod2<<std::endl;
    // std::cout<<"MOD 3: "<<mod3<<std::endl<<std::endl<<std::endl;

    


    std::vector<double> vectorX {vector1[1]*vector2[2] - vector1[2]*vector2[1], vector1[2]*vector2[0] - vector1[0]*vector2[2], vector2[1]*vector1[0] - vector2[0]*vector1[1]};
    if(vectorX[2] > 0){
      vectorX[0] = vectorX[0] * -1;
      vectorX[1] = vectorX[1] * -1;
      vectorX[2] = vectorX[2] * -1;
    }

    
        
    /*std::cout<<"Isolated point: "<<graspAndMiddlePointsWorldFrame[0][0]<<" "<<graspAndMiddlePointsWorldFrame[0][1]<<" "<<graspAndMiddlePointsWorldFrame[0][2]<<std::endl;
    std::cout<<"Middle point: "<<graspAndMiddlePointsWorldFrame[3][0]<<" "<<graspAndMiddlePointsWorldFrame[3][1]<<" "<<graspAndMiddlePointsWorldFrame[3][2]<<std::endl;
    std::cout<<"Middle point: "<<graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][0]<<" "<<graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][1]<<" "<<graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][2]<<std::endl;*/
    
    std::vector<double> vectorY {graspAndMiddlePointsWorldFrame[0][0]-graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][0], 
                                 graspAndMiddlePointsWorldFrame[0][1]-graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][1],   
                                graspAndMiddlePointsWorldFrame[0][2]-graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][2]}; 
    
    
    // Si hay un módulo que es muy pequeño quiere decir que hay al menos dos puntos similares, por lo que coge la
    //   la vertical del objeto como vector normal al plano
    if(mod1 == 0 or mod2 == 0 or mod3 == 0)
    {
      vectorX[0] = 0;
      vectorX[1] = 0;
      vectorX[2] = -1;

      vectorY[2] = 0;
    }

    std::vector<double> vectorZ {vectorX[1]*vectorY[2]-vectorY[1]*vectorX[2], vectorX[2]*vectorY[0]-vectorX[0]*vectorY[2], vectorX[0]*vectorY[1]-vectorY[0]*vectorX[1]};
    
    //std::cout<<"Final vector: "<<vectorZ[0]<<" "<<vectorZ[1]<<" "<<vectorZ[2]<<std::endl;
    
    double modX, modY, modZ;
    modX = sqrt(vectorX[0]*vectorX[0]+vectorX[1]*vectorX[1]+vectorX[2]*vectorX[2]);
    modY = sqrt(vectorY[0]*vectorY[0]+vectorY[1]*vectorY[1]+vectorY[2]*vectorY[2]);
    modZ = sqrt(vectorZ[0]*vectorZ[0]+vectorZ[1]*vectorZ[1]+vectorZ[2]*vectorZ[2]);
    
    for(int i=0;i<3; i++){
      vectorX [i]=  vectorX[i]/modX;
      vectorY [i]=  vectorY[i]/modY;
      vectorZ [i]=  vectorZ[i]/modZ;
    }

    

    std::cout<<"Final vector 1: "<<vectorX[0]<<" "<<vectorX[1]<<" "<<vectorX[2]<<std::endl;
    std::cout<<"Final vector 2: "<<vectorY[0]<<" "<<vectorY[1]<<" "<<vectorY[2]<<std::endl;
    std::cout<<"Final vector 3: "<<vectorZ[0]<<" "<<vectorZ[1]<<" "<<vectorZ[2]<<std::endl;
    
    

    std::fstream my_file;
    my_file.open(saveFilesPath+std::to_string(actualObject)+"_info.txt", std::ios::app);
    if (!my_file){
      ROS_INFO("ERROR: file was not created.");
    }
    else{
      my_file<<"Final vector 1: "<<vectorX[0]<<" "<<vectorX[1]<<" "<<vectorX[2]<<std::endl; 
      my_file<<"Final vector 2: "<<vectorY[0]<<" "<<vectorY[1]<<" "<<vectorY[2]<<std::endl;
      my_file<<"Final vector 3: "<<vectorZ[0]<<" "<<vectorZ[1]<<" "<<vectorZ[2]<<std::endl;  
      my_file.close();
    }  
    
    // Broadcast this axis system as objectAxis
    tf::TransformBroadcaster br;
    std::vector<Eigen::Vector3d> pointWorld;
    tf::Transform transformFrames;
    tf::Quaternion qM;    
    Eigen::Matrix3d matAux, matAux_;
    
    transformReceived = false;
    
    // We have a bias of +45 in x respect to objectAxis. Respect to base_link is in z -45
    Eigen::Matrix3d rotneg45z;
    rotneg45z << cos(45*M_PI/180), -sin(45*M_PI/180), 0, sin(45*M_PI/180), cos(45*M_PI/180), 0, 0, 0, 1;
    Eigen::Vector3d endLinkX, endLinkY, endLinkZ;
    endLinkX << -vectorZ[0], -vectorZ[1], -vectorZ[2];
    endLinkY << vectorY[0], vectorY[1], vectorY[2];
    endLinkZ << vectorX[0], vectorX[1], vectorX[2];

    Eigen::Matrix3d rotpos90z, rotpos90y;
    rotpos90z << cos(180*M_PI/180), -sin(180*M_PI/180), 0, sin(180*M_PI/180), cos(180*M_PI/180), 0, 0, 0, 1;
    rotpos90y << cos(90*M_PI/180), 0, sin(90*M_PI/180), 0, 1, 0, -sin(90*M_PI/180), 0, cos(90*M_PI/180);
    // endLinkX = rotpos90z * endLinkX;
    // endLinkY = rotpos90z * endLinkY;
    // endLinkZ = rotpos90z * endLinkZ;

    // endLinkX = rotpos90y * endLinkX;
    // endLinkY = rotpos90y * endLinkY;
    // endLinkZ = rotpos90y * endLinkZ;

    // auto endLinkX_aux = endLinkX;
    // endLinkX = endLinkZ;
    // endLinkZ = endLinkX_aux;




    matAux_ << endLinkX[0], endLinkY[0], endLinkZ[0], endLinkX[1], endLinkY[1], endLinkZ[1], endLinkX[2], endLinkY[2], endLinkZ[2];
    Eigen::Quaterniond qtfAux_(matAux_);

    

    std::cout<<"\n\n\n AAAAAAAAAAAAAAAAAa"<<std::endl;
    std::cout<<qtfAux_.x()<<std::endl;
    std::cout<<qtfAux_.y()<<std::endl;
    std::cout<<qtfAux_.z()<<std::endl;
    std::cout<<qtfAux_.w()<<std::endl;
    std::cout<<"\n\n\nAAAAAAAAAAAAAAAAaAAA"<<std::endl;

    endLinkX = rotneg45z*endLinkX;
    endLinkY = rotneg45z*endLinkY;
    endLinkZ = rotneg45z*endLinkZ;
    matAux << endLinkX[0], endLinkY[0], endLinkZ[0], endLinkX[1], endLinkY[1], endLinkZ[1], endLinkX[2], endLinkY[2], endLinkZ[2];



    Eigen::Quaterniond qtfAux(matAux);
    tf::Quaternion qtfDef(qtfAux.x(), qtfAux.y(), qtfAux.z(), qtfAux.w()); 
    
    transformFrames.setOrigin(tf::Vector3(graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][0], graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][1], graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][2]));
    transformFrames.setRotation(qtfDef);   

    std::cout<<"\n\n\n AAAAAAAAAAAAAAAAAa"<<std::endl;
    std::cout<<qtfAux.x()<<std::endl;
    std::cout<<qtfAux.y()<<std::endl;
    std::cout<<qtfAux.z()<<std::endl;
    std::cout<<qtfAux.w()<<std::endl;
    std::cout<<"\n\n\nAAAAAAAAAAAAAAAAaAAA"<<std::endl;


    std::thread t1(task1, nh, loop_rate, transformFrames, br);
    
    // Set two points in new axis system and get its position in world
    tf::Stamped<tf::Point> pointGrasp, pointPreGrasp, aux;
    pointGrasp.setX(-0.0); //-0.22
    pointGrasp.setY(0);
    pointGrasp.setZ(-0.245); //-0.235
    pointGrasp.frame_id_ = "objectAxis";
    
    pointPreGrasp.setX(-0.0);
    pointPreGrasp.setY(0);
    pointPreGrasp.setZ(-0.35);
    pointPreGrasp.frame_id_ = "objectAxis";

    
    tf::TransformListener listener;
    listener.waitForTransform("base_link", "objectAxis", ros::Time(0), ros::Duration(3.0));
    
    Eigen::Vector3d graspingPose = transformPoint(pointGrasp, "objectAxis", "base_link");
    Eigen::Vector3d preGraspingPose = transformPoint(pointPreGrasp, "objectAxis", "base_link");
    Eigen::Vector3d preGraspingPose2 = transformPoint(pointPreGrasp, "object", "base_link");
    // // ---------------------------------------------------

    // cout<<"PointGrasp: "<<pointGrasp<<std::endl;
    // cout<<"PointPreGrasp: "<<pointPreGrasp<<std::endl;
    // cout<<"graspAndMiddlePoints: "<<graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1]<<std::endl;   


    // IMPORTANTE: ajustar el valor del offset para que llegue a la posicion deseada, en una prueba se vio que se acercaba demasiado a la mesa
    // float offset_grasping = 0.22;
    // float offset_preGrasping = 0.35;

    // Eigen::Vector3d graspingPose(graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][0], graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][1], graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][2] );
    // Eigen::Vector3d preGraspingPose(graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][0], graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][1], graspAndMiddlePointsWorldFrame[graspAndMiddlePointsWorldFrame.size()-1][2] + 0.22);
    
    std::cout<<"\n\n\nREFERENCE FRAME \n\n";
    std::cout<<move_group_interface_arm->getPoseReferenceFrame()<<"\n\n\n";
    move_group_interface_arm->setPoseReferenceFrame("base_link");

    // Show and save points to future trajectory
    geometry_msgs::Pose desiredGrasp, desiredPreGrasp;
    desiredGrasp.position.x = preGraspingPose[0];
    desiredGrasp.position.y = preGraspingPose[1];
    desiredGrasp.position.z = preGraspingPose[2];
    desiredGrasp.orientation.x = qtfDef.x();
    desiredGrasp.orientation.y = qtfDef.y();
    desiredGrasp.orientation.z = qtfDef.z();
    desiredGrasp.orientation.w = qtfDef.w();
    desiredPoints.push_back(desiredGrasp);
    
    desiredPreGrasp.position.x = graspingPose[0];
    desiredPreGrasp.position.y = graspingPose[1];
    desiredPreGrasp.position.z = graspingPose[2];
    desiredPreGrasp.orientation.x = qtfDef.x();
    desiredPreGrasp.orientation.y = qtfDef.y();
    desiredPreGrasp.orientation.z = qtfDef.z();
    desiredPreGrasp.orientation.w = qtfDef.w();
    desiredPoints.push_back(desiredPreGrasp);

    
    std::cout<<desiredGrasp<<std::endl;
    std::cout<<desiredPreGrasp<<std::endl;

    // std::cout<<qtfDef.x()<<std::endl;
    // std::cout<<qtfDef.y()<<std::endl;
    // std::cout<<qtfDef.z()<<std::endl;
    // std::cout<<qtfDef.w()<<std::endl;


    
    for(int i=0;i<3;i++){    
    
      visualization_msgs::Marker axis1;
      axis1.header.frame_id = "base_link";
      axis1.header.stamp = ros::Time();
      axis1.id = *identifierMarkerRviz;
      axis1.type = visualization_msgs::Marker::ARROW;
      axis1.action = visualization_msgs::Marker::ADD;
      geometry_msgs::Point p,p1;
      p.x = preGraspingPose[0];
      p.y = preGraspingPose[1];
      p.z = preGraspingPose[2];
      axis1.points.push_back(p);
      axis1.scale.x = 0.01;
      axis1.scale.y = 0.01;
      axis1.scale.z = 0.01;
      axis1.color.a = 1.0; // Don't forget to set the alpha!
      if (i==0){
        axis1.color.r = 1.0f;
        p1.x = preGraspingPose[0]+(endLinkZ[0]);
        p1.y = preGraspingPose[1]+(endLinkZ[1]);
        p1.z = preGraspingPose[2]+(endLinkZ[2]);
        axis1.points.push_back(p1);
      } 
      else if(i==1){
        axis1.color.g = 1.0f; 
        p1.x = preGraspingPose[0]+(endLinkY[0]);
        p1.y = preGraspingPose[1]+(endLinkY[1]);
        p1.z = preGraspingPose[2]+(endLinkY[2]);
        axis1.points.push_back(p1);
      }
      else{
        axis1.color.b = 1.0f;  
        p1.x = preGraspingPose[0]+(endLinkX[0]);
        p1.y = preGraspingPose[1]+(endLinkX[1]);
        p1.z = preGraspingPose[2]+(endLinkX[2]);
        axis1.points.push_back(p1); 
      }
 
      axis1.lifetime = ros::Duration(1000000);

      vis_pub.publish(axis1);
      (*markerRvizVector).push_back(axis1);
      *identifierMarkerRviz = *identifierMarkerRviz + 1;
    }
    
    for(int i=0;i<3;i++){    
    
      visualization_msgs::Marker axis1;
      axis1.header.frame_id = "base_link";
      axis1.header.stamp = ros::Time();
      axis1.id = *identifierMarkerRviz;
      axis1.type = visualization_msgs::Marker::ARROW;
      axis1.action = visualization_msgs::Marker::ADD;
      geometry_msgs::Point p,p1;
      p.x = graspingPose[0];
      p.y = graspingPose[1];
      p.z = graspingPose[2];
      axis1.points.push_back(p);
      axis1.scale.x = 0.01;
      axis1.scale.y = 0.01;
      axis1.scale.z = 0.01;
      axis1.color.a = 1.0; // Don't forget to set the alpha!
      if (i==0){
        axis1.color.r = 1.0f;
        p1.x = graspingPose[0]+(endLinkZ[0]);
        p1.y = graspingPose[1]+(endLinkZ[1]);
        p1.z = graspingPose[2]+(endLinkZ[2]);
        axis1.points.push_back(p1);
      } 
      else if(i==1){
        axis1.color.g = 1.0f; 
        p1.x = graspingPose[0]+(endLinkY[0]);
        p1.y = graspingPose[1]+(endLinkY[1]);
        p1.z = graspingPose[2]+(endLinkY[2]);
        axis1.points.push_back(p1);
      }
      else{
        axis1.color.b = 1.0f;  
        p1.x = graspingPose[0]+(endLinkX[0]);
        p1.y = graspingPose[1]+(endLinkX[1]);
        p1.z = graspingPose[2]+(endLinkX[2]);
        axis1.points.push_back(p1); 
      }
 
      axis1.lifetime = ros::Duration(1000000);

      vis_pub.publish(axis1);
      (*markerRvizVector).push_back(axis1);
      *identifierMarkerRviz = *identifierMarkerRviz + 1;
    }
    
    // Move to that poses we said before
    (*move_group_interface_arm).setPlanningTime(5.0);
    (*move_group_interface_arm).setGoalTolerance(0.005);
    std::cout << "MOVE TO 'PRE_GRASP' POSITION" << std::endl;




    std::vector<geometry_msgs::Pose> waypoints;
    waypoints.push_back(desiredPoints[0]);
    waypoints.push_back(desiredPoints[1]);
    moveit_msgs::RobotTrajectory trajectory;

    // do{
      
    //   const double jump_threshold = 0.0;
    //   const double eef_step = 0.001;

    //   std::cout<<"Computing Cartesian Trajectory..."<<std::endl;
    //   double fraction = (*move_group_interface_arm).computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
    
    //   std::cout << "Insert 'n' to compute a new trajectory" << std::endl;
    //   std::cout << "Insert 's' to save this trajectory" << std::endl;
    //   std::cout << "Insert other letter to escape" << std::endl;
    //   std::cin >> *inputChar;
    // } while(*inputChar != 'n' and *inputChar != 's');

    if(*inputChar == 's'){
      std::cout << "Executing the trajectory..." << std::endl;
      (*move_group_interface_arm).execute(trajectory);   

      std::this_thread::sleep_for(std::chrono::seconds(2)); 
      *inputChar = ' ';      
    }

     

    else{
      // Reset '*inputChar' flag
      *inputChar = ' ';

      do{
        const double jump_threshold = 0.0;
        const double eef_step = 0.001;
        
        (*move_group_interface_arm).setPoseTarget(desiredPoints[0]);
        std::cout << "Preparing second trajectory preGrasp point..." << std::endl;
        (*move_group_interface_arm).plan(*my_plan_arm);

        std::cout << "Insert 'n' to compute a new trajectory" << std::endl;
        std::cout << "Insert 's' to save this trajectory" << std::endl;
        std::cout << "Insert other letter to escape" << std::endl;
        std::cin >> *inputChar;
      } while(*inputChar != 'n' and *inputChar != 's');
    

      if(*inputChar == 's'){
        std::cout << "Executing the trajectory..." << std::endl;
        (*move_group_interface_arm).execute(*my_plan_arm);

        // Reset '*inputChar' flag
        *inputChar = ' '; 

        std::this_thread::sleep_for(std::chrono::seconds(2));  

        do{
          const double jump_threshold = 0.0;
          const double eef_step = 0.01;
          
          (*move_group_interface_arm).setPoseTarget(desiredPoints[1]);
          std::cout << "Preparing second trajectory Grasping point..." << std::endl;
          (*move_group_interface_arm).plan(*my_plan_arm);
          
          std::cout << "Insert 'n' to compute a new trajectory" << std::endl;
          std::cout << "Insert 's' to save this trajectory" << std::endl;
          std::cout << "Insert other letter to escape" << std::endl;
          std::cin >> inputChar;
        } while(*inputChar != 'n' and *inputChar != 's');  

        if(*inputChar == 's') 
        {
          std::cout << "Executing the trajectory..." << std::endl;
          (*move_group_interface_arm).execute(*my_plan_arm);

          // Reset '*inputChar' flag
          *inputChar = ' '; 
        }     
      }
    }
  
    std::this_thread::sleep_for(std::chrono::seconds(2));   
    
    // Rejoin programs
    transformReceived = true;
    t1.join();
}

void contactCallback(const gazebo_msgs::ContactsStateConstPtr &contact1, const gazebo_msgs::ContactsStateConstPtr &contact2, const gazebo_msgs::ContactsStateConstPtr &contactMiddle){  
  if (contact1->states.size() != 0){
    contact1_b = true;
  }
  else{
    contact1_b = false;
    contact1_prev = false;
  }
  
  if (contact2->states.size() != 0){
    contact2_b = true;
  }
  else{
    contact2_b = false;
    contact2_prev = false;
  }
  
  if (contactMiddle->states.size() != 0){
    contact3_b = true;
  }
  else{
    contact3_b = false;
    contact3_prev = false;
  }
  if (contact1->states.size() != 0 and contact2->states.size() != 0 and contactMiddle->states.size() != 0){
    if(contact1_prev and contact2_prev and contact3_prev)
    {
      contact = true;
    }
  }
}

void closeGripper(ros::NodeHandle nh, moveit::planning_interface::MoveGroupInterface *move_group_interface_gripper, moveit::planning_interface::MoveGroupInterface::Plan *my_plan_arm, float gripper_value){

  (*move_group_interface_gripper).setJointValueTarget((*move_group_interface_gripper).getNamedTargetValues("open"));
  (*move_group_interface_gripper).plan(*my_plan_arm);
  (*move_group_interface_gripper).execute(*my_plan_arm);

  std::cout<<"\n\n\n\n"<<gripper_value<<"\n\n\n\n";

  // Subscribe to sensor contact messages
  contact = false;
  contact1_b = false;
  contact2_b = false;
  contact3_b = false;
  contact1_prev = false;
  contact2_prev = false;
  contact3_prev = false;
  message_filters::Subscriber<gazebo_msgs::ContactsState> finger1Contact(nh, "/aurova/robotiq_finger_1_link_3_contact_sensor", 10);
  message_filters::Subscriber<gazebo_msgs::ContactsState> finger2Contact(nh, "/aurova/robotiq_finger_2_link_3_contact_sensor", 10);
  message_filters::Subscriber<gazebo_msgs::ContactsState> fingerMiddleContact(nh, "/aurova/robotiq_finger_middle_link_3_contact_sensor", 10);
    
  typedef message_filters::sync_policies::ApproximateTime<gazebo_msgs::ContactsState, gazebo_msgs::ContactsState, gazebo_msgs::ContactsState> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), finger1Contact, finger2Contact, fingerMiddleContact);
  sync.registerCallback(boost::bind(&contactCallback, _1, _2, _3));
    
  // Close gripper until collision
  std::cout << "GRASPING OBJECT" << std::endl;
  while(contact == false){
    
    geometry_msgs::PoseStamped curr_pose = (*move_group_interface_gripper).getCurrentPose();
    std::vector<double> joint_values = (*move_group_interface_gripper).getCurrentJointValues();
    std::vector<std::string> joint_names = (*move_group_interface_gripper).getJoints();

    int mult_ini = 6;  
    
    for (int i=0;i<joint_values.size(); i++){
      if(i == 0 and not contact1_prev)
      {
        int mult = mult_ini;
        if(contact1_b)
        {
          mult = mult_ini / 2.0;
        }
        joint_values[i] += mult * 0.0174533 ;//gripper_value;
        joint_values[i+2] = -0.61085;
        contact1_prev = contact1_b;
      }

      else if(i == 3 and not contact2_prev)
      {
        int mult = mult_ini;
        if(contact2_b)
        {
          mult = mult_ini / 2.0;
        }
        joint_values[i] += mult * 0.0174533 ;//gripper_value;
        joint_values[i+2] = -0.61085;
        contact2_prev = contact2_b;
      }

      else if(i == 6 and not contact3_prev)
      {
        int mult = mult_ini;
        if(contact3_b)
        {
          mult = mult_ini / 2.0;
        }
        joint_values[i] += mult * 0.0174533 ;//gripper_value;
        joint_values[i+2] = -0.61085;
        contact3_prev = contact3_b;
      }
      // if (i==0 or i==3 or i==6){
      //   joint_values[i] = joint_values[i]+0.0174533*5;
      // }
    }
    
       
    (*move_group_interface_gripper).setJointValueTarget(joint_values);
    (*move_group_interface_gripper).plan(*my_plan_arm);
    (*move_group_interface_gripper).execute(*my_plan_arm);  
  }

  geometry_msgs::PoseStamped curr_pose = (*move_group_interface_gripper).getCurrentPose();
  std::vector<double> joint_values = (*move_group_interface_gripper).getCurrentJointValues();
  std::vector<std::string> joint_names = (*move_group_interface_gripper).getJoints();

  // ------- Movimiento Extra de Cierre para Apretar el Objeto -------
  for (int i=0;i<joint_values.size(); i++){
    if (i==0 or i==3 or i==6){
      joint_values[i] = joint_values[i]+0.0174533*0.9;
    }
  }   

  (*move_group_interface_gripper).setJointValueTarget(joint_values);
  (*move_group_interface_gripper).plan(*my_plan_arm);
  (*move_group_interface_gripper).execute(*my_plan_arm); 
  // ------------------------------------------------------------

  // std::string aux;

  // cin>>aux;

  std::this_thread::sleep_for(std::chrono::seconds(2));  
    
  if (contact == true){
    finger1Contact.unsubscribe();
    finger2Contact.unsubscribe();
    fingerMiddleContact.unsubscribe();                
  }
  
}

void armUpAndReset(moveit::planning_interface::MoveGroupInterface *move_group_interface_arm, moveit::planning_interface::MoveGroupInterface::Plan *my_plan_arm, std::string saveFilesPath, int actualObject, moveit::planning_interface::MoveGroupInterface *move_group_interface_gripper, int identifierMarkerRviz, ros::Publisher vis_pub, std::vector<visualization_msgs::Marker> *markerRvizVector){

  (*move_group_interface_arm).setPlanningTime(5.0);
  (*move_group_interface_arm).setGoalTolerance(0.0001);
  std::cout << "MOVE TO 'INITIAL' POSITION" << std::endl;

  (*move_group_interface_arm).setJointValueTarget((*move_group_interface_arm).getNamedTargetValues("home"));
  (*move_group_interface_arm).plan(*my_plan_arm);
  (*move_group_interface_arm).execute(*my_plan_arm);
  
  std::this_thread::sleep_for(std::chrono::seconds(2));
    
  bool decision = false;
  do{
    std::cout<<"Was test successful? (y/n)"<<std::endl;
      
    char option;
    std::cin>>option;
    decision = false;
    
    if (option=='y'){
      decision = true;
        
      std::fstream my_file;
      my_file.open(saveFilesPath+std::to_string(actualObject)+"_info.txt", std::ios::app);
      if (!my_file){
        ROS_INFO("ERROR: file was not created.");
      }
      else{
        my_file<<"FINAL RESULT: grasp SUCCESSFUL";        
        my_file.close();
      }  
    }
    else if (option=='n'){
      decision = true;
        
      std::fstream my_file;
      my_file.open(saveFilesPath+std::to_string(actualObject)+"_info.txt", std::ios::app);
      if (!my_file){
        ROS_INFO("ERROR: file was not created.");
      }
      else{
        my_file<<"FINAL RESULT: grasp NOT SUCCESSFUL";        
        my_file.close();
      } 
    }
    else{
      ROS_INFO("Option not valid. Try again!");
    }      
  } while(decision == false);
    
  (*move_group_interface_gripper).setPlanningTime(5.0);
  (*move_group_interface_gripper).setGoalTolerance(0.0001);
  std::cout << "MOVE TO 'RESET' POSITION" << std::endl;

  (*move_group_interface_gripper).setJointValueTarget((*move_group_interface_gripper).getNamedTargetValues("open"));
  (*move_group_interface_gripper).plan(*my_plan_arm);
  (*move_group_interface_gripper).execute(*my_plan_arm); 
  
  std::this_thread::sleep_for(std::chrono::seconds(2));
  
  //Remove all markers
  for (int i=0; i<identifierMarkerRviz; i++){
  
    (*markerRvizVector)[i].action = visualization_msgs::Marker::DELETE;
    (*markerRvizVector)[i].header.stamp = ros::Time(); 
    
    vis_pub.publish((*markerRvizVector)[i]);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  
  }  
}

int main(int argc, char *argv[]){

  // This must be called before anything else ROS-related
  ros::init(argc, argv, "moveit_grasp");

  // Create a ROS node handle
  ros::NodeHandle nh("~");
  ros::Rate loop_rate(10);
  
  // Get parameters from launch file
  std::string saveFilesPath;
  std::string mode;

  nh.getParam("save_files_path", saveFilesPath);
  nh.getParam("mode", mode);
  
  graspResult = false;
  int actualObject = 0;
  
  // Moveit 
  // For getting info of robot state
  ros::AsyncSpinner spinner(1);
  spinner.start();
  
  // Nombres de los grupos a controlar
  static const std::string PLANNING_GROUP_ARM = "arm";
  static const std::string PLANNING_GROUP_GRIPPER = "gripper";
  static const std::string PLANNING_GROUP_GRIPPER_MODE = "gripper_mode";
  
  moveit::planning_interface::MoveGroupInterface move_group_interface_arm(PLANNING_GROUP_ARM);
  moveit::planning_interface::MoveGroupInterface move_group_interface_gripper(PLANNING_GROUP_GRIPPER);
  moveit::planning_interface::MoveGroupInterface move_group_interface_gripper_mode(PLANNING_GROUP_GRIPPER_MODE);
  
  // Rviz publish things
  ros::Publisher vis_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);
  
  moveit::planning_interface::MoveGroupInterface::Plan my_plan_arm2; 

  (move_group_interface_arm).setJointValueTarget((move_group_interface_arm).getNamedTargetValues("home"));
  (move_group_interface_arm).plan(my_plan_arm2);
  (move_group_interface_arm).execute(my_plan_arm2); 

  geometry_msgs::PoseStamped current_pose;
  std::vector<double> joint_values;
  std::vector<std::string> joint_names;
  std::vector<geometry_msgs::Pose> keyPositionsVector;
   
  while(ros::ok()){  
  
    sub = nh.subscribe("/aurova/grasp/result", 1, graspResultCallback);
    while (graspResult == false){
      ros::spinOnce();  
    }    
    
    // Translate received message
    int identifierMarkerRviz = 0;
    std::vector<visualization_msgs::Marker> markerRvizVector;
    GraspEvoContacts contacts;
    float ranking;
    GraspEvoPose pose;
    Eigen::Matrix4f transformation;
    
    translateMessage(&contacts, &ranking, &pose, &transformation);

    // Draw in rviz Grasp points and middle point
    std::vector<Eigen::Vector3d> graspAndMiddlePointsWorldFrame;
    drawDataRviz(contacts, pose, vis_pub, transformation, &graspAndMiddlePointsWorldFrame, &identifierMarkerRviz, &markerRvizVector);
  
    // Move gripper close to grasp points
    moveit::planning_interface::MoveGroupInterface::Plan my_plan_arm;  
    char inputChar = ' ';
    float gripper_value = 0.0;
    moveCloseObject(graspAndMiddlePointsWorldFrame, nh, loop_rate, vis_pub, &move_group_interface_arm, &my_plan_arm, &identifierMarkerRviz, &markerRvizVector, &inputChar, saveFilesPath, actualObject, &gripper_value);    

    // Close gripper to grasp object
    if(inputChar == ' ')
    {
      closeGripper(nh, &move_group_interface_gripper, &my_plan_arm, gripper_value);
    }

    // Arm up to check if object is grasped and prepare everything for next test
    armUpAndReset(&move_group_interface_arm, &my_plan_arm, saveFilesPath, actualObject, &move_group_interface_gripper, identifierMarkerRviz, vis_pub, &markerRvizVector);
    
    // Advertise we have finished
    ros::Publisher finished_pub = nh.advertise<std_msgs::Int32>("/aurova/moveit_grasp_finished", 1);
    std_msgs::Int32 msg;
    msg.data = 1;
        
    int loop = 0;
    while(loop < 2){      
      ros::spinOnce();
      finished_pub.publish(msg);
      ros::Duration(0.5).sleep(); 
      loop = loop + 1;
    }    
        
    // Other variables to complete the loop    
    ros::spinOnce();
    graspResult = false;
    actualObject = actualObject + 1;   
  }
      
  return 0;
}
