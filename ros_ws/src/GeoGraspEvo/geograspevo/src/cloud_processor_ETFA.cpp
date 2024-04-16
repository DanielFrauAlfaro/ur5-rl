#include <iostream>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <pcl/filters/passthrough.h>

#include <pcl/io/pcd_io.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <geograspevo/GeoGraspEvo.h>

pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cloud viewer"));

template<typename T, typename U>
void voxelizeCloud(const T & inputCloud, const float & leafSize, T outputCloud) {
  U voxelFilter;
  voxelFilter.setInputCloud(inputCloud);
  voxelFilter.setLeafSize(leafSize, leafSize, leafSize);
  voxelFilter.filter(*outputCloud);
}

// callback signature
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr & inputCloudMsg) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::fromROSMsg(*inputCloudMsg, *cloud);
  
  /*******************************************************************************************************************/
  system("exec rm -r /ignacio/Desktop/tmp/*");
    
  pcl::io::savePCDFileASCII ("/ignacio/Desktop/tmp/a_first.pcd", *cloud);
  std::cerr << "Saved " << (*cloud).size () << " data points to a_first.pcd." << std::endl;
  /*******************************************************************************************************************/

  // Remove NaN values and make it dense
  std::vector<int> nanIndices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, nanIndices);

  // Remove background points
  pcl::PassThrough<pcl::PointXYZRGB> ptFilter;
  ptFilter.setInputCloud(cloud);
  ptFilter.setFilterFieldName("z");
  ptFilter.setFilterLimits(0.0, 0.75);
  ptFilter.filter(*cloud);

  ptFilter.setInputCloud(cloud);
  ptFilter.setFilterFieldName("y");
  ptFilter.setFilterLimits(-0.3, 0.3);
  ptFilter.filter(*cloud);

  ptFilter.setInputCloud(cloud);
  ptFilter.setFilterFieldName("x");
  ptFilter.setFilterLimits(-0.40, 0.40);
  ptFilter.filter(*cloud);
  
  /*******************************************************************************************************************/
  pcl::io::savePCDFileASCII ("/ignacio/Desktop/tmp/a_second.pcd", *cloud);
  std::cerr << "Saved " << (*cloud).size () << " data points to a_second.pcd." << std::endl;
  /*******************************************************************************************************************/

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZRGB> sacSegmentator;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPlane(new pcl::PointCloud<pcl::PointXYZRGB>());

  sacSegmentator.setModelType(pcl::SACMODEL_PLANE);
  sacSegmentator.setMethodType(pcl::SAC_RANSAC);
  sacSegmentator.setMaxIterations(50);
  sacSegmentator.setDistanceThreshold(0.02);
  sacSegmentator.setInputCloud(cloud);
  sacSegmentator.segment(*inliers, *coefficients);

  // Remove the planar inliers, extract the rest
  pcl::ExtractIndices<pcl::PointXYZRGB> indExtractor;
  indExtractor.setInputCloud(cloud);
  indExtractor.setIndices(inliers);
  indExtractor.setNegative(false);

  // Get the points associated with the planar surface
  indExtractor.filter(*cloudPlane);

  // Remove the planar inliers, extract the rest
  indExtractor.setNegative(true);
  indExtractor.filter(*cloud);

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> clusterIndices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ecExtractor;
  ecExtractor.setClusterTolerance(0.01);
  ecExtractor.setMinClusterSize(750);
  ecExtractor.setSearchMethod(tree);
  ecExtractor.setInputCloud(cloud);
  ecExtractor.extract(clusterIndices);
  
  /*pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> planeColor(cloudPlane, 0, 255, 0);

    viewer->removeAllPointClouds();
    viewer->removeAllShapes();

    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Main cloud");
    viewer->addPointCloud<pcl::PointXYZRGB>(cloudPlane, planeColor, "Plane");

    viewer->spinOnce();*/
    
  /*******************************************************************************************************************/  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr a_third(new pcl::PointCloud<pcl::PointXYZRGB>()); 
  
  pcl::copyPointCloud(*cloudPlane, *a_third);
  
  for (int a=0; a<a_third->points.size(); a++){
    // El plano lo pintamos de verde
    a_third->points[a].r = 0;
    a_third->points[a].g = 255;
    a_third->points[a].b = 0;  
  }  
  
  /*pcl::io::savePCDFileASCII ("/ignacio/Desktop/iros_2023/catkin_ws/src/GeoGraspEvo/experiments/a_third.pcd", *a_third);
  std::cerr << "Saved " << (*a_third).size () << " data points to a_third.pcd." << std::endl;*/
  /*******************************************************************************************************************/

  if (clusterIndices.empty()) {
    // Visualize the result
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> planeColor(cloudPlane, 0, 255, 0);

    viewer->removeAllPointClouds();
    viewer->removeAllShapes();

    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Main cloud");
    viewer->addPointCloud<pcl::PointXYZRGB>(cloudPlane, planeColor, "Plane");

    viewer->spinOnce();
  }
  else {
    std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin();
    int objectNumber = 0;

    viewer->removeAllPointClouds();
    viewer->removeAllShapes();

    // Every cluster found is considered an object
    //int i=0;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colouredPCUnmodified(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::string cad = std::string("/ignacio/Desktop/tmp/a_second")+".pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (cad, *colouredPCUnmodified) == -1){
          PCL_ERROR ("Couldn't read file \n");
    }
      
    /*******************************************************************************************************************/
    std::vector<std::vector<int>> colours = {{255,0,0},{0,0,255},{255,255,0},{255,0,255},{0,255,255}};   
    int objNumber = 0;    
    /*******************************************************************************************************************/
    
    for (it = clusterIndices.begin(); it != clusterIndices.end(); ++it) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr objectCloud(new pcl::PointCloud<pcl::PointXYZRGB>());

      for (std::vector<int>::const_iterator pit = it->indices.begin(); 
          pit != it->indices.end(); ++pit)
        objectCloud->points.push_back(cloud->points[*pit]);

      objectCloud->width = objectCloud->points.size();
      objectCloud->height = 1;
      objectCloud->is_dense = true;
      
      /*******************************************************************************************************************/
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr a_tmp(new pcl::PointCloud<pcl::PointXYZRGB>()); 
  
      pcl::copyPointCloud(*objectCloud, *a_tmp);
  
      for (int a=0; a<a_tmp->points.size(); a++){
        // El plano lo pintamos de verde
        a_tmp->points[a].r = colours[objNumber][0];
        a_tmp->points[a].g = colours[objNumber][1];
        a_tmp->points[a].b = colours[objNumber][2];  
      }  
      
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr a_end(new pcl::PointCloud<pcl::PointXYZRGB>()); 
      a_end  = a_third;
      (*a_end) += (*a_tmp);
   
      pcl::io::savePCDFileASCII ("/ignacio/Desktop/tmp/a_third"+std::to_string(objNumber)+".pcd", *a_third);
      std::cerr << "Saved " << (*a_end).size () << " data points to "<<"a_third"<<std::to_string(objNumber)<<".pcd" << std::endl;
      
      std::string cad = std::string("/ignacio/Desktop/tmp/b")+"_obj"+std::to_string(objNumber)+".pcd";
      std::string cad2 = std::string("b")+"_obj"+std::to_string(objNumber)+".pcd";
      pcl::io::savePCDFileASCII (cad, *objectCloud);
      std::cerr << "Saved " << (*objectCloud).size () << " data points to "+cad2<< std::endl;
      
      objNumber=objNumber+1;
      /*******************************************************************************************************************/

      // Create and initialise GeoGraspEvo
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPlaneXYZ(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::PointCloud<pcl::PointXYZ>::Ptr objectCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::copyPointCloud(*cloudPlane, *cloudPlaneXYZ);
      pcl::copyPointCloud(*objectCloud, *objectCloudXYZ);
      
      /*pcl::io::savePCDFileASCII ("/ignacio/Desktop/rl_pybullet/"+std::to_string(i)+".pcd", *objectCloudXYZ);
      std::cerr << "Saved " << objectCloudXYZ->points.size () << " data points to test_pcd.pcd." << std::endl;*/
      //i=i+1;

      GeoGraspEvo geoGraspPoints;
      geoGraspPoints.setBackgroundCloud(cloudPlaneXYZ);
      geoGraspPoints.setObjectCloud(objectCloudXYZ);
      geoGraspPoints.setGripTipSize(25); // 25mm grip
      //geoGraspPoints.setApertures({{10,15},{30,35},{-10,-15},{-30,-35}});
      geoGraspPoints.setApertures({{-0.015,-0.025}, {-0.005,0.010}, {0.035,0.040}});
      geoGraspPoints.setNumberFingers(4);
      geoGraspPoints.setGrasps(5); // Keep track only of the best

      // Calculate grasping points
      geoGraspPoints.compute();
      
      std::vector<float> var = geoGraspPoints.getRankings();
      std::vector<GraspEvoContacts> var1 = geoGraspPoints.getGrasps();
      
      for (int i=0; i<var1.size(); i++){
        std::cout<<"P"<<i<<": ";
        for (int j=0; j<var1[i].graspContactPoints.size(); j++){
          std::cout<<var1[i].graspContactPoints[j]<<" ";
        }
        std::cout<<std::endl;
      }
      
      std::cout<<"RR: ";
      for (int i=0; i<var.size(); i++){
        std::cout<<var[i]<<" ";
      }
      std::cout<<std::endl;
      
      if (var[0] > 0.1){
      
        /***********************************************************************************************************************************************/
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colouredPC(new pcl::PointCloud<pcl::PointXYZRGB>);
        std::string cad = std::string("/ignacio/Desktop/tmp/b")+"_obj"+std::to_string((objNumber-1))+".pcd";
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (cad, *colouredPC) == -1){
          PCL_ERROR ("Couldn't read file \n");
        }
        
          
        for(int length=0; length<(*colouredPC).points.size(); length++){    
          (*colouredPC).points[length].r = 0;
          (*colouredPC).points[length].g = 0;
          (*colouredPC).points[length].b = 0;
        }  
  
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxelCloud2(new pcl::PointCloud<pcl::PointXYZRGB>);
        float voxelRadius = (25 / 1000.0)/2.0;
        voxelizeCloud<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, 
                      pcl::VoxelGrid<pcl::PointXYZRGB> >(colouredPC, voxelRadius, voxelCloud2);
  
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr newScene(new pcl::PointCloud<pcl::PointXYZRGB>);        
        for(int i=0; i<voxelCloud2->points.size(); i++){
          float radius = 0.0005;
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>);
          for(float theta=0; theta<2*3.14; theta=theta+5*0.017){
            for(float phi=0; phi<3.14; phi = phi+5*0.017){
    
              pcl::PointXYZRGB punto;
              punto.x = radius * cos(theta) * sin(phi) + (voxelCloud2->points[i]).x;
              punto.y = radius * sin(theta) * sin(phi) + (voxelCloud2->points[i]).y;
              punto.z = radius * cos(phi) + (voxelCloud2->points[i]).z;
              punto.r = 0;
              punto.g = 0;
              punto.b = 0;
      
              (scene->points).push_back(punto);
            }
          } 
          (*newScene) += (*scene); 
        }        
  
        *voxelCloud2 = *newScene;
        
        std::vector<std::vector<int>> colourRed = {{178,0,0},{198,0,0},{217,0,0},{237,0,0},{255,0,0}};   
        std::vector<std::vector<int>> colourYellow = {{255,255,0},{255,255,51},{255,255,102},{255,255,153},{255,255,204}};   
        std::vector<std::vector<int>> colourMagenta = {{250,45,208},{255,85,227},{255,145,237},{255,185,243},{255,225,250}}; //oscuro a claro
        std::vector<std::vector<int>> colourCyan = {{0,206,209},{64,224,208},{175,238,238},{0,255,255},{224,255,255}};
      
        float radius = 0.005;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ranking(new pcl::PointCloud<pcl::PointXYZRGB>);
        for(int point=var1.size()-1; point>=0; point--){
          for(int finger=0; finger<var1[point].graspContactPoints.size(); finger++){
            for(float theta=0; theta<2*3.14; theta=theta+2.5*0.017){
              for(float phi=0; phi<3.14; phi = phi+2.5*0.017){
    
                pcl::PointXYZRGB punto;
                punto.x = radius * cos(theta) * sin(phi) + var1[point].graspContactPoints[finger].getVector3fMap()[0];
                punto.y = radius * sin(theta) * sin(phi) + var1[point].graspContactPoints[finger].getVector3fMap()[1];
                punto.z = radius * cos(phi) + var1[point].graspContactPoints[finger].getVector3fMap()[2];
                
                if (finger==0){
                  punto.r = colourRed[point][0];
                  punto.g = colourRed[point][1];
                  punto.b = colourRed[point][2];
                }
                else if (finger==1){
                  punto.r = colourYellow[point][0];
                  punto.g = colourYellow[point][1];
                  punto.b = colourYellow[point][2];                
                }
                else if (finger==2){
                  punto.r = colourMagenta[point][0];
                  punto.g = colourMagenta[point][1];
                  punto.b = colourMagenta[point][2];                
                }
                else{
                  punto.r = colourCyan[point][0];
                  punto.g = colourCyan[point][1];
                  punto.b = colourCyan[point][2]; 
                }      
                (ranking->points).push_back(punto);
              }
            }
          }
        }

        (*voxelCloud2) += (*ranking);
        
        cad = std::string("/ignacio/Desktop/tmp/c")+"_result"+std::to_string((objNumber-1))+".pcd";
        cad2 = std::string("c")+"_result"+std::to_string((objNumber-1))+".pcd";
  
        pcl::io::savePCDFileASCII (cad, *voxelCloud2);
        std::cerr << "Saved " << (*voxelCloud2).size () << " data points to " <<cad2<<"." << std::endl;
        
        radius = 0.005;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ranking1(new pcl::PointCloud<pcl::PointXYZRGB>);
        for(int point=0; point<1; point++){
          for(int finger=0; finger<var1[point].graspContactPoints.size(); finger++){
            for(float theta=0; theta<2*3.14; theta=theta+2.5*0.017){
              for(float phi=0; phi<3.14; phi = phi+2.5*0.017){
    
                pcl::PointXYZRGB punto;
                punto.x = radius * cos(theta) * sin(phi) + var1[point].graspContactPoints[finger].getVector3fMap()[0];
                punto.y = radius * sin(theta) * sin(phi) + var1[point].graspContactPoints[finger].getVector3fMap()[1];
                punto.z = radius * cos(phi) + var1[point].graspContactPoints[finger].getVector3fMap()[2];
                
                if (finger==0){
                  punto.r = colourRed[point][0];
                  punto.g = colourRed[point][1];
                  punto.b = colourRed[point][2];
                }
                else if (finger==1){
                  punto.r = colourYellow[point][0];
                  punto.g = colourYellow[point][1];
                  punto.b = colourYellow[point][2];                
                }
                else if (finger==2){
                  punto.r = colourMagenta[point][0];
                  punto.g = colourMagenta[point][1];
                  punto.b = colourMagenta[point][2];                
                }
                else{
                  punto.r = colourCyan[point][0];
                  punto.g = colourCyan[point][1];
                  punto.b = colourCyan[point][2]; 
                }      
                (ranking1->points).push_back(punto);
              }
            }
          }
        }

        (*colouredPCUnmodified) += (*ranking1);
        
        cad = std::string("/ignacio/Desktop/tmp/c_resultFinal")+".pcd";
        cad2 = std::string("c")+"_resultFinal.pcd";
  
        pcl::io::savePCDFileASCII (cad, *colouredPCUnmodified);
          
        /***********************************************************************************************************************************************/
        
        // Extract best pair of points
        GraspEvoContacts bestGrasp = geoGraspPoints.getBestGrasp();
        GraspEvoPose bestPose = geoGraspPoints.getBestGraspPose();                


        // Visualize the result
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(objectCloud);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> planeRGB(cloudPlane);

        std::string objectLabel = "";
        std::ostringstream converter;

        converter << objectNumber;
        objectLabel += converter.str();
        objectLabel += "-";

        viewer->addPointCloud<pcl::PointXYZRGB>(objectCloud, rgb, objectLabel + "Object");
        viewer->addPointCloud<pcl::PointXYZRGB>(cloudPlane, planeRGB, objectLabel + "Plane");


      
        for (int i=0; i<bestGrasp.graspContactPoints.size(); i++){
          std::string cad = "Best grasp n." + std::to_string(i);
          viewer->addSphere(bestGrasp.graspContactPoints[i], 0.01, 255, 255, 255, objectLabel + cad);
        }
        
        /*std::string cad = "Best grasp n." + std::to_string(0);
        viewer->addSphere(bestGrasp.graspContactPoints[0], 0.01, 0.835, 0.89, 0.835, objectLabel + cad);
        
        cad = "Best grasp n." + std::to_string(1);
        viewer->addSphere(bestGrasp.graspContactPoints[1], 0.01, 0.0, 0.4, 0.49, objectLabel + cad);
        
        cad = "Best grasp n." + std::to_string(2);
        viewer->addSphere(bestGrasp.graspContactPoints[2], 0.01, 1.0, 0.819, 0.168, objectLabel + cad);*/

        pcl::ModelCoefficients axeX;
        axeX.values.resize (6);    // We need 6 values
        axeX.values[0] = bestPose.midPointPose.translation()[0];
        axeX.values[1] = bestPose.midPointPose.translation()[1];
        axeX.values[2] = bestPose.midPointPose.translation()[2];
        axeX.values[3] = bestPose.midPointPose.linear()(0, 0);
        axeX.values[4] = bestPose.midPointPose.linear()(1, 0);
        axeX.values[5] = bestPose.midPointPose.linear()(2, 0);

        viewer->addLine(axeX, objectLabel + "Pose axeX");

        pcl::ModelCoefficients axeY;
        axeY.values.resize (6);    // We need 6 values
        axeY.values[0] = bestPose.midPointPose.translation()[0];
        axeY.values[1] = bestPose.midPointPose.translation()[1];
        axeY.values[2] = bestPose.midPointPose.translation()[2];
        axeY.values[3] = bestPose.midPointPose.linear()(0, 1);
        axeY.values[4] = bestPose.midPointPose.linear()(1, 1);
        axeY.values[5] = bestPose.midPointPose.linear()(2, 1);

        viewer->addLine(axeY, objectLabel + "Pose axeY");

        pcl::ModelCoefficients axeZ;
        axeZ.values.resize (6);    // We need 6 values
        axeZ.values[0] = bestPose.midPointPose.translation()[0];
        axeZ.values[1] = bestPose.midPointPose.translation()[1];
        axeZ.values[2] = bestPose.midPointPose.translation()[2];
        axeZ.values[3] = bestPose.midPointPose.linear()(0, 2);
        axeZ.values[4] = bestPose.midPointPose.linear()(1, 2);
        axeZ.values[5] = bestPose.midPointPose.linear()(2, 2);

        viewer->addLine(axeZ, objectLabel + "Pose axeZ");      

      }
      else{
        /*std::string cad = "exec rm /ignacio/Desktop/iros_2023/catkin_ws/src/GeoGraspEvo/experiments/a_third"+std::to_string((objNumber-1))+".pcd";
        system(cad.c_str());
        cad = "exec rm /ignacio/Desktop/iros_2023/catkin_ws/src/GeoGraspEvo/experiments/b_obj"+std::to_string((objNumber-1))+".pcd";
        system(cad.c_str());
        cad = "exec rm /ignacio/Desktop/iros_2023/catkin_ws/src/GeoGraspEvo/experiments/b_FinalCandidates"+std::to_string((objNumber-2))+"_0.pcd";
        system(cad.c_str());        
        cad = "exec rm /ignacio/Desktop/iros_2023/catkin_ws/src/GeoGraspEvo/experiments/b_FinalCandidates"+std::to_string((objNumber-2))+"_1.pcd";
        system(cad.c_str());                
        cad = "exec rm /ignacio/Desktop/iros_2023/catkin_ws/src/GeoGraspEvo/experiments/b_tillThumbCandidates"+std::to_string((objNumber-2))+".pcd";    
        system(cad.c_str());*/
      }
    }

    while (!viewer->wasStopped())
      viewer->spinOnce(100);
      
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "cloud_processor");

  viewer->initCameraParameters();
  viewer->addCoordinateSystem(0.1);

  ros::NodeHandle n("~");
  std::string cloudTopic;
  
  n.getParam("topic", cloudTopic);

  ros::Subscriber sub = n.subscribe<sensor_msgs::PointCloud2>(cloudTopic, 1, cloudCallback);

  ros::spin();

  return 0;
}
