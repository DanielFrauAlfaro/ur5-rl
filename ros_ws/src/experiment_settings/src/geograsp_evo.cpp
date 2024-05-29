#include "experiment_settings/Grasp.h"
#include "geograspevo/GeoGraspEvo.h"
#include <iostream>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/convergence_criteria.h>


#include <ros/ros.h>
#include "std_msgs/Int32.h"
#include <tf/tf.h>
#include <xmlrpcpp/XmlRpcValue.h>

ros::Subscriber sub;
bool rgbdImgReceived;
bool perform_ransac;
sensor_msgs::PointCloud2::ConstPtr pcImg;

std::vector<std::vector<float>> removeExceptNumbers(int numberFingers, std::string apertures_vector){

  // std::string cad = apertures_vector.toXml();
  // std::vector<char> deleteChar = {'v','a','l','u','e','<','>','{','}','/'};  
    
  // for (int i=0; i<deleteChar.size(); i++){
  //   cad.erase(remove(cad.begin(), cad.end(), deleteChar[i]), cad.end());
  // }
  
  // std::stringstream ss(cad);
  // std::vector<float> numbers;
  // std::vector<float> partialNumbers;
  // std::vector<std::vector<float>> correctedNumbers;
  // char ch;
  // int tmp;
  // while(ss >> tmp){
  //   numbers.push_back(tmp);
  //   ss >> ch;
  // }
  
  // int numberGroups = numberFingers - 1;
  // int cont=  0;
  // for (int i=0; i<numberGroups; i++){
  //   for(int j=0; j<2;j++){
  //       partialNumbers.push_back(numbers[cont]);
  //       cont = cont + 1;
  //   }
  //   correctedNumbers.push_back(partialNumbers);
  //   partialNumbers.clear();
  // } 

  std::vector<std::vector<float>> correctedNumbers; 
  std::vector<float> partial = {0.0, 0.0};
  
  for(int i=0; i<numberFingers-1; i++)
  {
    correctedNumbers.push_back(partial);
  }


  
  bool is_number= false;
  int row = 0;
  int col = 0;
  int count = 0;
  std::string aux = "";

  for(int i = 0; i<apertures_vector.size(); i++)
  {
    if(apertures_vector[i] != '{' &&
       apertures_vector[i] != '}' &&
       apertures_vector[i] != ',' &&
       apertures_vector[i] != ' ')
    {
      aux += apertures_vector[i];
      is_number = true;
    }

    else if(is_number)
    {
      is_number = false;

      correctedNumbers[col][row] = std::stof(aux);
      aux = "";
      col++;

      if(col == numberFingers-1)
      {
        col = 0;
        row++;
      }
    }
  }
  
  return correctedNumbers;
}

void PCCallback(const sensor_msgs::PointCloud2::ConstPtr& msg){

  if (msg->height != 0){
    pcImg = msg;
    sub.shutdown();
    rgbdImgReceived = true; 
  }
}

template<typename T, typename U>
void extractInliersCloud(const T & inputCloud, const pcl::PointIndices::Ptr & inputCloudInliers, T outputCloud) {
  U extractor;

  extractor.setInputCloud(inputCloud);
  extractor.setIndices(inputCloudInliers);
  extractor.filter(*outputCloud);
}




void processPC(ros::NodeHandle nh, std::string saveFilesPath, int gripTipSize, int numberFingers, int uniqueMobility, int graspsTrack, std::vector<std::vector<float>> apertures, int actualObject){

  // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cloud viewer"));      
  // viewer->initCameraParameters();
  // viewer->addCoordinateSystem(0.1);
  // viewer->setBackgroundColor (0, 0, 0);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*pcImg, *cloud);


  pcl::PointCloud<pcl::PointXYZRGB>::Ptr object (new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::io::loadPCDFile<pcl::PointXYZRGB> ("/daniel/Desktop/evo_pipe/catkin_ws/PCD/sugar_box_original.pcd", *object);
  // pcl::io::loadPCDFile<pcl::PointXYZ> ("/daniel/Desktop/evo_pipe/catkin_ws/PCD/cracker_box_original.pcd", *cloud);


  // Remove NaN values and make it dense
  std::vector<int> nanIndices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, nanIndices);

  

  // Remove background points
  pcl::PassThrough<pcl::PointXYZ> ptFilter;
  ptFilter.setInputCloud(cloud);
  ptFilter.setFilterFieldName("z");
  ptFilter.setFilterLimits(0.0, 0.9);
  ptFilter.filter(*cloud);

  ptFilter.setInputCloud(cloud);
  ptFilter.setFilterFieldName("y");
  ptFilter.setFilterLimits(-0.55, 0.40);
  ptFilter.filter(*cloud);

  ptFilter.setInputCloud(cloud);
  ptFilter.setFilterFieldName("x");
  ptFilter.setFilterLimits(-0.50, 0.30);
  ptFilter.filter(*cloud);
  
  // viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  
  // while (!viewer->wasStopped ()){
  //   viewer->spinOnce (100);
  // }
  
  // viewer->removeAllPointClouds();
  // viewer->removeAllShapes();
  
  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> sacSegmentator;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPlane(new pcl::PointCloud<pcl::PointXYZ>());

  sacSegmentator.setModelType(pcl::SACMODEL_PLANE);
  sacSegmentator.setMethodType(pcl::SAC_RANSAC);
  sacSegmentator.setMaxIterations(100);
  sacSegmentator.setDistanceThreshold(0.02);
  sacSegmentator.setInputCloud(cloud);
  sacSegmentator.segment(*inliers, *coefficients);

  // Remove the planar inliers, extract the rest
  pcl::ExtractIndices<pcl::PointXYZ> indExtractor;
  indExtractor.setInputCloud(cloud);
  indExtractor.setIndices(inliers);
  indExtractor.setNegative(false);

  // Get the points associated with the planar surface
  indExtractor.filter(*cloudPlane);

  // Remove the planar inliers, extract the rest
  indExtractor.setNegative(true);
  indExtractor.filter(*cloud);

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> clusterIndices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ecExtractor;
  ecExtractor.setClusterTolerance(0.04);
  ecExtractor.setMinClusterSize(750);
  ecExtractor.setSearchMethod(tree);
  ecExtractor.setInputCloud(cloud);
  ecExtractor.extract(clusterIndices);
  
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb(cloud, 255, 255, 255);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_red(cloud, 255, 0, 0);
  
  
  // viewer->addPointCloud<pcl::PointXYZ>(cloud, rgb, "Main cloud");
  // viewer->addPointCloud<pcl::PointXYZ>(cloudPlane, rgb_red, "Plane cloud");

  // while (!viewer->wasStopped ()){
  //   viewer->spinOnce (100);
  // }
  
  // viewer->removeAllPointClouds();
  // viewer->removeAllShapes();


  if(perform_ransac)
  {

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>);

    //pcl::copyPointCloud(object_test,my_cloud);
    pcl::copyPointCloud(*cloud,*scene);


    // pcl::visualization::PCLVisualizer viewer_("3D Viewer");
    // viewer_.setBackgroundColor(255, 255, 255);
    // viewer_.addPointCloud(object, "scene_cloud");
    // viewer_.addPointCloud(scene, "transformed_object");
    // viewer_.spin();

    



    // ------------------------ RANSAC -----------------------------
    std::cout<<"\n\nPERFORMING RANSAC\n\n"; // my_cloud --> object
                                            // cloud --> scene

    // Voxel
    std::cout<<"Voxeling ...\n";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_voxel(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_voxel(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(scene);
    sor.setLeafSize(0.001f, 0.001f, 0.001f);
    sor.filter(*scene_voxel);

    sor.setInputCloud(object);
    sor.filter(*object_voxel);

    // Normals
    std::cout<<"Computing Normals ...\n";
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr object_normals(new pcl::PointCloud<pcl::Normal>);
    float voxel_size = 0.01f;

    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud(scene_voxel);
    //ne.setKSearch(30);
    ne.setRadiusSearch(0.03);
    ne.compute(*scene_normals);

    ne.setInputCloud(object_voxel); 
    ne.compute(*object_normals);

    // ISS keypoints
    std::cout<<"Computing ISS keypoints ...\n";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_keypoints(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_keypoints(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::ISSKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZRGB> iss_detector;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_(new pcl::search::KdTree<pcl::PointXYZRGB>);
    iss_detector.setSearchMethod(tree_);
    iss_detector.setSalientRadius(0.01); // Saliency = ultimo eigenvalue. Para ser seleccionado tiene que ser el mayor en vecindad
    iss_detector.setNonMaxRadius(0.01);
    iss_detector.setThreshold21(1);  // Umbrales para los eigen values, es por ratios: e2/e1 y e3/e2
    iss_detector.setThreshold32(1);
    iss_detector.setMinNeighbors(3);
    iss_detector.setNumberOfThreads(8);
    iss_detector.setInputCloud(scene_voxel);
    iss_detector.compute(*scene_keypoints);

    iss_detector.setInputCloud(object_voxel);   // AQUI IBA object_voxel
    iss_detector.compute(*object_keypoints);


    // pcl::visualization::PCLVisualizer viewer_aux_3("3D Viewer");
    // viewer_aux_3.setBackgroundColor(255, 255, 255);
    // viewer_aux_3.addPointCloud(object_keypoints, "scene_cloud");
    // viewer_aux_3.addPointCloud(scene_keypoints, "transformed_object");
    // viewer_aux_3.spin();

    // Normal de los keypoints
    std::cout<<"Computing Normal Keypoints ...\n";
    pcl::PointCloud<pcl::Normal>::Ptr scene_keypoints_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr object_keypoints_normals(new pcl::PointCloud<pcl::Normal>);

    ne.setInputCloud(scene_keypoints);
    //ne.setKSearch(4);
    ne.setRadiusSearch(0.03);
    ne.compute(*scene_keypoints_normals);

    ne.setInputCloud(object_keypoints);
    ne.compute(*object_keypoints_normals);


    // FPFH
    std::cout<<"Computing FPFH features ...\n";
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr object_fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>);

    pcl::FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
    fpfh_est.setInputCloud(scene_keypoints);
    fpfh_est.setInputNormals(scene_keypoints_normals);
    fpfh_est.setNumberOfThreads(8);
    fpfh_est.setSearchMethod(tree_);
    fpfh_est.setRadiusSearch(0.07);
    //fpfh_est.setKSearch(35);
    fpfh_est.compute(*scene_fpfhs);

    fpfh_est.setInputCloud(object_keypoints);
    fpfh_est.setInputNormals(object_keypoints_normals);
    fpfh_est.compute(*object_fpfhs);


    // Apply RANSAC
    std::cout<<"Applying RANSAC ...\n";
    pcl::SampleConsensusPrerejective<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::FPFHSignature33> reg;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_aligned(new pcl::PointCloud<pcl::PointXYZRGB>);
    reg.setInputSource(object_keypoints);
    reg.setSourceFeatures(object_fpfhs);
    reg.setInputTarget(scene_keypoints);
    reg.setTargetFeatures(scene_fpfhs);
    reg.setMaximumIterations(5000000);
    reg.setNumberOfSamples(3);
    reg.setSimilarityThreshold(0.9f);
    reg.setMaxCorrespondenceDistance(0.08f);
    reg.align(*object_aligned);

    // Transformation matrix 
    Eigen::Matrix4f transformation_matrix = reg.getFinalTransformation();

    pcl::transformPointCloud(*object_voxel, *object_aligned, transformation_matrix);
    pcl::transformPointCloud(*object, *object, transformation_matrix);

    // pcl::visualization::PCLVisualizer viewer_aux_2("3D Viewer");
    // viewer_aux_2.setBackgroundColor(255, 255, 255);
    // viewer_aux_2.addPointCloud(scene_voxel, "scene_cloud");
    // viewer_aux_2.addPointCloud(object_aligned, "transformed_object");
    // viewer_aux_2.spin();

    // ICP
    std::cout<<"Applying ICP ...\n";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_aligned_icp(new pcl::PointCloud<pcl::PointXYZRGB>);
      
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setInputSource(object_aligned);
    icp.setInputTarget(scene);
    icp.setMaxCorrespondenceDistance(0.04);
    icp.setEuclideanFitnessEpsilon(1e-09);
    icp.setTransformationEpsilon(1e-09);
    icp.setMaximumIterations(300);

    icp.align(*object_aligned);

    pcl::transformPointCloud(*object, *object, icp.getFinalTransformation());

    std::cout<<"Second ICP ...\n";
    icp.setInputSource(object_aligned);
    icp.setInputTarget(scene);
    icp.setMaxCorrespondenceDistance(0.02);
    icp.setEuclideanFitnessEpsilon(1e-09);
    icp.setTransformationEpsilon(1e-09);
    icp.setMaximumIterations(300);
    icp.align(*object_aligned);
    pcl::transformPointCloud(*object, *object, icp.getFinalTransformation());

    std::cout<<"Third ICP ...\n";
    icp.setInputSource(object_aligned);
    icp.setInputTarget(scene_voxel);
    icp.setMaxCorrespondenceDistance(0.008);
    icp.setEuclideanFitnessEpsilon(1e-09);
    icp.setTransformationEpsilon(1e-09);
    icp.setMaximumIterations(500);
    icp.align(*object_aligned);
    pcl::transformPointCloud(*object, *object, icp.getFinalTransformation());

    

    std::cout << "\n\n\nFitness: " << icp.getFitnessScore() << std::endl <<std::endl<<std::endl;

    pcl::visualization::PCLVisualizer viewer_aux("RANSAC");
    viewer_aux.setBackgroundColor(255, 255, 255);
    viewer_aux.addPointCloud(scene, "scene_cloud");
    viewer_aux.addPointCloud(object, "transformed_object");
    viewer_aux.spin();

    
    pcl::copyPointCloud(*object, *cloud_); 

    pcl::copyPointCloud(*object, *cloud); 


  }
  else{
    pcl::copyPointCloud(*cloud, *cloud_); 
  }

  if (clusterIndices.empty()) {
    ROS_INFO("Cluster indices is empty. Can not keep on with the GeoGraspEvo algorithm.");
  }
  else {
    std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin();
    int objectNumber = 0;
    std::vector<float> rankings;
    std::vector<GraspEvoContacts> grasps;
    GraspEvoContacts bestGrasp;
    GraspEvoPose bestPose;   
    float ranking;

    // Every cluster found is considered a
    // Ideado para trabajar con 1 objeto unicamente. CUIDADO
    for (it = clusterIndices.begin(); it != clusterIndices.end(); ++it) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr objectCloud(new pcl::PointCloud<pcl::PointXYZ>());

      if (objectNumber == 0){ 
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
          objectCloud->points.push_back(cloud->points[*pit]);

        objectCloud->width = objectCloud->points.size();
        objectCloud->height = 1;
        objectCloud->is_dense = !perform_ransac;

        // Create and initialise GeoGraspEvo
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPlaneXYZ(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr objectCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*cloudPlane, *cloudPlaneXYZ);
        pcl::copyPointCloud(*objectCloud, *objectCloudXYZ);

        // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb(cloud,  255,255,255);
        // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_Red(cloud,  255,0,0);
        // pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Viewer FINAL")); 
        // viewer1->setBackgroundColor (0, 0, 0);
        // viewer1->addPointCloud<pcl::PointXYZ>(cloud_, rgb,"Object");
        // viewer1->addPointCloud<pcl::PointXYZ>(cloudPlaneXYZ, rgb_Red,"Points");
        
        // while (!viewer1->wasStopped())
        //   viewer1->spinOnce(100);

        GeoGraspEvo geoGraspPoints;
        geoGraspPoints.setBackgroundCloud(cloudPlaneXYZ);
        if(perform_ransac)
        {
          geoGraspPoints.setObjectCloud(cloud_);
        }
        else
        {
          geoGraspPoints.setObjectCloud(objectCloudXYZ);
        }
        
        geoGraspPoints.setGripTipSize(gripTipSize);
        geoGraspPoints.setApertures(apertures);
        geoGraspPoints.setNumberFingers(numberFingers);
        geoGraspPoints.setUniqueMobility(uniqueMobility); //Move fingers independent
        geoGraspPoints.setGrasps(graspsTrack); // Keep track only of the best

        
        // ############################## SAVE PCDs ##########################################
        // Guardar la nube del objeto
        std::cout<<"\n\nGUARDANDO LA NUBE\n\n";
        pcl::io::savePCDFileASCII (saveFilesPath+std::to_string(actualObject)+"_object.pcd", *cloud_);
        pcl::io::savePCDFileASCII (saveFilesPath+std::to_string(actualObject)+"_plane.pcd", *cloudPlane);
        // ###################################################################################


        std::cout<<"\n\n######## COMPROBACIÓN DE PARÁMETROS ######\n";
        std::cout<<"gripTipSize: "<<gripTipSize<<std::endl;

        std::cout<<"apertures\n";
        for(int p = 0; p<apertures.size(); p++)
        {
          for(int k = 0; k<apertures[p].size(); k++)
          {
            std::cout<<apertures[p][k]<<"   ";
          }       
          std::cout<<"\n";
        }
        
        std::cout<<"numberFingers: "<<numberFingers<<std::endl;
        std::cout<<"uniqueMobility: "<<uniqueMobility<<std::endl;
        std::cout<<"graspsTrack: "<<graspsTrack<<std::endl;
        std::cout<<"\n\n######## COMPROBACIÓN DE PARÁMETROS ######\n";

        
        // Calculate grasping points
        geoGraspPoints.compute();
      
        rankings = geoGraspPoints.getRankings();
        grasps = geoGraspPoints.getGrasps();
      
        std::fstream my_file;
        my_file.open(saveFilesPath+std::to_string(actualObject)+"_info.txt", std::ios::app);
        if (!my_file){
          ROS_INFO("ERROR: file was not created.");
        }
        else{
          my_file<<"ALL GRASPING DATA CALCULATED: \n";
          
          for(int i=0;i<grasps.size();i++){
            my_file<<"\t Point "<<i+1<<"/"<<grasps.size()<<": ";
            for(int j=0;j<grasps[i].graspContactPoints.size();j++){
              my_file<<grasps[i].graspContactPoints[j]<<" ";
            }
            my_file<<"with ranking "<<rankings[i]<<"\n";
          }
        
          my_file.close();
        }  
      
        /*for (int i=0; i<grasps.size(); i++){
          std::cout<<"P"<<i<<": ";
          for (int j=0; j<grasps[i].graspContactPoints.size(); j++){
            std::cout<<grasps[i].graspContactPoints[j]<<" ";
          }
          std::cout<<std::endl;
        }
      
        std::cout<<"RR: ";
        for (int i=0; i<rankings.size(); i++){
          std::cout<<rankings[i]<<" ";
        }
        std::cout<<std::endl;*/
      
        // Extract best pair of points
        bestGrasp = geoGraspPoints.getBestGrasp();
        bestPose = geoGraspPoints.getBestGraspPose();   
        ranking = geoGraspPoints.getBestRanking();     

        //Visualize the result
        pcl::visualization::PointCloudGeometryHandlerXYZ<pcl::PointXYZ> rgb_2(objectCloud);
        pcl::visualization::PointCloudGeometryHandlerXYZ<pcl::PointXYZ> planeRGB_2(cloudPlane);
  
        std::string objectLabel = "";
        std::ostringstream converter;

        converter << objectNumber;
        objectLabel += converter.str();
        objectLabel += "-";

        // viewer->addPointCloud<pcl::PointXYZ>(objectCloud, rgb_2, objectLabel + "Object");
        // viewer->addPointCloud<pcl::PointXYZ>(cloudPlane, planeRGB_2, objectLabel + "Plane");

        // for (int i=0; i<bestGrasp.graspContactPoints.size(); i++){
        //   std::string cad = "Best grasp n." + std::to_string(i);
        //   viewer->addSphere(bestGrasp.graspContactPoints[i], 0.01, 255, 255, 255, objectLabel + cad);
        // }      
      
        // Best of the best cout
      
        my_file.open(saveFilesPath+std::to_string(actualObject)+"_info.txt", std::ios::app);
        if (!my_file){
          ROS_INFO("ERROR: file was not created.");
        }
        else{
          my_file<<"BEST GRASPING DATA FOUND: \n";
        
          my_file<<"\t Point: (";
          for(int i=0;i<bestPose.graspPosePoints.size();i++){
            for(int j=0;j<bestPose.graspPosePoints[i].size();j++){
              my_file<<bestPose.graspPosePoints[i][j];
              if(j!=bestPose.graspPosePoints[i].size()-1){
                my_file<<",";
              }
            }
            if(i==bestPose.graspPosePoints.size()-1){
              my_file<<") ";
            }
            else{
              my_file<<") (";
            }
          }
          my_file<<"with ranking "<<ranking<<"\n";
        
          my_file<<"\t Mid Point: \n";
        
          my_file<<"\t\t Pose: ";
          for(int i=0;i<bestPose.midPointPose.translation().size();i++){
            my_file<<bestPose.midPointPose.translation()[i];
            if(i!=bestPose.midPointPose.translation().size()-1){
              my_file<<", ";
            }
          }
        
          my_file<<"\n";
        
          my_file<<"\t\t Orientation: \n";  
        
          Eigen::Quaternionf q1 (bestPose.midPointPose.rotation());
          tf::Quaternion qtf1(q1.x(),q1.y(),q1.z(),q1.w());
          
          my_file<<"\t\t\t Quaternion ->"<<qtf1.x()<<", "<<qtf1.y()<<", "<<qtf1.z()<<", "<<qtf1.w()<<"\n";   
        
          tf::Matrix3x3 m(qtf1);
          double roll, pitch, yaw;
          m.getRPY(roll, pitch, yaw);
                
          my_file<<"\t\t\t RPY ->"<<roll<<", "<<pitch<<", "<<yaw<<"\n";        
          my_file.close();
        }
      
        /*std::cout<<"Best ranking: "<<ranking<<std::endl;
      
        std::cout<<"Best pose: "<<std::endl;
        std::cout<<"GraspPosePoints"<<std::endl;
        for(int i=0;i<bestPose.graspPosePoints.size();i++){
          std::cout<<bestPose.graspPosePoints[i][0]<<" "<<bestPose.graspPosePoints[i][1]<<" "<<bestPose.graspPosePoints[i][2]<<std::endl;
        }
        std::cout<<"MidPointPose: "<<std::endl;
        Eigen::Quaternionf q1 (bestPose.midPointPose.rotation());
        tf::Quaternion qtf1(q1.x(),q1.y(),q1.z(),q1.w());
        std::cout<<bestPose.midPointPose.translation().x()<<" "<<bestPose.midPointPose.translation().y()<<" "<<bestPose.midPointPose.translation().z()<<" "<<qtf1.x()<<" " <<qtf1.y()<<" "<<qtf1.z()<<" "<<qtf1.w()<<std::endl;*/
      
        std::string message = "Best grasping point: (";
        for(int i=0;i<bestPose.graspPosePoints.size();i++){
          for(int j=0;j<bestPose.graspPosePoints[i].size();j++){
            message+=std::to_string(bestPose.graspPosePoints[i][j]);
            if(j!=bestPose.graspPosePoints[i].size()-1){
              message+=",";
            }
          }
          if(i==bestPose.graspPosePoints.size()-1){
            message+=") ";
          }
          else{
            message+=") (";
          }
        }
        message+="with ranking "+std::to_string(ranking)+".";
      
        ROS_INFO(" ");
        //ROS_INFO(message);
        ROS_INFO("%s\n", message.c_str());
        ROS_INFO(" ");

        pcl::ModelCoefficients axeX;
        axeX.values.resize (6);    // We need 6 values
        axeX.values[0] = bestPose.midPointPose.translation()[0];
        axeX.values[1] = bestPose.midPointPose.translation()[1];
        axeX.values[2] = bestPose.midPointPose.translation()[2];
        axeX.values[3] = bestPose.midPointPose.linear()(0, 0);
        axeX.values[4] = bestPose.midPointPose.linear()(1, 0);
        axeX.values[5] = bestPose.midPointPose.linear()(2, 0);

        // viewer->addLine(axeX, objectLabel + "Pose axeX");

        pcl::ModelCoefficients axeY;
        axeY.values.resize (6);    // We need 6 values
        axeY.values[0] = bestPose.midPointPose.translation()[0];
        axeY.values[1] = bestPose.midPointPose.translation()[1];
        axeY.values[2] = bestPose.midPointPose.translation()[2];
        axeY.values[3] = bestPose.midPointPose.linear()(0, 1);
        axeY.values[4] = bestPose.midPointPose.linear()(1, 1);
        axeY.values[5] = bestPose.midPointPose.linear()(2, 1);

        // viewer->addLine(axeY, objectLabel + "Pose axeY");

        pcl::ModelCoefficients axeZ;
        axeZ.values.resize (6);    // We need 6 values
        axeZ.values[0] = bestPose.midPointPose.translation()[0];
        axeZ.values[1] = bestPose.midPointPose.translation()[1];
        axeZ.values[2] = bestPose.midPointPose.translation()[2];
        axeZ.values[3] = bestPose.midPointPose.linear()(0, 2);
        axeZ.values[4] = bestPose.midPointPose.linear()(1, 2);
        axeZ.values[5] = bestPose.midPointPose.linear()(2, 2);

        // viewer->addLine(axeZ, objectLabel + "Pose axeZ");
      
        objectNumber++;
      }
    }

    // while (!viewer->wasStopped())
    //   viewer->spinOnce(100);
      
    // Save result as pcd. For that load complete cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr PC_color_result(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (int i=0;i<cloud->points.size();i++){
      pcl::PointXYZRGB singlePoint;
      singlePoint.x = cloud->points[i].x;
      singlePoint.y = cloud->points[i].y;
      singlePoint.z = cloud->points[i].z;
      singlePoint.r = 255;
      singlePoint.g = 255;
      singlePoint.b = 255;
      
      PC_color_result->push_back(singlePoint);
    }
    for(int i=0;i<cloudPlane->points.size();i++){
      pcl::PointXYZRGB singlePoint;
      singlePoint.x = cloudPlane->points[i].x;
      singlePoint.y = cloudPlane->points[i].y;
      singlePoint.z = cloudPlane->points[i].z;
      singlePoint.r = 127;
      singlePoint.g = 127;
      singlePoint.b = 127;
      
      PC_color_result->push_back(singlePoint);
    }
    
    PC_color_result->width = PC_color_result->points.size();
    PC_color_result->height = 1;
    PC_color_result->is_dense = true;
    
    // Get grasping points
    std::vector<pcl::PointXYZRGB> graspPoints;
    for(int i=0;i<bestPose.graspPosePoints.size(); i++){
      pcl::PointXYZRGB posePoint;
      posePoint.x = bestPose.graspPosePoints[i][0];
      posePoint.y = bestPose.graspPosePoints[i][1];
      posePoint.z = bestPose.graspPosePoints[i][2];
      
      graspPoints.push_back(posePoint);
    }
    
    // Get neighbours of those points and paint them in red
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    for(int i=0;i<graspPoints.size(); i++){
      pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
      float radius = 0.007;    
      std::vector<int> pointIdxRadiusSearch;
      std::vector<float> pointRadiusSquaredDistance;
      pcl::PointXYZRGB searchPoint(graspPoints[i]);
      
      kdtree.setInputCloud (PC_color_result); 
      kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);

      for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i){
        pcl::PointXYZRGB point;
        point.x = PC_color_result->points[pointIdxRadiusSearch[i]].x;
        point.y = PC_color_result->points[pointIdxRadiusSearch[i]].y;
        point.z = PC_color_result->points[pointIdxRadiusSearch[i]].z;
        point.r = 255;
        point.g = 0;
        point.b = 0;
        cloud_cluster->points.push_back(point);
      }
        
      cloud_cluster->width = cloud_cluster->points.size ();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;
    }
    
    // ###################### SAVE PCDs ###########################################
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr save_PC (new pcl::PointCloud<pcl::PointXYZRGB>);
    // *save_PC = *PC_color_result;
    // *save_PC += *cloud_cluster;
    
    // pcl::io::savePCDFileASCII (saveFilesPath+std::to_string(actualObject)+"_PC_graspPoints.pcd", *save_PC);
    // #############################################################################
    
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(PC_color_result);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> points_rgb(cloud_cluster);
    // pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Viewer")); 
    // viewer1->addPointCloud<pcl::PointXYZRGB>(PC_color_result, rgb,"Object");
    // viewer1->addPointCloud<pcl::PointXYZRGB>(cloud_cluster, points_rgb,"Points");
    
    // while (!viewer1->wasStopped())
    //   viewer1->spinOnce(100);
    
    // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(save_PC); 
    // pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Viewer1")); 
    // viewer1->addPointCloud<pcl::PointXYZRGB>(save_PC, rgb,"Object");
    
    // while (!viewer1->wasStopped())
    //   viewer1->spinOnce(100);

    // Extra code for sending the grasp to the other programs      
    ros::Publisher graspMessage_pub = nh.advertise<experiment_settings::Grasp>("/aurova/grasp/result", 1);
    ros::Time now = ros::Time::now();
    experiment_settings::Grasp graspMessage;
    graspMessage.header.stamp = now;
    graspMessage.header.frame_id = "camera_depth_frame";
    
    experiment_settings::GraspEvoContacts contact;
    sensor_msgs::PointCloud2 points;
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_pcl(new pcl::PointCloud<pcl::PointXYZ>());
    contact.header.stamp = now;
    contact.header.frame_id = "camera_depth_frame";
    for (int j=0; j<grasps[0].graspContactPoints.size(); j++){
          points_pcl->push_back(grasps[0].graspContactPoints[j]);
    }    
    pcl::toROSMsg(*points_pcl, points);
    contact.graspContactPoints = points;
    graspMessage.bestGrasp = contact;            
    
    graspMessage.ranking = ranking;
    
    experiment_settings::GraspEvoPose pose;
    pose.header.stamp = now;
    pose.header.frame_id = "camera_depth_frame";
    std::vector<geometry_msgs::Vector3> posePoints;
    for(int i=0;i<bestPose.graspPosePoints.size(); i++){
      geometry_msgs::Vector3 posePoint;
      posePoint.x = bestPose.graspPosePoints[i][0];
      posePoint.y = bestPose.graspPosePoints[i][1];
      posePoint.z = bestPose.graspPosePoints[i][2];
      
      posePoints.push_back(posePoint);
    }
    pose.graspPosePoints = posePoints;  
      
    geometry_msgs::Pose midposePoint;
    midposePoint.position.x = bestPose.midPointPose.translation().x();
    midposePoint.position.y = bestPose.midPointPose.translation().y();
    midposePoint.position.z = bestPose.midPointPose.translation().z();
    Eigen::Quaternionf q (bestPose.midPointPose.rotation());
    tf::Quaternion qtf(q.x(),q.y(),q.z(),q.w());
    midposePoint.orientation.x = qtf.x();
    midposePoint.orientation.y = qtf.y();
    midposePoint.orientation.z = qtf.z();
    midposePoint.orientation.w = qtf.w();
    pose.midPointPose = midposePoint;
    graspMessage.bestPose = pose;
    
    graspMessage_pub.publish(graspMessage);
    
    int loop = 0;
    while(loop < 2){      
      ros::spinOnce();
      graspMessage_pub.publish(graspMessage);
      ros::Duration(0.5).sleep(); 
      loop = loop + 1;
    }
  }

  std::cout<<"FIN DEL NODO DE GEOGRASP\n\n";
}

int main(int argc, char *argv[]){

  // This must be called before anything else ROS-related
  ros::init(argc, argv, "geograsp_evo");

  // Create a ROS node handle
  ros::NodeHandle nh("~");
  ros::Rate loop_rate(10);
  
  // Get parameters from launch file
  std::string saveFilesPath;
  int gripTipSize, numberFingers, uniqueMobility, graspsTrack;
  
  nh.getParam("save_files_path", saveFilesPath);  
  nh.getParam("geograspEvo_grip_tip_size", gripTipSize);
  nh.getParam("geograspEvo_set_number_fingers", numberFingers);
  nh.getParam("geograspEvo_set_unique_mobility", uniqueMobility);
  nh.getParam("geograspEvo_set_grasps_track", graspsTrack);
  nh.getParam("geograspEvo_ransac", perform_ransac);

  std::string aperturesVector;
  nh.getParam("geograspEvo_set_apertures", aperturesVector);
  std::vector<std::vector<float>> apertures = removeExceptNumbers(numberFingers, aperturesVector);

  int actualObject = 0;
   
  while(ros::ok()){  
    
    // Wait for image to be processed
    rgbdImgReceived = false;
    sub = nh.subscribe("/aurova/image/RGBD", 1, PCCallback);
    while (rgbdImgReceived == false){
      ros::spinOnce();  
    }    

    // Get grasping points
    processPC(nh, saveFilesPath, gripTipSize, numberFingers, uniqueMobility, graspsTrack, apertures, actualObject);
  
    actualObject = actualObject + 1;
    
  }
      
  return 0;
}
