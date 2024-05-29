// catkin_make -DCMAKE_BUILD_TYPE=Release
#include "geograspevo/GeoGraspEvo.h"

const float GeoGraspEvo::kGraspPlaneApprox = 0.007;
const float GeoGraspEvo::kCloudNormalRadius = 0.03;

GeoGraspEvo::GeoGraspEvo() :
    backgroundCloud(new pcl::PointCloud<pcl::PointXYZ>),
    objectCloud(new pcl::PointCloud<pcl::PointXYZ>),
    objectNormalCloud(new pcl::PointCloud<pcl::PointNormal>),
    graspPlaneCloud(new pcl::PointCloud<pcl::PointNormal>),
    backgroundPlaneCoeff(new pcl::ModelCoefficients),
    objectAxisCoeff(new pcl::ModelCoefficients),
    graspPlaneCoeff(new pcl::ModelCoefficients) 
    { 
  numberBestGrasps = 1;
  gripTipSize = 30;
  apertures = {{0.0010,0.0030},{0.0015,0.0020}};
  numFingers = 2;
}

GeoGraspEvo::~GeoGraspEvo() {
  graspingPoints.clear();
  rankings.clear();
  pointsDistance.clear();
}

void GeoGraspEvo::setBackgroundCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  backgroundCloud = cloud;
}

void GeoGraspEvo::setBackgroundPlaneCoeff(
    const pcl::ModelCoefficients & coefficients) {
  *backgroundPlaneCoeff = coefficients;
}

void GeoGraspEvo::setObjectCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  objectCloud = cloud;
}

void GeoGraspEvo::setGrasps(const int & grasps) {
  numberBestGrasps = grasps;
}

void GeoGraspEvo::setApertures(const std::vector<std::vector<float>> & open) {

  apertures.clear();
  
  for (int i=0; i<open.size(); i++){

    if (open[i][0] < open[i][1]){
      apertures.push_back({open[i][0], open[i][1]});
    }
    else{
      apertures.push_back({open[i][1], open[i][0]});
    }
  }
}

void GeoGraspEvo::setNumberFingers(const int & fingers){
  numFingers = fingers;
}

void GeoGraspEvo::setGripTipSize(const int & size) {
  gripTipSize = size;
}

void GeoGraspEvo::setUniqueMobility(const bool & mov){
  uniqueMobility = mov;
}

GraspEvoContacts GeoGraspEvo::getBestGrasp() const {
  GraspEvoContacts grasp;

  if (this->bestGraspingPoints.empty())
    std::cout << "No grasp configurations were found during the computation\n";
  else {
  
    for (int i=0; i<this->bestGraspingPoints[0].graspContactPoints.size(); i++){
      grasp.graspContactPoints.push_back(this->bestGraspingPoints[0].graspContactPoints[i]);
    }

  }

  return grasp;
}

std::vector<GraspEvoContacts> GeoGraspEvo::getGrasps() const {
  return this->bestGraspingPoints;
}

GraspEvoPose GeoGraspEvo::getBestGraspPose() const {
  GraspEvoPose grasp;

  if (this->bestGraspingPoints.empty())
    std::cout << "No grasp configurations were found during the computation\n";
  else {
  
    for (int i=0; i<this->bestGraspingPoints[0].graspContactPoints.size(); i++){
      Eigen::Vector3f point(this->bestGraspingPoints[0].graspContactPoints[i].x, this->bestGraspingPoints[0].graspContactPoints[i].y, this->bestGraspingPoints[0].graspContactPoints[i].z);
      grasp.graspPosePoints.push_back(point);
    }
    std::cout<<"Mejor pto agarre final: ";    
    for (int i=0; i<grasp.graspPosePoints.size(); i++){
      std::cout<<"(";
      for (int j=0; j<3; j++){
        std::cout<<grasp.graspPosePoints[i][j]<<" ";
      }
      std::cout<<")"<<" ";
    }
    std::cout<<std::endl;
    
    Eigen::Vector3f midPoint;
    Eigen::Vector3f axeZ;
    switch (this->numFingers){
      case 2:
        midPoint = (grasp.graspPosePoints[0] + grasp.graspPosePoints[1]) / 2.0;
        axeZ = grasp.graspPosePoints[0] - grasp.graspPosePoints[1];        
        break;
      case 3:
        midPoint = (grasp.graspPosePoints[0] + ((grasp.graspPosePoints[1] + grasp.graspPosePoints[2]) / 2.0)) / 2.0;
        axeZ = grasp.graspPosePoints[0] - (grasp.graspPosePoints[1] + grasp.graspPosePoints[2]) / 2;
        break;
      case 4:
        midPoint = (grasp.graspPosePoints[0] + ((grasp.graspPosePoints[1] + grasp.graspPosePoints[2] + grasp.graspPosePoints[3]) / 3.0)) / 2.0;
        axeZ = grasp.graspPosePoints[0] - (grasp.graspPosePoints[1] + grasp.graspPosePoints[2] + grasp.graspPosePoints[3]) / 3;        
        break;
      case 5:
        midPoint = (grasp.graspPosePoints[0] + ((grasp.graspPosePoints[1] + grasp.graspPosePoints[2] + grasp.graspPosePoints[3] + grasp.graspPosePoints[4]) / 4.0)) / 2.0;
        axeZ = grasp.graspPosePoints[0] - (grasp.graspPosePoints[1] + grasp.graspPosePoints[2] + grasp.graspPosePoints[3] + grasp.graspPosePoints[4]) / 4;      
        break;
      default:
        std::cout<<"INVALID NUMBER OF FINGERS"<<std::endl;
    }

    Eigen::Vector3f objAxisVector(this->objectAxisCoeff->values[3],
                                  this->objectAxisCoeff->values[4],
                                  this->objectAxisCoeff->values[5]);
    Eigen::Vector3f axeX, axeY;
  
    axeX = axeZ.cross(objAxisVector);
    axeY = axeZ.cross(axeX);
   
    axeX.normalize();
    axeY.normalize();
    axeZ.normalize();    

    Eigen::Matrix3f midPointRotation;
    midPointRotation << axeX[0], axeY[0], axeZ[0], 
                        axeX[1], axeY[1], axeZ[1],
                        axeX[2], axeY[2], axeZ[2];
    grasp.midPointPose.translation() = midPoint;
    grasp.midPointPose.linear() = midPointRotation;
  }

  return grasp;
}

float GeoGraspEvo::getBestRanking() const {
  float ranking;

  if (this->bestGraspingPoints.empty())
    std::cout << "No grasp configurations were found during the computation\n";
  else
    ranking = this->rankings[0];

  return ranking;
}

std::vector<float> GeoGraspEvo::getRankings() const {
  return this->rankings;
}

pcl::ModelCoefficients GeoGraspEvo::getObjectAxisCoeff() const {
  return *objectAxisCoeff;
}

pcl::ModelCoefficients GeoGraspEvo::getBackgroundPlaneCoeff() const {
  return *backgroundPlaneCoeff;
}

void GeoGraspEvo::compute() {
  // Some ratios dependant of the gripper
  this->voxelRadius = (this->gripTipSize / 1000.0)/2.0;
  this->graspRadius = 2.0 * this->gripTipSize / 1000.0;

  std::cout << "Grasp radius: " << this->graspRadius << "\n"
            << "Voxel radius: " << this->voxelRadius << "\n";

  std::cout << "Loaded " << this->objectCloud->width * this->objectCloud->height 
            << " data points from object cloud\n";

  // Background plane
  if (backgroundPlaneCoeff->values.size() == 0)
    computeCloudPlane(this->backgroundCloud, this->backgroundPlaneCoeff);

  // Cloud plane filtering
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr originalCloud(new pcl::PointCloud<pcl::PointXYZ>);
  float outliersThreshold = 1.0;
  int meanNeighbours = 50;

  filterOutliersFromCloud(this->objectCloud, meanNeighbours, outliersThreshold,
    originalCloud);

  // Get points and their normals after applying voxel
  pcl::PointCloud<pcl::PointXYZ>::Ptr voxelCloud(new pcl::PointCloud<pcl::PointXYZ>);
  voxelizeCloud<pcl::PointCloud<pcl::PointXYZ>::Ptr, 
                pcl::VoxelGrid<pcl::PointXYZ> >(originalCloud, this->voxelRadius, voxelCloud);
                
  // Object normals
  // This could be optimised passing a voxelized cloud instead
  computeCloudNormals(voxelCloud, this->kCloudNormalRadius, this->objectNormalCloud);
                
  cloud = voxelCloud;

  // Aproximate the object's main axis and centroid
  computeCloudGeometry(voxelCloud, this->objectAxisCoeff, this->objectCentroidPoint);
  
  // Find camera orientation wrt background plane
  Eigen::Vector3f backNormalVector(this->backgroundPlaneCoeff->values[0],
                                   this->backgroundPlaneCoeff->values[1],
                                   this->backgroundPlaneCoeff->values[2]);
  Eigen::Vector3f worldZVector(0, 0, 1);

  float backWorldZAngleCos = std::abs((backNormalVector.dot(worldZVector)) / 
    (backNormalVector.norm() * worldZVector.norm()));
    
  bool chooseX = false;
  bool topView = false;
  
  this->chosenPoints.clear();
  this->chosenPointsNormal.clear();

  // Compute initial points accordingly
  if (backWorldZAngleCos > 0.9) {
    std::cout << "Camera in top view\n";
    findInitialPointsInTopView(&chooseX, &topView);
  }
  else {
    std::cout << "Camera in side view\n";
    findInitialPointsInSideView(&chooseX);
  }

  // A grasp plane must have been built, otherwise stop!
  if (graspPlaneCloud->points.size() == 0) {
    std::cout << "ERROR: GRASP PLANE IS EMPTY. Interrumpting.\n";

    return;
  }

#ifdef _DEBUG
  std::cout<<"Mejor pto de agarre (inicial): ";
  for (int i=0; i<this->graspingPoints.size(); i++){
    std::cout<<"( ";
    for (int j=0; j<this->graspingPoints[i].getVector3fMap().size(); j++){
      std::cout<<this->graspingPoints[i].getVector3fMap()[j]<<" ";
    }
    std::cout<<")";
  }
  std::cout<<std::endl;
#endif
  
  Eigen::Vector3f new_diff = this->graspingPoints[0].getVector3fMap() - this->graspingPoints[1].getVector3fMap();
  float objWidth = new_diff.norm();

  if (this->graspRadius * 2.0 >= objWidth) {
    this->graspRadius = objWidth * 0.7 / 2.0;
	std::cout << "Grasp radius adjusted: " << this->graspRadius <<std::endl;
  }

  // Grasping areas
  pcl::PointCloud<pcl::PointXYZ>::Ptr candidatePoints (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointNormal>::Ptr candidatePointsNormal (new pcl::PointCloud<pcl::PointNormal>);
  
  this->chosenPoints.clear();
  this->chosenPointsNormal.clear();

  getClosestPointsByRadius(this->graspingPoints[0], this->graspRadius, cloud, 
                           this->objectNormalCloud, candidatePoints, candidatePointsNormal);
                           
  this->chosenPoints.push_back(candidatePoints);
  this->chosenPointsNormal.push_back(candidatePointsNormal);
  
  getClosestPointsByRadius(this->graspingPoints[1], this->graspRadius, cloud, 
                           this->objectNormalCloud, candidatePoints, candidatePointsNormal);
                           
  this->chosenPoints.push_back(candidatePoints);
  this->chosenPointsNormal.push_back(candidatePointsNormal);     
  
  //drawPossiblePoints(originalCloud, 0);    

  // Refine those areas
  pcl::PointCloud<pcl::PointNormal>::Ptr candidatePointsVoxel(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr candidatePointsVoxel1(new pcl::PointCloud<pcl::PointNormal>);

#ifdef _DEBUG    
  std::cout<<"Puntos antes ("<<this->chosenPointsNormal[0]->size()<<","<<this->chosenPointsNormal[1]->size()<<")";
#endif  
  
  this->chosenPointsNormalVoxel.clear();
  
  voxelizeCloud<pcl::PointCloud<pcl::PointNormal>::Ptr, 
                pcl::VoxelGrid<pcl::PointNormal> >(this->chosenPointsNormal[0], this->voxelRadius, candidatePointsVoxel);
                

  this->chosenPointsNormalVoxel.push_back(candidatePointsVoxel);

  voxelizeCloud<pcl::PointCloud<pcl::PointNormal>::Ptr, 
                pcl::VoxelGrid<pcl::PointNormal> >(this->chosenPointsNormal[1], this->voxelRadius, candidatePointsVoxel1);

  this->chosenPointsNormalVoxel.push_back(candidatePointsVoxel1); 
  
#ifdef _DEBUG  
  std::cout<<" Después ("<<this->chosenPointsNormalVoxel[0]->size()<<","<<this->chosenPointsNormalVoxel[1]->size()<<")"<<std::endl;
#endif  
  
//  this->chosenPointsNormalVoxel.push_back(this->chosenPointsNormal[0]);
//  this->chosenPointsNormalVoxel.push_back(this->chosenPointsNormal[1]);
  
        
  //drawPossiblePoints(originalCloud, 1);                      

  // Best ranking
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  
  getBestGraspingPoints(cloud, this->graspPlaneCoeff, this->objectCentroidPoint, this->numberBestGrasps, 
                        this->bestGraspingPoints, this->rankings, chooseX, topView);
           
  // Cambio P1 y P2 de orden y vuelvo a llamar a la funcion nuevamente             
  pcl::PointCloud<pcl::PointNormal>::Ptr aux (new pcl::PointCloud<pcl::PointNormal>);
  aux = this->chosenPointsNormalVoxel[0];
  this->chosenPointsNormalVoxel[0] = this->chosenPointsNormalVoxel[1];
  this->chosenPointsNormalVoxel[1] = aux;
  
  getBestGraspingPoints(cloud, this->graspPlaneCoeff, this->objectCentroidPoint, this->numberBestGrasps, 
                        this->bestGraspingPoints, this->rankings, chooseX, topView);
    
	
  this->rankings.resize(this->numberBestGrasps);
  this->bestGraspingPoints.resize(this->numberBestGrasps);
  
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout<<"Elapsed time: "<<elapsed_seconds.count()<<"s"<<std::endl;
  
  for (size_t i = 0; i < this->rankings.size(); ++i) {
    std::cout << "Grasp Configuration Rank: " << this->rankings[i] <<std::endl;
  }

  std::cout << "===============================\n";
  
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// - - - - - - - - - - - - - - - AUXILIARY FUNCTIONS - - - - - - - - - - - - - - -
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void GeoGraspEvo::computeCloudPlane(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & inputCloud,
    pcl::ModelCoefficients::Ptr backPlaneCoeff) {
  pcl::PointIndices::Ptr cloudPlaneInliers(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;

  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);
  seg.setInputCloud(inputCloud);
  seg.segment(*cloudPlaneInliers, *backPlaneCoeff);
}

void GeoGraspEvo::filterOutliersFromCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & inputCloud,
    const int & meanNeighbours, const float & distanceThreshold,
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud) {
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sorFilter;

  sorFilter.setInputCloud(inputCloud);
  sorFilter.setMeanK(meanNeighbours);
  sorFilter.setStddevMulThresh(distanceThreshold);
  sorFilter.filter(*outputCloud);
}

void GeoGraspEvo::computeCloudNormals(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & inputCloud,
    const float & searchRadius,
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudNormals) {

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::NormalEstimationOMP<pcl::PointXYZ,pcl::PointNormal> ne;

  ne.setInputCloud(inputCloud);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(searchRadius);
  ne.compute(*cloudNormals);

  // NormalEstimation only fills with normals the output
  for (size_t i = 0; i < cloudNormals->points.size(); ++i) {
    cloudNormals->points[i].x = inputCloud->points[i].x;
    cloudNormals->points[i].y = inputCloud->points[i].y;
    cloudNormals->points[i].z = inputCloud->points[i].z;
  }
}

void GeoGraspEvo::computeCloudGeometry(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & inputCloud,
    pcl::ModelCoefficients::Ptr objAxisCoeff, pcl::PointXYZ & objCenterMass) {
    
  Eigen::Vector3f massCenter, majorVector, middleVector, minorVector;
  pcl::MomentOfInertiaEstimation<pcl::PointXYZ> featureExtractor;
  featureExtractor.setInputCloud(inputCloud);
  featureExtractor.compute();

  featureExtractor.getEigenVectors(majorVector, middleVector, minorVector);
  featureExtractor.getMassCenter(massCenter);
  
  objAxisCoeff->values.resize(6);
  objAxisCoeff->values[0] = massCenter[0];
  objAxisCoeff->values[1] = massCenter[1];
  objAxisCoeff->values[2] = massCenter[2];
  objAxisCoeff->values[3] = majorVector[0];
  objAxisCoeff->values[4] = majorVector[1];
  objAxisCoeff->values[5] = majorVector[2];
  

  objCenterMass.x = massCenter[0];
  objCenterMass.y = massCenter[1];
  objCenterMass.z = massCenter[2];
  
  float min_dist = 0.0;
  float max_dist = 0.0;
    
  for (int i=0;i<inputCloud->points.size();i++){
    float x = inputCloud->points[i].x;  
    float y = inputCloud->points[i].y;  
    float z = inputCloud->points[i].z;
  }
}

template<typename T, typename U>
void GeoGraspEvo::extractInliersCloud(const T & inputCloud,
    const pcl::PointIndices::Ptr & inputCloudInliers, T outputCloud) {
  U extractor;

  extractor.setInputCloud(inputCloud);
  extractor.setIndices(inputCloudInliers);
  extractor.filter(*outputCloud);
}

void GeoGraspEvo::findInitialPointsInSideView(bool *chooseX) {

  // Background plane normal and object main axis angle
  Eigen::Vector3f objAxisVector(this->objectAxisCoeff->values[3],
                                this->objectAxisCoeff->values[4],
                                this->objectAxisCoeff->values[5]);
  Eigen::Vector3f backNormalVector(this->backgroundPlaneCoeff->values[0],
                                   this->backgroundPlaneCoeff->values[1],
                                   this->backgroundPlaneCoeff->values[2]);
  Eigen::Vector3f worldZVector(0, 0, 1);
  
  /*pcl::PointXYZ a(1,0,0);
  pcl::PointXYZ b(0,1,0);
  pcl::PointXYZ c(0,0,1);
  pcl::PointXYZ d(0,0,0);
  
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cloud viewer"));
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb(this->objectCloud, 255, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ>(this->objectCloud, rgb, "Object");
  viewer->addSphere(d, 0.01, 255, 255, 255, "Centroid");
  viewer->addSphere(c, 0.01, 0, 0, 255, "Z");
  viewer->addSphere(a, 0.01, 255, 0, 0, "objectAxisCoeff");
  viewer->addSphere(b, 0.01, 0, 255, 0, "backgroundPlaneCoeff");
  
  while (!viewer->wasStopped())
    viewer->spinOnce(100);*/

  float objGraspNormalAngleCos = std::abs((objAxisVector.dot(backNormalVector)) / 
    (objAxisVector.norm() * backNormalVector.norm()));
  float objWorldZAngleCos = std::abs((objAxisVector.dot(worldZVector)) / 
    (objAxisVector.norm() * worldZVector.norm()));

  std::cout << "Object angle cosine to background normal: " << objGraspNormalAngleCos << "\n";
  std::cout << "Object angle cosine to world Z: " << objWorldZAngleCos << "\n";

  // Grasping plane and initial points
  float planesAngleThreshold = 0.55, worldZAngleThreshold = 0.45;
  bool oppositeAxisVector = false;

  // Object's axis is reversed for consistency
  // Standing object but vector pointing down
  if (objGraspNormalAngleCos > (1.0 - planesAngleThreshold) && 
    this->objectAxisCoeff->values[4] > 0.0)
        oppositeAxisVector = true;
  else if (objGraspNormalAngleCos < planesAngleThreshold) { // Laying objects
    // Laying in the X direction but vector point left
    if (objWorldZAngleCos <= 0.1 && this->objectAxisCoeff->values[3] > 0.0)
        oppositeAxisVector = true;
    // Laying and pointing forward-right but vector points backwards-left
    else if (objWorldZAngleCos > 0.1 && this->objectAxisCoeff->values[3] < 0.0 &&
      this->objectAxisCoeff->values[4] > 0.0 && this->objectAxisCoeff->values[5] < 0.0)
        oppositeAxisVector = true;
    // Laying and pointing forward-left but vector points backwards-right
    else if (objWorldZAngleCos > 0.1 && this->objectAxisCoeff->values[3] > 0.0 &&
      this->objectAxisCoeff->values[4] > 0.0 && this->objectAxisCoeff->values[5] < 0.0)
        oppositeAxisVector = true;
  }

  if (oppositeAxisVector) {
      this->objectAxisCoeff->values[3] *= -1.0;
      this->objectAxisCoeff->values[4] *= -1.0;
      this->objectAxisCoeff->values[5] *= -1.0;

      objAxisVector = -objAxisVector;
  }

  // Grasping plane and initial points
  size_t firstPointIndex = -1, secondPointIndex = -1, thirdPointIndex = -1, forthPointIndex = -1;

  buildGraspingPlane(this->objectCentroidPoint, objAxisVector, 
    this->kGraspPlaneApprox, this->objectNormalCloud, this->graspPlaneCoeff, 
    this->graspPlaneCloud);

  // Laying object in the X axis
  if (objGraspNormalAngleCos < planesAngleThreshold 
    && objWorldZAngleCos < worldZAngleThreshold) {
    std::cout << "It is a laying object in the X axis\n";

    float minZ = std::numeric_limits<float>::max();
    float maxZ = -std::numeric_limits<float>::max();
    *chooseX = false;

    for (size_t i = 0; i < this->graspPlaneCloud->points.size(); ++i) {
      if (this->graspPlaneCloud->points[i].z < minZ) {
        minZ = this->graspPlaneCloud->points[i].z;
        firstPointIndex = i;
      }

      if (this->graspPlaneCloud->points[i].z > maxZ) {
        maxZ = this->graspPlaneCloud->points[i].z;
        secondPointIndex = i;
      }
    }
  }
  else { // Standing objects and laying ones in the Z axis
    std::cout << "It is a laying object in the Z axis or it is standing\n";

    float minX = std::numeric_limits<float>::max();
    float maxX = -std::numeric_limits<float>::max();
    *chooseX = true;

    for (size_t i = 0; i < this->graspPlaneCloud->points.size(); ++i) {
      if (this->graspPlaneCloud->points[i].x < minX) {
        minX = this->graspPlaneCloud->points[i].x;
        firstPointIndex = i;
      }

      if (this->graspPlaneCloud->points[i].x > maxX) {
        maxX = this->graspPlaneCloud->points[i].x;
        secondPointIndex = i;
      }
    }    
  }
  
  this->graspingPoints.clear();

  if (graspPlaneCloud->points.size() != 0) {
    pcl::PointNormal punto(this->graspPlaneCloud->points[firstPointIndex]);
    this->graspingPoints.push_back(punto);
    pcl::PointNormal punto2(this->graspPlaneCloud->points[secondPointIndex]);
    this->graspingPoints.push_back(punto2); 
  }
}

void GeoGraspEvo::findInitialPointsInTopView(bool *chooseX, bool *topView) {

  // Background plane normal and object main axis angle
  Eigen::Vector3f objAxisVector(this->objectAxisCoeff->values[3],
                                this->objectAxisCoeff->values[4],
                                this->objectAxisCoeff->values[5]);
  Eigen::Vector3f worldXVector(1, 0, 0);
  Eigen::Vector3f worldZVector(0, 0, 1);

  float objWorldXAngleCos = std::abs((objAxisVector.dot(worldXVector)) / 
    (objAxisVector.norm() * worldXVector.norm()));
  float objWorldZAngleCos = std::abs((objAxisVector.dot(worldZVector)) / 
    (objAxisVector.norm() * worldZVector.norm()));

  std::cout << "Object angle cosine to world X: " << objWorldXAngleCos << "\n";
  std::cout << "Object angle cosine to world Z: " << objWorldZAngleCos << "\n";

  float worldXAngleThreshold = 0.5, worldZAngleThreshold = 0.25;
  bool reverseAxisVector = false;

  // Object's axis is reversed for consistency
  if (objWorldZAngleCos < worldZAngleThreshold) { // Laying objects
    // Object in the X axis but vector pointing right
    if (objWorldXAngleCos > worldXAngleThreshold && this->objectAxisCoeff->values[3] > 0)
      reverseAxisVector = true;
    // Object in the Y axis but vector pointing forward from the robot
    else if (objWorldXAngleCos < worldXAngleThreshold && this->objectAxisCoeff->values[4] < 0)
      reverseAxisVector = true;
  }
  // Standing object but vector pointing down with the Z axis
  else if (objWorldZAngleCos > worldZAngleThreshold && this->objectAxisCoeff->values[5] > 0)
    reverseAxisVector = true;

  if (reverseAxisVector) {
      this->objectAxisCoeff->values[3] *= -1.0;
      this->objectAxisCoeff->values[4] *= -1.0;
      this->objectAxisCoeff->values[5] *= -1.0;

      objAxisVector = -objAxisVector;
  }

  // Grasping plane and initial points
  size_t firstPointIndex = -1, secondPointIndex = -1, thirdPointIndex = -1, forthPointIndex = -1, fifthPointIndex = -1;

  buildGraspingPlane(this->objectCentroidPoint, objAxisVector, 
    this->kGraspPlaneApprox, this->objectNormalCloud, this->graspPlaneCoeff, 
    this->graspPlaneCloud);
  
  // Object in the X axis
  if (objWorldXAngleCos > worldXAngleThreshold) {
    std::cout << "It is oriented with the X axis\n";

    float minY = std::numeric_limits<float>::max();
    float maxY = -std::numeric_limits<float>::max();
    *chooseX = false;
    *topView = true;

    for (size_t i = 0; i < this->graspPlaneCloud->points.size(); ++i) {
      if (this->graspPlaneCloud->points[i].y < minY) {
        minY = this->graspPlaneCloud->points[i].y;
        firstPointIndex = i;
      }

      if (this->graspPlaneCloud->points[i].y > maxY) {
        maxY = this->graspPlaneCloud->points[i].y;
        secondPointIndex = i;
      }
    }
  }
  else { // Object in the Y axis
    std::cout << "It is oriented with the Y axis\n";

    float minX = std::numeric_limits<float>::max();
    float maxX = -std::numeric_limits<float>::max();
    *chooseX = true;
    *topView = false;

    for (size_t i = 0; i < this->graspPlaneCloud->points.size(); ++i) {
      if (this->graspPlaneCloud->points[i].x < minX) {
        minX = this->graspPlaneCloud->points[i].x;
        firstPointIndex = i;
      }

      if (this->graspPlaneCloud->points[i].x > maxX) {
        maxX = this->graspPlaneCloud->points[i].x;
        secondPointIndex = i;
      }
    }
  }

  if (graspPlaneCloud->points.size() != 0) {
    pcl::PointNormal punto(this->graspPlaneCloud->points[firstPointIndex]);
    this->graspingPoints.push_back(punto);
    pcl::PointNormal punto2(this->graspPlaneCloud->points[secondPointIndex]);
    this->graspingPoints.push_back(punto2);
  }
}

void GeoGraspEvo::drawPossiblePoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, bool voxel){

  pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Chosen Points"));
  pcl::PointCloud<pcl::PointXYZ> p3;
    
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color0(cloud, 124, 0, 0);  
      
  p3.push_back(pcl::PointXYZ(this->graspingPoints[0].getVector3fMap()[0], this->graspingPoints[0].getVector3fMap()[1], this->graspingPoints[0].getVector3fMap()[2]));
  p3.push_back(pcl::PointXYZ(this->graspingPoints[1].getVector3fMap()[0], this->graspingPoints[1].getVector3fMap()[1], this->graspingPoints[1].getVector3fMap()[2]));
  
  viewer1->addCoordinateSystem (0.1);  
  viewer1->addPointCloud<pcl::PointXYZ>(cloud, single_color0, "Global PC");
  
  viewer1->addSphere(p3[0], 0.001, 0, 0, 255, "First best grasp point");                  
  viewer1->addSphere(p3[1], 0.001, 0, 255, 255, "Second best grasp point"); 

  if (voxel == 0){
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(this->chosenPoints[0], 0, 255, 0);  
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(this->chosenPoints[1], 0, 0, 255); 
    
    viewer1->addPointCloud<pcl::PointXYZ>(this->chosenPoints[0], single_color1, "Chosen points 1");
    viewer1->addPointCloud<pcl::PointXYZ>(this->chosenPoints[1], single_color2, "Chosen points 2");
  }
  else{
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color1(this->chosenPointsNormalVoxel[0], 255, 255, 255);  
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color2(this->chosenPointsNormalVoxel[1], 255, 0, 255);
    
    viewer1->addPointCloud<pcl::PointNormal>(this->chosenPointsNormalVoxel[0], single_color1, "Chosen points 1");
    viewer1->addPointCloud<pcl::PointNormal>(this->chosenPointsNormalVoxel[1], single_color2, "Chosen points 2");
  }
    
  while (!viewer1->wasStopped())
    viewer1->spinOnce(100);  
}

void GeoGraspEvo::buildGraspingPlane(const pcl::PointXYZ & planePoint,
    const Eigen::Vector3f & planeNormalVector, const float & distanceThreshold,
    const pcl::PointCloud<pcl::PointNormal>::Ptr & inputCloud,
    pcl::ModelCoefficients::Ptr graspPlaneCoeff,
    pcl::PointCloud<pcl::PointNormal>::Ptr graspPlaneCloud) {
    
  // Grasping plane model
  Eigen::Vector3f planePointVector(planePoint.x, planePoint.y, planePoint.z);
  Eigen::Hyperplane<float,3> graspHyperplane(planeNormalVector, planePointVector);

  graspPlaneCoeff->values.resize(4);
  graspPlaneCoeff->values[0] = graspHyperplane.coeffs()[0];
  graspPlaneCoeff->values[1] = graspHyperplane.coeffs()[1];
  graspPlaneCoeff->values[2] = graspHyperplane.coeffs()[2];
  graspPlaneCoeff->values[3] = graspHyperplane.coeffs()[3];

  // Grasping plane cloud
  Eigen::Vector4f graspPlaneVector(graspHyperplane.coeffs()[0],
                                   graspHyperplane.coeffs()[1],
                                   graspHyperplane.coeffs()[2],
                                   graspHyperplane.coeffs()[3]);
  pcl::SampleConsensusModelPlane<pcl::PointNormal>::Ptr planeSAC(new pcl::SampleConsensusModelPlane<pcl::PointNormal>(inputCloud));
  pcl::PointIndices::Ptr graspPlaneIndices(new pcl::PointIndices);

  // TODO: SOMETIMES IT DOESN'T FIND POINTS NEAR THE GRASP PLANE
  planeSAC->selectWithinDistance(graspPlaneVector, distanceThreshold, 
    graspPlaneIndices->indices);
  extractInliersCloud<pcl::PointCloud<pcl::PointNormal>::Ptr, 
                      pcl::ExtractIndices<pcl::PointNormal> >(inputCloud,
                        graspPlaneIndices, graspPlaneCloud);
}

void GeoGraspEvo::getClosestPointsByRadius(const pcl::PointNormal & point,
    const float & radius, const pcl::PointCloud<pcl::PointXYZ>::Ptr & inputCloud,
    const pcl::PointCloud<pcl::PointNormal>::Ptr & inputNormalCloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr & chosenPoints, 
    pcl::PointCloud<pcl::PointNormal>::Ptr & chosenPointsNormal) {
    
  pcl::search::KdTree<pcl::PointNormal>::Ptr treeSearch(new pcl::search::KdTree<pcl::PointNormal>());
  pcl::PointIndices::Ptr pointsIndex(new pcl::PointIndices);
  std::vector<float> pointsSquaredDistance;
  pcl::PointCloud<pcl::PointXYZ>::Ptr preChosenPoints(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointNormal>::Ptr preChosenPointsNormal(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr chosenPointsSphere(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointNormal>::Ptr chosenPointsSphereNormal(new pcl::PointCloud<pcl::PointNormal>);

  treeSearch->setInputCloud(inputNormalCloud);

  if (treeSearch->radiusSearch(point, radius, pointsIndex->indices, 
      pointsSquaredDistance)) {
    extractInliersCloud<pcl::PointCloud<pcl::PointXYZ>::Ptr,
                        pcl::ExtractIndices<pcl::PointXYZ> >(inputCloud,
                          pointsIndex, preChosenPoints);
    extractInliersCloud<pcl::PointCloud<pcl::PointNormal>::Ptr,
                        pcl::ExtractIndices<pcl::PointNormal> >(inputNormalCloud, 
                          pointsIndex, preChosenPointsNormal);
  }
  
  for( int j = 0; j < preChosenPoints->size(); j++){
      
    chosenPointsSphere->push_back(preChosenPoints->points[j]);
    chosenPointsSphereNormal->push_back(preChosenPointsNormal->points[j]);
  }
  
  chosenPoints = chosenPointsSphere;
  chosenPointsNormal = chosenPointsSphereNormal;
}

template<typename T, typename U>
void GeoGraspEvo::voxelizeCloud(const T & inputCloud, const float & leafSize,
    T outputCloud) {
  U voxelFilter;
  voxelFilter.setInputCloud(inputCloud);
  voxelFilter.setLeafSize(leafSize, leafSize, leafSize);
  voxelFilter.filter(*outputCloud);
}

void GeoGraspEvo::candidatePointsDraw(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
                
  pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("Chosen Points"));
  pcl::PointCloud<pcl::PointXYZ> p4;
    
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color5(cloud, 255, 0, 0);  
      
  p4.push_back(pcl::PointXYZ(this->graspingPoints[0].getVector3fMap()[0], this->graspingPoints[0].getVector3fMap()[1], this->graspingPoints[0].getVector3fMap()[2]));
  p4.push_back(pcl::PointXYZ(this->graspingPoints[1].getVector3fMap()[0], this->graspingPoints[1].getVector3fMap()[1], this->graspingPoints[1].getVector3fMap()[2]));
  
  viewer2->addCoordinateSystem (0.1);  
  viewer2->addPointCloud<pcl::PointXYZ>(cloud, single_color5, "Global PC");
  
  viewer2->addSphere(p4[0], 0.001, 255, 255, 255, "First best grasp point");                  
  viewer2->addSphere(p4[1], 0.001, 255, 255, 255, "Second best grasp point");   
  
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color2(this->chosenPointsNormalVoxel[0], 0, 255, 0); 
  std::string cad1 = "Chosen points x"+ std::to_string(39);
  viewer2->addPointCloud<pcl::PointNormal>(this->chosenPointsNormalVoxel[0], single_color2, cad1);
   
  
  for (int j=0; j<this->apertures.size(); j++){
    for (int k=0; k<chosenPointsNormalVoxel[1]->size(); k++){

      if (j==0){
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color6(this->candidates[j][k], 0, 0, 255); 
        std::string cad = "Chosen points x "+ std::to_string(j*20+k);
        viewer2->addPointCloud<pcl::PointXYZ>(this->candidates[j][k], single_color6, cad);
     }
     else if (j==1){
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color7(this->candidates[j][k], 255, 255, 0); 
        std::string cad = "Chosen points x "+ std::to_string(j*20+k);
        viewer2->addPointCloud<pcl::PointXYZ>(this->candidates[j][k], single_color7, cad);
     }
     else if (j==2){
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color8(this->candidates[j][k], 255, 0, 255); 
        std::string cad = "Chosen points x "+ std::to_string(j*20+k);
        viewer2->addPointCloud<pcl::PointXYZ>(this->candidates[j][k], single_color8, cad);
     }
     else{
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color9(this->candidates[j][k], 0, 255, 255); 
        std::string cad = "Chosen points y "+ std::to_string(j*20+k);
        viewer2->addPointCloud<pcl::PointXYZ>(this->candidates[j][k], single_color9, cad);
      }
    }
  }
    
  while (!viewer2->wasStopped())
    viewer2->spinOnce(100); 
}

void GeoGraspEvo::getPointInfoNormalization(pcl::PointCloud<pcl::PointNormal> normalCloud, Eigen::Vector3f centroidVector, Eigen::Hyperplane<float,3> graspHyperplane, float *centroidDistanceMin, 
                               float *centroidDistanceMax, float *graspPlaneDistanceMin, float *graspPlaneDistanceMax, float *curvatureMin, float *curvatureMax){

  for (size_t i = 0; i < normalCloud.points.size(); i++) {
    pcl::PointNormal thisPoint = normalCloud.points[i];
    Eigen::Vector3f thisVector(thisPoint.x, thisPoint.y, thisPoint.z);
    Eigen::Vector3f new_diff = thisVector - centroidVector;
    float pointCentroidDistance = new_diff.norm();
    float pointGraspPlaneDistance = graspHyperplane.absDistance(thisVector);
    float pointCurvature = thisPoint.curvature;

    if (pointCentroidDistance < *centroidDistanceMin)
      *centroidDistanceMin = pointCentroidDistance;
    if (pointCentroidDistance > *centroidDistanceMax)
      *centroidDistanceMax = pointCentroidDistance;

    if (pointGraspPlaneDistance < *graspPlaneDistanceMin)
      *graspPlaneDistanceMin = pointGraspPlaneDistance;
    if (pointGraspPlaneDistance > *graspPlaneDistanceMax)
      *graspPlaneDistanceMax = pointGraspPlaneDistance; 

    if (pointCurvature < *curvatureMin)
      *curvatureMin = pointCurvature;
    if (pointCurvature > *curvatureMax)
      *curvatureMax = pointCurvature; 
  }
}

void GeoGraspEvo::getPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                       pcl::PointNormal point,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr & chosenPoints, 
                       pcl::PointCloud<pcl::PointNormal>::Ptr & chosenPointsNormal, 
                       bool chooseX,                        
                       bool topView, 
                       int apertureNumber){

  // Need x grasping point and majorVector of cloud to draw "line of border"
  // this->objectAxisCoeff->values[3-5]
  // this->(first)(second)GraspPoint.getVector3fMap()[0-2]  
  Eigen::Vector3f vectorPPPoint(point.getVector3fMap()[0], point.getVector3fMap()[1], point.getVector3fMap()[2]);
  Eigen::Vector3f vectorPPVector(this->objectAxisCoeff->values[3], this->objectAxisCoeff->values[4], this->objectAxisCoeff->values[5]);
  
  // Get points that are between minRadius and maxRadius applying searchLineRadius in the generated line (cm)
  float minRadius = this->apertures[apertureNumber][0];
  float maxRadius = this->apertures[apertureNumber][1];  
  
  Eigen::Vector3f new_diff = graspingPoints[0].getVector3fMap() - graspingPoints[1].getVector3fMap();
  float objWidth = new_diff.norm();

//  float graspRadius = 2.0 * this->gripTipSize / 1000.0;
//  if (graspRadius * 2.0 >= objWidth)
//    graspRadius = objWidth * 0.7 / 2.0;
  const float stepRadius = this->graspRadius/2.0;// 0.001;


  const float searchLineRadius = this->graspRadius/2.0;//0.01;//graspRadius;

////  std::cout<<"gripTipSize="<<this->gripTipSize/1000.0<<", StepRadius="<<stepRadius<<", objWidth="<<objWidth<<", graspRadius="<<graspRadius<<", searchLineRadius="<<searchLineRadius<<std::endl;


    

  float mod = sqrtf(vectorPPVector[0]*vectorPPVector[0]+vectorPPVector[1]*vectorPPVector[1]+vectorPPVector[2]*vectorPPVector[2]);
  Eigen::Vector3f vectorPPVectorNorm(vectorPPVector[0]/mod, vectorPPVector[1]/mod, vectorPPVector[2]/mod);
  
  pcl::search::KdTree<pcl::PointXYZ>::Ptr treeSearch(new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::PointIndices::Ptr pointsIndex(new pcl::PointIndices);
  std::vector<float> pointsSquaredDistance;

  treeSearch->setInputCloud(cloud);
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr preChosenPoints(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointNormal>::Ptr preChosenPointsNormal(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr chosenPointsSphere(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointNormal>::Ptr chosenPointsSphereNormal(new pcl::PointCloud<pcl::PointNormal>);
    
////std::cout<<"Vector ("<<vectorPPVectorNorm[0]<<","<<vectorPPVectorNorm[1]<<","<<vectorPPVectorNorm[2]<<")"<<std::endl;	
	
  for (float i=minRadius; i<maxRadius; i+=stepRadius){
  
    pcl::PointXYZ point (vectorPPPoint[0], vectorPPPoint[1], vectorPPPoint[2]);
    
    point.x = point.x + vectorPPVectorNorm[0]*i;
    point.y = point.y + vectorPPVectorNorm[1]*i;
    point.z = point.z + vectorPPVectorNorm[2]*i;
        
    treeSearch->radiusSearch(point, searchLineRadius, pointsIndex->indices, pointsSquaredDistance);
    
    extractInliersCloud<pcl::PointCloud<pcl::PointXYZ>::Ptr,
                        pcl::ExtractIndices<pcl::PointXYZ> >(cloud, pointsIndex, preChosenPoints);
                        
    extractInliersCloud<pcl::PointCloud<pcl::PointNormal>::Ptr,
                        pcl::ExtractIndices<pcl::PointNormal> >(objectNormalCloud, pointsIndex, preChosenPointsNormal);                    
                      
    for( int j = 0; j < preChosenPoints->size(); j++){
      
        chosenPointsSphere->push_back(preChosenPoints->points[j]);
        chosenPointsSphereNormal->push_back(preChosenPointsNormal->points[j]);
    }                           
    
    pointCloud->push_back(point);
  }   
  
  this->pointClouds.push_back(pointCloud);  
  
  chosenPoints = chosenPointsSphere;
  chosenPointsNormal = chosenPointsSphereNormal;
}


// Tests every combination of points from both initial areas and ranks them in
// function of:
//    - distance to the object's centroid
//    - distance to the grasping plane
//    - curvature
//    - angle between the line formed by the grasping points and their normals
//
// In addition, these ranked points cannot be closer to the centroid than the
// initial points to avoid searching in the objects facing surface, and they
// must be distanced from each other at least as the initial points. 
// Furthermore, those new points cannot be closer to the other side initial
// points.
//
// @param cloud
// @param graspPlaneCoeff -> Coefficients of the grasping plane
// @param centroidPoint -> Object's centroid point
// @param numGrasps -> Since every combination is ranked, this parameter
//                     indicates the number of best combinations that will
//                     be kept in the returning vectors
// @param bestGrasps -> Vector to be filled with best grasps found
// @param bestRanks -> Vector to be filled with the best configurations' ranks
// @param chooseX
// @param topView

void GeoGraspEvo::getBestGraspingPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,    
                                     const pcl::ModelCoefficients::Ptr & graspPlaneCoeff,
                                     const pcl::PointXYZ & centroidPoint,
                                     const int & numGrasps, 
                                     std::vector<GraspEvoContacts> & bestGrasps,
                                     std::vector<float> & bestRanks,
                                     bool chooseX,
                                     bool topView) {
                                     
  const float weightR1 = 1.5, weightR2 = 1.0; // BEST VALUES FOR W1 AND W2

  ////////////////////////////////////////////////////////////////////////////////
  // First we obtain the min and max values in order to normalize data

  Eigen::Vector3f graspNormalVector(graspPlaneCoeff->values[0],
                                    graspPlaneCoeff->values[1],
                                    graspPlaneCoeff->values[2]);
  Eigen::Hyperplane<float,3> graspHyperplane(graspNormalVector,graspPlaneCoeff->values[3]);

#ifdef _DEBUG    
	std::cout<<"Grasp Normal Vector = ("<<graspNormalVector[0]<<","<<graspNormalVector[1]<<","<<graspNormalVector[2]<<")"<<std::endl;
#endif	
	
  Eigen::Vector3f centroidVector(centroidPoint.x,
                                 centroidPoint.y,
                                 centroidPoint.z);
  
  // Inicializacion del vector que contiene posibles puntos para cada dedo para cada punto del voxel
  candidates.clear();
  candidates.resize(this->numFingers-1);
  candidatesNormal.resize(this->numFingers-1);
  
  for (int i=0; i<this->numFingers-1; i++){
    candidates[i].resize(this->chosenPointsNormalVoxel[1]->size());
    candidatesNormal[i].resize(this->chosenPointsNormalVoxel[1]->size());
  }
                                 
  //for (int i=0; i<2; i++){
  
  // Get from initial graspPoint 1 or 2 their normalization parameters
  float centroidDistanceMin, centroidDistanceMax;
  float graspPlaneDistanceMin, graspPlaneDistanceMax;
  float curvatureMin, curvatureMax;
  
  curvatureMin = std::numeric_limits<float>::max();
  centroidDistanceMin = graspPlaneDistanceMin = curvatureMin;
  curvatureMax = std::numeric_limits<float>::min();
  centroidDistanceMax = graspPlaneDistanceMax = curvatureMax; 
  
  getPointInfoNormalization(*this->chosenPointsNormalVoxel[0], centroidVector, graspHyperplane, &centroidDistanceMin, &centroidDistanceMax, &graspPlaneDistanceMin, &graspPlaneDistanceMax, &curvatureMin, &curvatureMax);
                            
  //std::cout<<chosenPointsNormalVoxel[0]->points[0]<<std::endl;
                            
  // For each candidate in voxel, we need possible points 3 and 4
  for (int j=0; j<this->apertures.size(); j++){
    for (int k=0; k<chosenPointsNormalVoxel[1]->size(); k++){

      getPoint(cloud, chosenPointsNormalVoxel[1]->points[k], this->candidates[j][k], this->candidatesNormal[j][k], chooseX, topView, j);
    }
  }
  
  //candidatePointsDraw(cloud);
      
//  const float voxelRadius = (this->gripTipSize / 2000.0)/2;
//  std::cout<<"VoxelRadius value="<<this->voxelRadius<<std::endl;
  pcl::PointCloud<pcl::PointNormal>::Ptr GraspPlaneCloud1(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr GraspPlaneCloud(new pcl::PointCloud<pcl::PointXYZ>);
  
  for (int j=0; j<this->apertures.size(); j++){
    for (int k=0; k<chosenPointsNormalVoxel[1]->size(); k++){
      
      voxelizeCloud<pcl::PointCloud<pcl::PointXYZ>::Ptr, 
                    pcl::VoxelGrid<pcl::PointXYZ> >( this->candidates[j][k], this->voxelRadius, this->candidates[j][k]);
      voxelizeCloud<pcl::PointCloud<pcl::PointNormal>::Ptr, 
                    pcl::VoxelGrid<pcl::PointNormal> >( this->candidatesNormal[j][k], this->voxelRadius, this->candidatesNormal[j][k]);
    }
  }
  
  //candidatePointsDraw(cloud);
  
  candidatesNormalVoxel.clear();
  candidatesNormalVoxel.resize(this->numFingers-1);
  
  for(int j=0; j<this->apertures.size();j++){
    pcl::PointCloud<pcl::PointNormal>::Ptr tempCloud(new pcl::PointCloud<pcl::PointNormal>);
    for (int k=0; k<this->candidatesNormal[j].size(); k++){
      for (int l=0;l<this->candidatesNormal[j][k]->points.size(); l++){
        tempCloud->push_back(this->candidatesNormal[j][k]->points[l]);
      }
    }
    this->candidatesNormalVoxel[j] = tempCloud; 
  }

#ifdef _DEBUG
	std::cout<<"Cloud = "<<cloud->size()<<" ";
	std::cout<<"Puntos en dedos ("<<this->candidatesNormalVoxel[0]->size()<<","<<this->candidatesNormalVoxel[1]->size()<<")";
#endif
  for( int f=0;f<this->apertures.size();f++) {
	voxelizeCloud<pcl::PointCloud<pcl::PointNormal>::Ptr, 
                    pcl::VoxelGrid<pcl::PointNormal> >( this->candidatesNormalVoxel[f], this->voxelRadius, this->candidatesNormalVoxel[f]);
  }

#ifdef _DEBUG
	std::cout<<" -- ("<<this->candidatesNormalVoxel[0]->size()<<","<<this->candidatesNormalVoxel[1]->size()<<")"<<std::endl;
#endif
  
  // YA TENEMOS LOS CANDIDATOS DE P3 Y P4. AHORA SACAR MIN/MAX DE P3 Y P4 COMO EN P1
  std::vector<float> centroidDistanceMinVector;
  std::vector<float> centroidDistanceMaxVector;
  std::vector<float> graspPlaneDistanceMinVector; 
  std::vector<float> graspPlaneDistanceMaxVector;
  std::vector<float> curvatureMinVector;
  std::vector<float> curvatureMaxVector;
  
  centroidDistanceMinVector.clear();
  centroidDistanceMinVector.resize(this->numFingers-1);
  centroidDistanceMaxVector.clear();
  centroidDistanceMaxVector.resize(this->numFingers-1);
  graspPlaneDistanceMinVector.clear();
  graspPlaneDistanceMinVector.resize(this->numFingers-1);
  graspPlaneDistanceMaxVector.clear();
  graspPlaneDistanceMaxVector.resize(this->numFingers-1);
  curvatureMinVector.clear();
  curvatureMinVector.resize(this->numFingers-1);
  curvatureMaxVector.clear();
  curvatureMaxVector.resize(this->numFingers-1);
  
  for(int j=0; j<this->numFingers-1;j++){
    float max = std::numeric_limits<float>::max();
    float min = std::numeric_limits<float>::min();
    
    centroidDistanceMinVector[j] = max;
    graspPlaneDistanceMinVector[j] = max;
    curvatureMinVector[j] = max;
    
    centroidDistanceMaxVector[j] = min;
    graspPlaneDistanceMaxVector[j] = min;
    curvatureMaxVector[j] = min;
  }
  
  for(int j=0; j<this->apertures.size();j++){
      getPointInfoNormalization(*this->candidatesNormalVoxel[j], centroidVector, graspHyperplane, &centroidDistanceMinVector[j], &centroidDistanceMaxVector[j], &graspPlaneDistanceMinVector[j], &graspPlaneDistanceMaxVector[j], &curvatureMinVector[j], &curvatureMaxVector[j]);
  } 
  
  // Also we need max and min distance between P3 and P4
  float maxDP3_P4 = std::numeric_limits<float>::min();
  float minDP3_P4 = std::numeric_limits<float>::max();
  
#ifdef _DEBUG  
std::cout<<"GPDm: "<<graspPlaneDistanceMin<<" "<<graspPlaneDistanceMax<<std::endl;
  std::cout<<"GPD3: "<<graspPlaneDistanceMinVector[0]<<" "<<graspPlaneDistanceMaxVector[0]<<std::endl;
  std::cout<<"GPD4: "<<graspPlaneDistanceMinVector[1]<<" "<<graspPlaneDistanceMaxVector[1]<<std::endl;
#endif  
  /***********************
  for (size_t j = 0; j < this->candidatesNormalVoxel[0]->size(); j++) {
  
    pcl::PointNormal thirdPoint = this->candidatesNormalVoxel[0]->points[j];
    Eigen::Vector3f thirdVector(thirdPoint.x, thirdPoint.y, thirdPoint.z);  
       
    for (size_t k = 0; k < this->candidatesNormalVoxel[1]->size(); k++) {
    
      pcl::PointNormal forthPoint = this->candidatesNormalVoxel[1]->points[k];
      Eigen::Vector3f forthVector(forthPoint.x, forthPoint.y, forthPoint.z);
      float pointDistance = (thirdVector-forthVector).norm();
      
	  
      if(pointDistance>maxDP3_P4)
        maxDP3_P4 = pointDistance;
      if(pointDistance<minDP3_P4)
        minDP3_P4 = pointDistance;
    }
  }
  
#ifdef _DEBUG  
  std::cout<< "Distance P3 to P4= ["<<minDP3_P4<<" - "<<maxDP3_P4<<"]"<<std::endl;
#endif  

*/
  /*std::cout<<"CDm: "<<centroidDistanceMin<<" "<<centroidDistanceMax<<std::endl;
  std::cout<<"GPDm: "<<graspPlaneDistanceMin<<" "<<graspPlaneDistanceMin<<std::endl;
  std::cout<<"Cm: "<<curvatureMin<<" "<<curvatureMax<<std::endl;
  
  std::cout<<"CDm: "<<centroidDistanceMinVector[0]<<" "<<centroidDistanceMaxVector[0]<<std::endl;
  std::cout<<"GPDm: "<<graspPlaneDistanceMinVector[0]<<" "<<graspPlaneDistanceMaxVector[0]<<std::endl;
  std::cout<<"Cm: "<<curvatureMinVector[0]<<" "<<curvatureMaxVector[0]<<std::endl;
    
  std::cout<<"CDm: "<<centroidDistanceMinVector[1]<<" "<<centroidDistanceMaxVector[1]<<std::endl;
  std::cout<<"GPDm: "<<graspPlaneDistanceMinVector[1]<<" "<<graspPlaneDistanceMaxVector[1]<<std::endl;
  std::cout<<"Cm: "<<curvatureMinVector[1]<<" "<<curvatureMaxVector[1]<<std::endl;   */
  
  /*pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("Chosen Points"));
  pcl::PointCloud<pcl::PointXYZ> p4;
    
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color5(cloud, 255, 0, 0);  
      
  p4.push_back(pcl::PointXYZ(this->graspingPoints[0].getVector3fMap()[0], this->graspingPoints[0].getVector3fMap()[1], this->graspingPoints[0].getVector3fMap()[2]));
  p4.push_back(pcl::PointXYZ(this->graspingPoints[1].getVector3fMap()[0], this->graspingPoints[1].getVector3fMap()[1], this->graspingPoints[1].getVector3fMap()[2]));
  
  viewer2->addCoordinateSystem (0.1);  
  viewer2->addPointCloud<pcl::PointXYZ>(cloud, single_color5, "Global PC");
  
  viewer2->addSphere(p4[0], 0.001, 255, 255, 255, "First best grasp point");                  
  viewer2->addSphere(p4[1], 0.001, 255, 255, 255, "Second best grasp point");   
  
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color2(this->chosenPointsNormalVoxel[0], 0, 255, 0); 
  std::string cad1 = "Chosen points x"+ std::to_string(39);
  viewer2->addPointCloud<pcl::PointNormal>(this->chosenPointsNormalVoxel[0], single_color2, cad1);
  
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color6(this->candidatesNormalVoxel[0], 0, 0, 255); 
  std::string cad = "Chosen points x "+ std::to_string(0);
  viewer2->addPointCloud<pcl::PointNormal>(this->candidatesNormalVoxel[0], single_color6, cad);
  
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color7(this->candidatesNormalVoxel[1], 255, 255, 0); 
  std::string cad0 = "Chosen points x "+ std::to_string(1);
  viewer2->addPointCloud<pcl::PointNormal>(this->candidatesNormalVoxel[1], single_color7, cad0);
        
  while (!viewer2->wasStopped())
    viewer2->spinOnce(100); */
                
  ////////////////////////////////////////////////////////////////////////////////
  // Loop in search of the best points tuple

  float pointRank1 = 0.0, pointRank = 0.0, pointRank2 = 0.0, pointRank3 = 0.0;
  float epsilon = 1e-8; // Needed to avoid division by zero

#ifdef _DEBUG
	int contarIteraciones=0;
	int contarIteracionesTotales=0;
	float minmax[2][2]={{1000.0,-1000.0},{1000.0,-1000.0}};
	float minmax2[3][2]={{1000.0,-1000.0},{1000.0,-1000.0},{1000.0,-1000.0}};
	float minmax3[1][2]={{1000.0,-1000.0}};
#endif

  if(bestRanks.size()==0)
    bestRanks.insert(bestRanks.begin(), std::numeric_limits<float>::min());

#ifdef _DEBUG    
  std::cout<<this->apertures[0][0]<<","<<this->apertures[1][1]<<std::endl;
  std::cout<<this->apertures[0][1]<<","<<this->apertures[1][0]<<std::endl;
#endif
  
  std::vector<float> minDistance;
  std::vector<float> maxDistance;
  minDistance.clear();
  minDistance.resize(this->apertures.size()-1);
  maxDistance.clear();
  maxDistance.resize(this->apertures.size()-1);

  std::cout<<"Distancia entre dedos: ";
  for(int aux=0;aux< this->apertures.size()-1;aux++) {
		minDistance[aux] = (this->apertures[aux+1][0]-this->apertures[0][1]);//-epsilon;
		maxDistance[aux] = (this->apertures[aux+1][1]-this->apertures[0][0]);//+epsilon;

		std::cout<<"("<<minDistance[aux]<<","<<maxDistance[aux]<<") ";
  }
  std::cout<<std::endl;
  	
//  float minDistance = (this->apertures[0][0]-this->apertures[1][1]) / 1000.0-epsilon;
//  float maxDistance = (this->apertures[0][1]-this->apertures[1][0]) / 1000.0+epsilon;  

#ifdef _DEBUG
	std::cout<<"Distancia entre dedos ("<<minDistance<<","<<maxDistance<<")"<<std::endl;
	std::cout<<"Bucles ("<<this->chosenPointsNormalVoxel[0]->points.size()<<","<<this->chosenPointsNormalVoxel[1]->points.size()<<","<<this->candidatesNormalVoxel[0]->size()<<","<<this->candidatesNormalVoxel[1]->size()<<")"<<std::endl;
#endif

// Calculus of Max/Min distances
pcl::PointNormal pini, searchPoint;
Eigen::Vector3f vini, searchVector;

float minDistanceLine = std::numeric_limits<float>::max();
float maxDistanceLine = -std::numeric_limits<float>::max();
float dist;

for(size_t k=0; k<this->candidatesNormalVoxel[0]->size();k++ ) {
	pini = this->candidatesNormalVoxel[0]->points[k];
	vini = Eigen::Vector3f(pini.x, pini.y, pini.z);
			
	Eigen::ParametrizedLine<float,3> LineAlongFingers(vini, graspNormalVector);

	for(size_t f = 1;f < this->numFingers-1;f++) {		  
		for(size_t l=0; l<this->candidatesNormalVoxel[f]->size();l++ ) {
			searchPoint = this->candidatesNormalVoxel[f]->points[l];
			searchVector = Eigen::Vector3f(searchPoint.x, searchPoint.y, searchPoint.z);

			dist = (vini - searchVector).norm();
			if(dist>=minDistance[f] && dist<=maxDistance[f]) {
				dist = LineAlongFingers.distance(searchVector);
				if(dist<minDistanceLine) minDistanceLine = dist;
				if(dist>maxDistanceLine) maxDistanceLine = dist;
			}
		}
	}
}

std::cout<<"Distancias a líneas ("<<minDistanceLine<<","<<maxDistanceLine<<")"<<std::endl;
std::cout<<"Número de dedos: "<<numFingers<<std::endl;



bool newTres;


// Varialbes Point 1
pcl::PointNormal firstPoint;
Eigen::Vector3f firstVector;
Eigen::Vector3f firstNormalVector;
float firstPointGraspPlaneDistance;
float firstPointCurvature;

// Variables Point 3
pcl::PointNormal thirdPoint;
Eigen::Vector3f thirdVector;
Eigen::Vector3f thirdNormalVector;
float thirdPointGraspPlaneDistance;
float thirdPointCurvature;

// Variables Point 4
pcl::PointNormal forthPoint;
Eigen::Vector3f forthVector;
Eigen::Vector3f forthNormalVector;
float forthPointGraspPlaneDistance;
float forthPointCurvature;

Eigen::Vector3f normal3_4;
Eigen::Vector3f middle3_4;
Eigen::Vector3f line1_m34_Vector;
float P1_m34AngleCos;
float P34_m34AngleCos;

std::vector<pcl::PointNormal> fingerPoint;
std::vector<Eigen::Vector3f> fingerVector;
std::vector<Eigen::Vector3f> fingerNormalVector;
std::vector<float> fingerPointGraspPlaneDistance;
std::vector<float> fingerPointCurvature;

Eigen::Vector3f fingerPointMiddle;
Eigen::Vector3f fingerNormalMiddle;
Eigen::Vector3f fingerLine1ToMiddleVector;
float fingerAngleCos1ToMiddle;
float fingerAngleCos1ToMiddle2;

  
fingerPoint.clear();
fingerPoint.resize(this->numFingers-1);
fingerVector.clear();
fingerVector.resize(this->numFingers-1);
fingerNormalVector.clear();
fingerNormalVector.resize(this->numFingers-1);
fingerPointGraspPlaneDistance.clear();
fingerPointGraspPlaneDistance.resize(this->numFingers-1);
fingerPointCurvature.clear();
fingerPointCurvature.resize(this->numFingers-1);

std::vector<float> fingerDistanceParallelLine;
fingerDistanceParallelLine.clear();
fingerDistanceParallelLine.resize(this->numFingers-1);

float pointRank4=0;
float minDis = std::numeric_limits<float>::max();;
float maxDis = -std::numeric_limits<float>::max();
float finger1Distance;


Eigen::ParametrizedLine<float,3> LineAlongFingers;
bool valid, endLoop;
int f;
int finger;

std::vector<size_t> fingerPosition;
fingerPosition.clear();
fingerPosition.resize(this->numFingers-1);


int conta=0;


for(finger = 0;finger < this->numFingers-1;finger++) fingerPosition[finger]=0;


  for (size_t i = 0; i < this->chosenPointsNormalVoxel[0]->points.size(); i++) {
  
    firstPoint = this->chosenPointsNormalVoxel[0]->points[i];
    firstVector = Eigen::Vector3f(firstPoint.x, firstPoint.y, firstPoint.z);
    firstNormalVector = Eigen::Vector3f (firstPoint.normal_x, firstPoint.normal_y, firstPoint.normal_z);
    finger1Distance = graspHyperplane.signedDistance(firstVector);
	firstPointGraspPlaneDistance = std::abs(finger1Distance);
    firstPointCurvature = firstPoint.curvature;

    // Normalize
    firstPointGraspPlaneDistance = (firstPointGraspPlaneDistance + epsilon - 
      graspPlaneDistanceMin) / (graspPlaneDistanceMax - 
      graspPlaneDistanceMin);
    firstPointCurvature = (firstPointCurvature + epsilon - curvatureMin) /
      (curvatureMax - curvatureMin);

	endLoop = false;
	finger = 0;	  
	
    while(!endLoop) {

		fingerPointMiddle = Eigen::Vector3f(0,0,0);
		fingerNormalMiddle = Eigen::Vector3f(0,0,0);

		float sumDistances = 0;
		float sumCurvatures = 0;

#ifdef _DEBUG
std::cout<<"Inicio del loop Finger="<<finger<<" fingerPosition ("<<fingerPosition[0];
for(f=1;f<this->numFingers-1;f++) std::cout<<","<<fingerPosition[f];
std::cout<<") de ("<<this->candidatesNormalVoxel[0]->size();
for(f=1;f<this->numFingers-1;f++) std::cout<<","<<this->candidatesNormalVoxel[f]->size();
std::cout<<")"<<std::endl;
#endif
		valid = true;

		for(f = 0;f < this->numFingers-1 && valid;f++) {
		  
			fingerPoint[f] = this->candidatesNormalVoxel[f]->points[fingerPosition[f]];
			fingerVector[f] = Eigen::Vector3f(fingerPoint[f].x, fingerPoint[f].y, fingerPoint[f].z);
			fingerPointGraspPlaneDistance[f] = graspHyperplane.signedDistance(fingerVector[f]);
			dist = fingerPointGraspPlaneDistance[f] - finger1Distance;
			if(dist<this->apertures[f][0] || dist>this->apertures[f][1]) {
				valid = false;				
				conta++;
			} else {
//////////				std::cout<<i<<"-Tras("<<conta<<")-"<<f<<"-"<<fingerPosition[f]<<std::endl;
				conta=0;
//				std::cout<<i<<"-"<<f<<"-"<<"Dist="<<dist<<":"<<finger1Distance*1000.0<<";"<<fingerPointGraspPlaneDistance[f]*1000.0<<" ("<<this->apertures[f][0]<<","<<this->apertures[f][1]<<")"<<std::endl;
				//break;		
				fingerNormalVector[f] = Eigen::Vector3f(fingerPoint[f].normal_x, fingerPoint[f].normal_y, fingerPoint[f].normal_z);
				fingerPointCurvature[f] = fingerPoint[f].curvature;
				fingerPointGraspPlaneDistance[f] = (fingerPointGraspPlaneDistance[f] + epsilon - graspPlaneDistanceMinVector[f]) / (graspPlaneDistanceMaxVector[f] - graspPlaneDistanceMinVector[f]);
				fingerPointCurvature[f] = (fingerPointCurvature[f] + epsilon - curvatureMinVector[f]) / (curvatureMaxVector[f] - curvatureMinVector[f]);
				
				fingerPointMiddle = fingerPointMiddle + fingerVector[f];
				fingerNormalMiddle = fingerNormalMiddle + fingerNormalVector[f];

				sumDistances += fingerPointGraspPlaneDistance[f];
				sumCurvatures += fingerPointCurvature[f];

				if(f==0) {				
					Eigen::ParametrizedLine<float,3> l(fingerVector[f], graspNormalVector);
					LineAlongFingers = l;
				} else {
	//				dist = (fingerVector[0] - fingerVector[f]).norm();
	//				if(dist<minDistance[f-1] || dist>maxDistance[f-1]) valid = false;
					fingerDistanceParallelLine[f] = LineAlongFingers.distance(fingerVector[f]);
				}
			}
		}
		pointRank3 = 0;

		for(f=0;valid && f< this->numFingers-2;f++) {
			 dist = (fingerVector[0] - fingerVector[f+1]).norm();
			 if(dist<minDistance[f] || dist>maxDistance[f]) valid = false;
 			// Normalizo el valor entre 0 y 1.
 			dist = epsilon + (dist-minDistance[f])/(maxDistance[f]-minDistance[f]);

			pointRank3 += dist;
		}
		
//std::cout<<"("<<fingerPosition[0];
//for(f=1;f<this->numFingers-1;f++) std::cout<<","<<fingerPosition[f];
//std::cout<<") -- ("<<minDis<<","<<maxDis<<")"<<std::endl;
		
		// Se sale de la distancia entre dedos de la pinza.
		if( valid ) {

			pointRank3 /= this->numFingers-1;

			fingerPointMiddle = fingerPointMiddle/numFingers;
			fingerNormalMiddle = fingerNormalMiddle/fingerNormalMiddle.norm();
            
			fingerLine1ToMiddleVector = Eigen::ParametrizedLine<float,3>::Through(firstVector, fingerPointMiddle).direction();
			fingerAngleCos1ToMiddle = epsilon + (fingerLine1ToMiddleVector.dot(firstNormalVector)) / (fingerLine1ToMiddleVector.norm() * firstNormalVector.norm());
			fingerAngleCos1ToMiddle2 = epsilon + (fingerLine1ToMiddleVector.dot(fingerNormalMiddle)) / (fingerLine1ToMiddleVector.norm() * fingerNormalMiddle.norm());


			pointRank1 = 2.0 - std::abs(firstPointGraspPlaneDistance - sumDistances);

			pointRank2 = 3.0 - (firstPointCurvature + sumCurvatures) +		  
				std::abs(fingerAngleCos1ToMiddle) + std::abs(fingerAngleCos1ToMiddle2) 
				- std::abs(std::abs(fingerAngleCos1ToMiddle) - std::abs(fingerAngleCos1ToMiddle2));

			pointRank4 = 0;
			for(f=1;f < this->numFingers-1 && valid;f++) pointRank4 +=fingerDistanceParallelLine[f];
			pointRank4 = 1 - ((pointRank4/(numFingers-2))-minDistanceLine)/(maxDistanceLine-minDistanceLine);

			pointRank = weightR1 * pointRank1 + weightR2 * pointRank2 + pointRank3 * 3.0 + pointRank4*4.0;
			
//std::cout<<pointRank<<"=("<<pointRank1<<","<<pointRank2<<","<<pointRank3<<","<<pointRank4<<")"<<std::endl;

if(pointRank4<minDis) minDis = pointRank4;
if(pointRank4>maxDis) maxDis = pointRank4;
			
			
//			std::cout<<"pointRank4="<<pointRank4<<std::endl;

            size_t rank = 0;
            std::vector<GraspEvoContacts>::iterator graspsIt;

            for (rank = 0, graspsIt = bestGrasps.begin(); rank < bestRanks.size(); ++rank, ++graspsIt) {
              if (pointRank > bestRanks[rank]) {
				GraspEvoContacts newGrasp;
				pcl::PointXYZ punto(firstPoint.x, firstPoint.y, firstPoint.z);
				newGrasp.graspContactPoints.push_back(punto);
				for(f = 0;f < this->numFingers-1;f++) {
					punto.x = fingerPoint[f].x;
					punto.y = fingerPoint[f].y;
					punto.z = fingerPoint[f].z;
					newGrasp.graspContactPoints.push_back(punto);
				}
				bestRanks.insert(bestRanks.begin() + rank, pointRank);
				bestGrasps.insert(graspsIt, newGrasp);
				if(bestRanks.size() > numGrasps){
					bestRanks.pop_back();
					bestGrasps.pop_back();
				}
				break;
			  }
			}
		}

// Update the fingers points
/*****
std::cout<<"F="<<finger<<" ("<<fingerPosition[0];
for(f=1;f<numFingers-1;f++) std::cout<<","<<fingerPosition[f];
std::cout<<" de ("<<this->candidatesNormalVoxel[0]->size();
for(f=1;f<numFingers-1;f++) std::cout<<","<<this->candidatesNormalVoxel[f]->size();
std::cout<<")"<<std::endl;
******/

		while(fingerPosition[finger]==this->candidatesNormalVoxel[finger]->size() ) {
			fingerPosition[finger] = 0;
			finger++;
			if(finger==numFingers-1)
				break;
		}
		if(finger==numFingers-1) 
			endLoop= true;
		else if(finger>0) {
			fingerPosition[finger]++;
			finger = 0;
		} else
			fingerPosition[finger]++;
	  }
    }
	
std::cout<<"Distances to line = ("<<minDis<<","<<maxDis<<")"<<std::endl;
  bestRanks.resize(numGrasps);
  bestGrasps.resize(numGrasps);
  
  
  
  pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("Final"));
  pcl::PointCloud<pcl::PointXYZ> p4;
    
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color5(cloud, 255, 0, 0);  
      
  p4.push_back(pcl::PointXYZ(bestGrasps[0].graspContactPoints[0].getVector3fMap()[0], bestGrasps[0].graspContactPoints[0].getVector3fMap()[1], bestGrasps[0].graspContactPoints[0].getVector3fMap()[2]));
  p4.push_back(pcl::PointXYZ(bestGrasps[0].graspContactPoints[1].getVector3fMap()[0], bestGrasps[0].graspContactPoints[1].getVector3fMap()[1], bestGrasps[0].graspContactPoints[1].getVector3fMap()[2]));  
  p4.push_back(pcl::PointXYZ(bestGrasps[0].graspContactPoints[2].getVector3fMap()[0], bestGrasps[0].graspContactPoints[2].getVector3fMap()[1], bestGrasps[0].graspContactPoints[2].getVector3fMap()[2]));
  
  viewer2->addCoordinateSystem (0.1);  
  viewer2->addPointCloud<pcl::PointXYZ>(cloud, single_color5, "Global PC");
  
  viewer2->addSphere(p4[0], 0.001, 255, 255, 255, "First best grasp point");                  
  viewer2->addSphere(p4[1], 0.001, 255, 0, 255, "Second best grasp point");
  viewer2->addSphere(p4[2], 0.001, 0, 255, 255, "Third best grasp point");      
  
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color2(this->chosenPointsNormalVoxel[0], 0, 255, 0); 
  std::string cad1 = "Chosen points x"+ std::to_string(0);
  viewer2->addPointCloud<pcl::PointNormal>(this->chosenPointsNormalVoxel[0], single_color2, cad1);
  
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color6(this->candidatesNormalVoxel[0], 0, 0, 255); 
  std::string cad = "Chosen points x "+ std::to_string(1);
  viewer2->addPointCloud<pcl::PointNormal>(this->candidatesNormalVoxel[0], single_color6, cad);
  
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color7(this->candidatesNormalVoxel[1], 255, 255, 0); 
  std::string cad0 = "Chosen points x "+ std::to_string(2);
  viewer2->addPointCloud<pcl::PointNormal>(this->candidatesNormalVoxel[1], single_color7, cad0);
        
  while (!viewer2->wasStopped())
    viewer2->spinOnce(100);
}
