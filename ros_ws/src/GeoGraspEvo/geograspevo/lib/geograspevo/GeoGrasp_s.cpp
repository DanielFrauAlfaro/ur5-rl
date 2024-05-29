#include "geograsp/GeoGrasp.h"

const float GeoGrasp::kGraspPlaneApprox = 0.007;
const float GeoGrasp::kCloudNormalRadius = 0.03;

GeoGrasp::GeoGrasp() :
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
  apertures = {{10,30},{15,20}};
  numFingers = 2;
}

GeoGrasp::~GeoGrasp() {
  graspingPoints.clear();
  rankings.clear();
  pointsDistance.clear();
}

void GeoGrasp::setBackgroundCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  backgroundCloud = cloud;
}

void GeoGrasp::setBackgroundPlaneCoeff(
    const pcl::ModelCoefficients & coefficients) {
  *backgroundPlaneCoeff = coefficients;
}

void GeoGrasp::setObjectCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  objectCloud = cloud;
}

void GeoGrasp::setGrasps(const int & grasps) {
  numberBestGrasps = grasps;
}

void GeoGrasp::setApertures(const std::vector<std::vector<int>> & open) {

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

void GeoGrasp::setNumberFingers(const int & fingers){
  numFingers = fingers;
}

void GeoGrasp::setGripTipSize(const int & size) {
  gripTipSize = size;
}

void GeoGrasp::setUniqueMobility(const bool & mov){
  uniqueMobility = mov;
}

GraspContacts GeoGrasp::getBestGrasp() const {
  GraspContacts grasp;

  if (this->bestGraspingPoints.empty())
    std::cout << "No grasp configurations were found during the computation\n";
  else {
  
    for (int i=0; i<this->bestGraspingPoints[0].graspContactPoints.size(); i++){
      grasp.graspContactPoints.push_back(this->bestGraspingPoints[0].graspContactPoints[i]);
    }

  }

  return grasp;
}

std::vector<GraspContacts> GeoGrasp::getGrasps() const {
  return this->bestGraspingPoints;
}

GraspPose GeoGrasp::getBestGraspPose() const {
  GraspPose grasp;

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

float GeoGrasp::getBestRanking() const {
  float ranking;

  if (this->bestGraspingPoints.empty())
    std::cout << "No grasp configurations were found during the computation\n";
  else
    ranking = this->rankings[0];

  return ranking;
}

std::vector<float> GeoGrasp::getRankings() const {
  return this->rankings;
}

pcl::ModelCoefficients GeoGrasp::getObjectAxisCoeff() const {
  return *objectAxisCoeff;
}

pcl::ModelCoefficients GeoGrasp::getBackgroundPlaneCoeff() const {
  return *backgroundPlaneCoeff;
}

void GeoGrasp::compute() {
  // Some ratios dependant of the gripper
  const float voxelRadius = this->gripTipSize / 2000.0;
  float graspRadius = 2.0 * this->gripTipSize / 1000.0;

  std::cout << "Grasp radius: " << graspRadius << "\n"
            << "Voxel radius: " << voxelRadius << "\n";

  std::cout << "Loaded " << this->objectCloud->width * this->objectCloud->height 
            << " data points from object cloud\n";

  for (int j=0; j<this->apertures.size(); j++){
	std::cout << "Apertura ("<<j<<") -> ["<<this->apertures[j][0]/1000.0<<", "<<this->apertures[j][1]/1000.0<<"]"<<std::endl;  
  }

  // Background plane
  if (backgroundPlaneCoeff->values.size() == 0)
    computeCloudPlane(this->backgroundCloud, this->backgroundPlaneCoeff);

  // Cloud plane filtering
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  float outliersThreshold = 1.0;
  int meanNeighbours = 50;

  filterOutliersFromCloud(this->objectCloud, meanNeighbours, outliersThreshold,
    cloud);

  // Object normals
  // This could be optimised passing a voxelized cloud instead
  ////////////computeCloudNormals(cloud, this->kCloudNormalRadius, this->objectNormalCloud);


  // Aproximate the object's main axis and centroid
  pcl::PointCloud<pcl::PointXYZ>::Ptr voxelCloud(new pcl::PointCloud<pcl::PointXYZ>);
  voxelizeCloud<pcl::PointCloud<pcl::PointXYZ>::Ptr, 
                pcl::VoxelGrid<pcl::PointXYZ> >(cloud, voxelRadius, voxelCloud);
  ////////////////////////
  // Object normals
  // This could be optimised passing a voxelized cloud instead
  computeCloudNormals(voxelCloud, this->kCloudNormalRadius, this->objectNormalCloud);


std::cout<<"Nube global, tiene "<<cloud->size()<<" y voxelizada "<<voxelCloud->size()<<std::endl;

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

  std::cout<<"Mejor pto de agarre (inicial de "<<this->graspingPoints.size()<<"): ";
  for (int i=0; i<this->graspingPoints.size(); i++){
    std::cout<<"( ";
    for (int j=0; j<this->graspingPoints[i].getVector3fMap().size(); j++){
      std::cout<<this->graspingPoints[i].getVector3fMap()[j]<<" ";
    }
    std::cout<<")";
  }
  std::cout<<std::endl;
  
  Eigen::Vector3f new_diff = this->graspingPoints[0].getVector3fMap() - this->graspingPoints[1].getVector3fMap();
  float objWidth = new_diff.norm();

  if (graspRadius * 2.0 >= objWidth)
    graspRadius = objWidth * 0.7 / 2.0;

  // Grasping areas
  pcl::PointCloud<pcl::PointXYZ>::Ptr candidatePoints (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointNormal>::Ptr candidatePointsNormal (new pcl::PointCloud<pcl::PointNormal>);
  
  this->chosenPoints.clear();
  this->chosenPointsNormal.clear();

std::cout<<"GrapRadius ="<<graspRadius<<std::endl;
/////////////////////////////
  getClosestPointsByRadius(this->graspingPoints[0], graspRadius, voxelCloud, 
                           this->objectNormalCloud, candidatePoints, candidatePointsNormal);
  std::cout<<"GetClosestPointsByRadius 0 "<<std::endl;                        
  this->chosenPoints.push_back(candidatePoints);
  this->chosenPointsNormal.push_back(candidatePointsNormal);
  
/////////////////////////////
  getClosestPointsByRadius(this->graspingPoints[1], graspRadius, voxelCloud, 
                           this->objectNormalCloud, candidatePoints, candidatePointsNormal);
  std::cout<<"GetClosestPointsByRadius 1"<<std::endl;                         
  this->chosenPoints.push_back(candidatePoints);
  this->chosenPointsNormal.push_back(candidatePointsNormal);     
  
  //drawPossiblePoints(cloud, 0);    

  // Refine those areas
  pcl::PointCloud<pcl::PointNormal>::Ptr candidatePointsVoxel(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr candidatePointsVoxel1(new pcl::PointCloud<pcl::PointNormal>);
  
  this->chosenPointsNormalVoxel.clear();
  
  voxelizeCloud<pcl::PointCloud<pcl::PointNormal>::Ptr, 
                pcl::VoxelGrid<pcl::PointNormal> >(this->chosenPointsNormal[0], voxelRadius, candidatePointsVoxel);
                
std::cout<<"Nube Normal Points 0 tiene "<<chosenPointsNormal[0]->size()<<" y voxelizada "<<candidatePointsVoxel->size()<<std::endl;

  this->chosenPointsNormalVoxel.push_back(candidatePointsVoxel);

  voxelizeCloud<pcl::PointCloud<pcl::PointNormal>::Ptr, 
                pcl::VoxelGrid<pcl::PointNormal> >(this->chosenPointsNormal[1], voxelRadius, candidatePointsVoxel1);

std::cout<<"Nube Normal Points 1 tiene "<<chosenPointsNormal[1]->size()<<" y voxelizada "<<candidatePointsVoxel1->size()<<std::endl;

  this->chosenPointsNormalVoxel.push_back(candidatePointsVoxel1); 
        
  //drawPossiblePoints(cloud, 1);                      

  // Best ranking
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  
  /////////////////////////////
  getBestGraspingPoints(voxelCloud, this->graspPlaneCoeff, this->objectCentroidPoint, this->numberBestGrasps, 
                        this->bestGraspingPoints, this->rankings, chooseX, topView);
           
  // Cambio P1 y P2 de orden y vuelvo a llamar a la funcion nuevamente             
  pcl::PointCloud<pcl::PointNormal>::Ptr aux (new pcl::PointCloud<pcl::PointNormal>);
  aux = this->chosenPointsNormalVoxel[0];
  this->chosenPointsNormalVoxel[0] = this->chosenPointsNormalVoxel[1];
  this->chosenPointsNormalVoxel[1] = aux;

/////////////////////////////
  
  getBestGraspingPoints(voxelCloud, this->graspPlaneCoeff, this->objectCentroidPoint, this->numberBestGrasps, 
                        this->bestGraspingPoints, this->rankings, chooseX, topView);
    
  this->rankings.resize(this->numberBestGrasps);
  this->bestGraspingPoints.resize(this->numberBestGrasps);
  
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout<<"Elapsed time: "<<elapsed_seconds.count()<<"s"<<std::endl;
  
  for (size_t i = 0; i < this->rankings.size(); ++i) {
    std::cout << "Grasp Configuration Rank: " << this->rankings[i];
	
	 Eigen::Vector3f thirdVector = this->bestGraspingPoints[i].graspContactPoints[1].getVector3fMap();
     Eigen::Vector3f forthVector = this->bestGraspingPoints[i].graspContactPoints[2].getVector3fMap();
     float pointDistance = (thirdVector-forthVector).norm();
      
     std::cout<< " Distance Finger 3-4: "<<pointDistance;
  
	
	std::cout << std::endl;
  }

  std::cout << "===============================\n";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// - - - - - - - - - - - - - - - AUXILIARY FUNCTIONS - - - - - - - - - - - - - - -
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void GeoGrasp::computeCloudPlane(
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

void GeoGrasp::filterOutliersFromCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & inputCloud,
    const int & meanNeighbours, const float & distanceThreshold,
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud) {
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sorFilter;

  sorFilter.setInputCloud(inputCloud);
  sorFilter.setMeanK(meanNeighbours);
  sorFilter.setStddevMulThresh(distanceThreshold);
  sorFilter.filter(*outputCloud);
}

void GeoGrasp::computeCloudNormals(
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

void GeoGrasp::computeCloudGeometry(
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
void GeoGrasp::extractInliersCloud(const T & inputCloud,
    const pcl::PointIndices::Ptr & inputCloudInliers, T outputCloud) {
  U extractor;

  extractor.setInputCloud(inputCloud);
  extractor.setIndices(inputCloudInliers);
  extractor.filter(*outputCloud);
}

void GeoGrasp::findInitialPointsInSideView(bool *chooseX) {

  // Background plane normal and object main axis angle
  Eigen::Vector3f objAxisVector(this->objectAxisCoeff->values[3],
                                this->objectAxisCoeff->values[4],
                                this->objectAxisCoeff->values[5]);
  Eigen::Vector3f backNormalVector(this->backgroundPlaneCoeff->values[0],
                                   this->backgroundPlaneCoeff->values[1],
                                   this->backgroundPlaneCoeff->values[2]);
  Eigen::Vector3f worldZVector(0, 0, 1);

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

void GeoGrasp::findInitialPointsInTopView(bool *chooseX, bool *topView) {

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

void GeoGrasp::drawPossiblePoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, bool voxel){

  pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Chosen Points"));
  pcl::PointCloud<pcl::PointXYZ> p3;
    
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color0(cloud, 255, 0, 0);  
      
  p3.push_back(pcl::PointXYZ(this->graspingPoints[0].getVector3fMap()[0], this->graspingPoints[0].getVector3fMap()[1], this->graspingPoints[0].getVector3fMap()[2]));
  p3.push_back(pcl::PointXYZ(this->graspingPoints[1].getVector3fMap()[0], this->graspingPoints[1].getVector3fMap()[1], this->graspingPoints[1].getVector3fMap()[2]));
  
  viewer1->addCoordinateSystem (0.1);  
  viewer1->addPointCloud<pcl::PointXYZ>(cloud, single_color0, "Global PC");
  
  viewer1->addSphere(p3[0], 0.001, 255, 255, 255, "First best grasp point");                  
  viewer1->addSphere(p3[1], 0.001, 255, 255, 255, "Second best grasp point"); 

  if (voxel == 0){
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(this->chosenPoints[0], 0, 255, 0);  
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(this->chosenPoints[1], 0, 0, 255); 
    
    viewer1->addPointCloud<pcl::PointXYZ>(this->chosenPoints[0], single_color1, "Chosen points 1");
    viewer1->addPointCloud<pcl::PointXYZ>(this->chosenPoints[1], single_color2, "Chosen points 2");
  }
  else{
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color1(this->chosenPointsNormalVoxel[0], 0, 255, 0);  
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color2(this->chosenPointsNormalVoxel[1], 0, 0, 255);
    
    viewer1->addPointCloud<pcl::PointNormal>(this->chosenPointsNormalVoxel[0], single_color1, "Chosen points 1");
    viewer1->addPointCloud<pcl::PointNormal>(this->chosenPointsNormalVoxel[1], single_color2, "Chosen points 2");
  }
    
  while (!viewer1->wasStopped())
    viewer1->spinOnce(100);  
}

void GeoGrasp::buildGraspingPlane(const pcl::PointXYZ & planePoint,
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

void GeoGrasp::getClosestPointsByRadius(const pcl::PointNormal & point,
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

std::cout<<"Tamaño de la nube de entrada es "<<inputCloud->size()<<" Tamaño normales de entrada es: "<<inputNormalCloud->size()<<std::endl;
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
void GeoGrasp::voxelizeCloud(const T & inputCloud, const float & leafSize,
    T outputCloud) {
  U voxelFilter;
  voxelFilter.setInputCloud(inputCloud);
  voxelFilter.setLeafSize(leafSize, leafSize, leafSize);
  voxelFilter.filter(*outputCloud);
}

void GeoGrasp::candidatePointsDraw(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
                
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

void GeoGrasp::getPointInfoNormalization(pcl::PointCloud<pcl::PointNormal> normalCloud, Eigen::Vector3f centroidVector, Eigen::Hyperplane<float,3> graspHyperplane, float *centroidDistanceMin, 
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

void GeoGrasp::getPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
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
  float stepRadius = 0.001;
  float minRadius = this->apertures[apertureNumber][0] / 1000.0;
  float maxRadius = this->apertures[apertureNumber][1] / 1000.0;  
  
  Eigen::Vector3f new_diff = graspingPoints[0].getVector3fMap() - graspingPoints[1].getVector3fMap();
  float objWidth = new_diff.norm();

  float graspRadius = 2.0 * this->gripTipSize / 1000.0;
    
  if (graspRadius * 2.0 >= objWidth)
    graspRadius = objWidth * 0.7 / 2.0;
    
  float searchLineRadius = 0.01;//graspRadius;

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

void GeoGrasp::getBestGraspingPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,    
                                     const pcl::ModelCoefficients::Ptr & graspPlaneCoeff,
                                     const pcl::PointXYZ & centroidPoint,
                                     const int & numGrasps, 
                                     std::vector<GraspContacts> & bestGrasps,
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
	std::cout<<"Candidatos ("<<i<<") = "<<candidates[i].size()<<std::endl;
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
  
  getPointInfoNormalization(*this->chosenPointsNormalVoxel[0], centroidVector, graspHyperplane, &centroidDistanceMin, &centroidDistanceMax, 
                            &graspPlaneDistanceMin, &graspPlaneDistanceMax, &curvatureMin, &curvatureMax);
                            
  //std::cout<<chosenPointsNormalVoxel[0]->points[0]<<std::endl;
                            
  // For each candidate in voxel, we need possible points 3 and 4
  for (int j=0; j<this->apertures.size(); j++){
    for (int k=0; k<chosenPointsNormalVoxel[1]->size(); k++){

      getPoint(cloud, chosenPointsNormalVoxel[1]->points[k], this->candidates[j][k], this->candidatesNormal[j][k], chooseX, topView, j);
    }
  }
  
  //candidatePointsDraw(cloud);
      
  const float voxelRadius = this->gripTipSize / 2000.0;
  pcl::PointCloud<pcl::PointNormal>::Ptr GraspPlaneCloud1(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr GraspPlaneCloud(new pcl::PointCloud<pcl::PointXYZ>);
  
  for (int j=0; j<this->apertures.size(); j++){
std::cout<<"Inicialmente hay candidatos de la apertura "<<j<<" hay "<<this->candidatesNormal[j].size()<<std::endl;
    for (int k=0; k<chosenPointsNormalVoxel[1]->size(); k++){
      
      voxelizeCloud<pcl::PointCloud<pcl::PointXYZ>::Ptr, 
                    pcl::VoxelGrid<pcl::PointXYZ> >( this->candidates[j][k], voxelRadius, this->candidates[j][k]);
      voxelizeCloud<pcl::PointCloud<pcl::PointNormal>::Ptr, 
                    pcl::VoxelGrid<pcl::PointNormal> >( this->candidatesNormal[j][k], voxelRadius, this->candidatesNormal[j][k]);
    }
  }
  
  //candidatePointsDraw(cloud);
  
  candidatesNormalVoxel.clear();
  candidatesNormalVoxel.resize(this->numFingers-1);
  
  std::cout<<"Aperturas = "<<this->apertures.size()<<std::endl;
  
  for(int j=0; j<this->apertures.size();j++){
    pcl::PointCloud<pcl::PointNormal>::Ptr tempCloud(new pcl::PointCloud<pcl::PointNormal>);
std::cout<<"Candidatos de la apertura "<<j<<" hay "<<this->candidatesNormal[j].size()<<std::endl;
    for (int k=0; k<this->candidatesNormal[j].size(); k++){
      for (int l=0;l<this->candidatesNormal[j][k]->points.size(); l++){
        tempCloud->push_back(this->candidatesNormal[j][k]->points[l]);
      }
    }
    this->candidatesNormalVoxel[j] = tempCloud; 
std::cout<<"Candidatos del Voxel "<<j<<" son: "<<this->candidatesNormalVoxel[j]->size()<<std::endl;
  }
  
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
      getPointInfoNormalization(*this->candidatesNormalVoxel[j], centroidVector, graspHyperplane, &centroidDistanceMinVector[j], &centroidDistanceMaxVector[j], 
                                &graspPlaneDistanceMinVector[j], &graspPlaneDistanceMaxVector[j], &curvatureMinVector[j], &curvatureMaxVector[j]);
  } 
  
  // Also we need max and min distance between P3 and P4
  float maxDP3_P4 = std::numeric_limits<float>::min();
  float minDP3_P4 = std::numeric_limits<float>::max();
  
  std::cout<<"Puntos candidatos 0 = "<<this->candidatesNormalVoxel[0]->size()<<" Puntos candidatos 1 = "<<this->candidatesNormalVoxel[1]->size()<<std::endl;
  
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
  std::cout<< "Distance P3 to P4= ["<<minDP3_P4<<" - "<<maxDP3_P4<<"]"<<std::endl;
  
  /*std::cout<<"CDm: "<<centroidDistanceMin<<" "<<centroidDistanceMax<<std::endl;
  std::cout<<"GPDm: "<<graspPlaneDistanceMin<<" "<<graspPlaneDistanceMax<<std::endl;
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
                
                
  std::cout<<"Bucles (1,2,3)=("<<this->chosenPointsNormalVoxel[0]->points.size() <<","<< this->candidatesNormalVoxel[0]->size()<<","<< this->candidatesNormalVoxel[1]->size() <<")"<<std::endl;
  int mejoraGrasp=0;  
  int calculadosGrasp = 0;
  ////////////////////////////////////////////////////////////////////////////////
  // Loop in search of the best points tuple

  float pointRank1 = 0.0, pointRank = 0.0, pointRank2 = 0.0, pointRank3 = 0.0;
  float epsilon = 1e-8; // Needed to avoid division by zero

  if(bestRanks.size()==0)
    bestRanks.insert(bestRanks.begin(), std::numeric_limits<float>::min());

std::cout<<this->apertures[0][0]<<","<<this->apertures[1][1]<<std::endl;
std::cout<<this->apertures[0][1]<<","<<this->apertures[1][0]<<std::endl;
  float minDistance = (this->apertures[0][0]-this->apertures[1][1]) / 1000.0-epsilon;
  float maxDistance = (this->apertures[0][1]-this->apertures[1][0]) / 1000.0+epsilon;  
 
  std::cout<< "MIN/MAX = ["<<minDistance<<", "<<maxDistance<<"]"<<std::endl;

  for (size_t i = 0; i < this->chosenPointsNormalVoxel[0]->points.size(); i++) {
  
    pcl::PointNormal firstPoint = this->chosenPointsNormalVoxel[0]->points[i];
    Eigen::Vector3f firstVector(firstPoint.x, firstPoint.y, firstPoint.z);
    Eigen::Vector3f firstNormalVector(firstPoint.normal_x, firstPoint.normal_y, firstPoint.normal_z);
    float firstPointGraspPlaneDistance = graspHyperplane.absDistance(firstVector);
    float firstPointCurvature = firstPoint.curvature;

    // Normalize
    firstPointGraspPlaneDistance = (firstPointGraspPlaneDistance + epsilon - 
      graspPlaneDistanceMin) / (graspPlaneDistanceMax - 
      graspPlaneDistanceMin);
    firstPointCurvature = (firstPointCurvature + epsilon - curvatureMin) /
      (curvatureMax - curvatureMin);

   for (size_t j = 0; j < this->chosenPointsNormalVoxel[1]->points.size(); j++) {
    for (size_t k = 0; k < this->candidatesNormal[0][j]->points.size(); k++) {
      pcl::PointNormal thirdPoint = this->candidatesNormal[0][j]->points[k];
      Eigen::Vector3f thirdVector(thirdPoint.x, thirdPoint.y, thirdPoint.z);
      Eigen::Vector3f thirdNormalVector(thirdPoint.normal_x, thirdPoint.normal_y, thirdPoint.normal_z);
      float thirdPointGraspPlaneDistance = graspHyperplane.absDistance(thirdVector);
      float thirdPointCurvature = thirdPoint.curvature;

      Eigen::Vector3f graspFirstThirdDirectionVector = Eigen::ParametrizedLine<float,3>::Through(firstVector, thirdVector).direction();
      float graspPlaneAngleCosFirstThird = epsilon + 
                                           std::abs((graspFirstThirdDirectionVector.dot(graspNormalVector)) / (graspFirstThirdDirectionVector.norm() * graspNormalVector.norm()));

      float firstGraspAngleCos = epsilon + 
                                 std::abs((graspFirstThirdDirectionVector.dot(firstNormalVector)) / (graspFirstThirdDirectionVector.norm() * firstNormalVector.norm()));
      float thirdGraspAngleCos = epsilon + 
                                 std::abs((graspFirstThirdDirectionVector.dot(thirdNormalVector)) / (graspFirstThirdDirectionVector.norm() * thirdNormalVector.norm()));

      // Normalize
      thirdPointGraspPlaneDistance = (thirdPointGraspPlaneDistance + epsilon - graspPlaneDistanceMinVector[0]) / 
                                     (graspPlaneDistanceMaxVector[0] - graspPlaneDistanceMinVector[0]);
      thirdPointCurvature = (thirdPointCurvature + epsilon - curvatureMinVector[0]) / 
                            (curvatureMaxVector[0] - curvatureMinVector[0]);

// Me faltarían las aperturas, que pongo 0 de entrada.
      for (size_t l = 0; l < this->candidatesNormal[1][j]->points.size(); l++) {
        pcl::PointNormal forthPoint = this->candidatesNormal[1][j]->points[l];
        Eigen::Vector3f forthVector(forthPoint.x, forthPoint.y, forthPoint.z);
        Eigen::Vector3f forthNormalVector(forthPoint.normal_x, forthPoint.normal_y,forthPoint.normal_z);
        float forthPointGraspPlaneDistance = graspHyperplane.absDistance(forthVector);
        float forthPointCurvature = forthPoint.curvature;

        Eigen::Vector3f graspFirstForthDirectionVector = Eigen::ParametrizedLine<float,3>::Through(firstVector, forthVector).direction();
        float graspPlaneAngleCosFirstForth = epsilon + 
                                             std::abs((graspFirstForthDirectionVector.dot(graspNormalVector)) / (graspFirstForthDirectionVector.norm() * graspNormalVector.norm()));

        float firstGraspAngleCos2 = epsilon + 
                                   std::abs((graspFirstForthDirectionVector.dot(firstNormalVector)) / (graspFirstForthDirectionVector.norm() * firstNormalVector.norm()));
        float forthGraspAngleCos = epsilon + 
                                   std::abs((graspFirstForthDirectionVector.dot(forthNormalVector)) / (graspFirstForthDirectionVector.norm() * forthNormalVector.norm()));

        // Normalize
        forthPointGraspPlaneDistance = (forthPointGraspPlaneDistance + epsilon - graspPlaneDistanceMinVector[1]) / 
                                       (graspPlaneDistanceMaxVector[1] - graspPlaneDistanceMinVector[1]);
        forthPointCurvature = (forthPointCurvature + epsilon - curvatureMinVector[1]) / 
                              (curvatureMaxVector[1] - curvatureMinVector[1]);            

        pointRank1 = 3.0 - (firstPointGraspPlaneDistance + thirdPointGraspPlaneDistance + forthPointGraspPlaneDistance) -
          (std::abs(graspPlaneAngleCosFirstThird - graspPlaneAngleCosFirstForth) - 0.2) * 10.0; // This [-8, 2], being cos=0.2 -> 0 // 1.0 - std::exp(graspPlaneAngleCos * 10.0) / 100.0;
        pointRank2 = 2.0 - (firstPointCurvature + thirdPointCurvature + forthPointCurvature) / 3.0 +
          (firstGraspAngleCos + thirdGraspAngleCos + forthGraspAngleCos) / 3.0 -
          (std::abs(firstGraspAngleCos - thirdGraspAngleCos) + std::abs(firstGraspAngleCos2 - forthGraspAngleCos)) / 2.0;
        pointRank3 = (thirdVector - forthVector).norm();
		
		if(pointRank3<minDistance || pointRank3>maxDistance)
			continue; // Se sale de la distancia entre dedos de la pinza.
		
		
        pointRank3 = ((pointRank3-minDP3_P4) / (maxDP3_P4-minDP3_P4)) - 0.5f;
         
        pointRank = weightR1 * pointRank1 + weightR2 * pointRank2 + pointRank3 * 16.0;

        // Minimum quality criteria
        if (pointRank1 < 1.0 || pointRank2 < 1.0)
          continue;
        
        size_t rank = 0;
        std::vector<GraspContacts>::iterator graspsIt;

calculadosGrasp++;
        for (rank = 0, graspsIt = bestGrasps.begin(); rank < bestRanks.size(); 
            ++rank, ++graspsIt) {
          if (pointRank > bestRanks[rank]) {
          mejoraGrasp++;
            GraspContacts newGrasp;
            pcl::PointXYZ punto(firstPoint.x, firstPoint.y, firstPoint.z);
            newGrasp.graspContactPoints.push_back(punto);
            punto.x = thirdPoint.x;
            punto.y = thirdPoint.y;
            punto.z = thirdPoint.z;
            newGrasp.graspContactPoints.push_back(punto);
            punto.x = forthPoint.x;
            punto.y = forthPoint.y;
            punto.z = forthPoint.z;
            newGrasp.graspContactPoints.push_back(punto);

            bestRanks.insert(bestRanks.begin() + rank, pointRank);
            bestGrasps.insert(graspsIt, newGrasp);
            
            if(bestRanks.size() > numGrasps){
              bestRanks.pop_back();
              bestGrasps.pop_back();
            }

            break;
          }
	  } }
      }  
    }
  }

std::cout<<"Mejorados="<<mejoraGrasp<<" de "<<calculadosGrasp<<". Hay "<<numGrasps<<" agarres"<<std::endl;
  bestRanks.resize(numGrasps);
  bestGrasps.resize(numGrasps);
  
  /*pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("Final"));
  pcl::PointCloud<pcl::PointXYZ> p4;
    
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color5(cloud, 255, 0, 0);  
      
  p4.push_back(pcl::PointXYZ(bestGrasps[0].graspContactPoints[0].getVector3fMap()[0], bestGrasps[0].graspContactPoints[0].getVector3fMap()[1], bestGrasps[0].graspContactPoints[0].getVector3fMap()[2]));
  p4.push_back(pcl::PointXYZ(bestGrasps[0].graspContactPoints[1].getVector3fMap()[0], bestGrasps[0].graspContactPoints[1].getVector3fMap()[1], bestGrasps[0].graspContactPoints[1].getVector3fMap()[2]));  
  p4.push_back(pcl::PointXYZ(bestGrasps[0].graspContactPoints[2].getVector3fMap()[0], bestGrasps[0].graspContactPoints[2].getVector3fMap()[1], bestGrasps[0].graspContactPoints[2].getVector3fMap()[2]));
  
  viewer2->addCoordinateSystem (0.1);  
  viewer2->addPointCloud<pcl::PointXYZ>(cloud, single_color5, "Global PC");
  
  viewer2->addSphere(p4[0], 0.001, 255, 255, 255, "First best grasp point");                  
  viewer2->addSphere(p4[1], 0.001, 255, 255, 255, "Second best grasp point");
  viewer2->addSphere(p4[2], 0.001, 255, 255, 255, "Third best grasp point");      
  
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
    viewer2->spinOnce(100);*/
}
