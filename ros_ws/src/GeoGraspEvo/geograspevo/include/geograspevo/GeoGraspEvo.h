#ifndef GEOGRASPEVO_H_
#define GEOGRASPEVO_H_

#include <iostream>
#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/centroid.h>
#include <pcl/common/geometry.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/moment_of_inertia_estimation.h>

struct GraspEvoContacts {
  std::vector <pcl::PointXYZ> graspContactPoints;
};

struct GraspEvoPose {
  std::vector <Eigen::Vector3f> graspPosePoints;
  Eigen::Affine3f midPointPose;
};

class GeoGraspEvo {
  public:
    GeoGraspEvo();
    ~GeoGraspEvo();

    void setBackgroundCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    void setBackgroundPlaneCoeff(const pcl::ModelCoefficients & coefficients);
    void setObjectCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    void setGrasps(const int & grasps);
    void setGripTipSize(const int & size);
    void setUniqueMobility(const bool & mov);
    void setApertures(const std::vector<std::vector<float>> & open);
    void setNumberFingers(const int & fingers);

    GraspEvoContacts getBestGrasp() const;
    std::vector<GraspEvoContacts> getGrasps() const;

    GraspEvoPose getBestGraspPose() const;

    float getBestRanking() const;
    std::vector<float> getRankings() const;

    pcl::ModelCoefficients getObjectAxisCoeff() const;
    pcl::ModelCoefficients getBackgroundPlaneCoeff() const;

    void compute();

  private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr backgroundCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr objectCloud;
    pcl::PointCloud<pcl::PointNormal>::Ptr objectNormalCloud;
    pcl::PointCloud<pcl::PointNormal>::Ptr graspPlaneCloud;

    pcl::ModelCoefficients::Ptr backgroundPlaneCoeff;
    pcl::ModelCoefficients::Ptr objectAxisCoeff;
    pcl::ModelCoefficients::Ptr graspPlaneCoeff;

    pcl::PointXYZ objectCentroidPoint;
    std::vector < pcl::PointNormal > graspingPoints;
    pcl::PointNormal secondGraspPoint;

    std::vector<GraspEvoContacts> bestGraspingPoints;
    std::vector<float> rankings;
    std::vector<float> pointsDistance;
    
    std::vector < pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator <pcl::PointCloud <pcl::PointXYZ>::Ptr > > pointClouds;
    std::vector < pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator <pcl::PointCloud<pcl::PointXYZ>::Ptr > > chosenPoints;
    std::vector < pcl::PointCloud<pcl::PointNormal>::Ptr, Eigen::aligned_allocator <pcl::PointCloud<pcl::PointNormal>::Ptr > > chosenPointsNormal;
    std::vector < pcl::PointCloud<pcl::PointNormal>::Ptr, Eigen::aligned_allocator <pcl::PointCloud<pcl::PointNormal>::Ptr > > chosenPointsNormalVoxel;  
    
    std::vector <std::vector < pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator <pcl::PointCloud <pcl::PointXYZ>::Ptr > > > candidates;  
    std::vector <std::vector < pcl::PointCloud<pcl::PointNormal>::Ptr, Eigen::aligned_allocator <pcl::PointCloud <pcl::PointNormal>::Ptr > > > candidatesNormal;  
    std::vector <pcl::PointCloud<pcl::PointNormal>::Ptr, Eigen::aligned_allocator <pcl::PointCloud <pcl::PointNormal>::Ptr > > candidatesNormalVoxel;
        
    int numberBestGrasps;
    int gripTipSize;
    bool uniqueMobility;
    
    std::vector<std::vector<float>> apertures;
    
    int numFingers;

    // Distance threshold for the grasp plane cloud
    static const float kGraspPlaneApprox;
    // Radius used to compute point cloud normals
    static const float kCloudNormalRadius;
	// Meassure error for a valid points.
	static const float kEpsilon;
	// Scale factor of the voxelization procedure.
	static const float kVoxelFactor;

    // Auxiliary functions

    void computeCloudPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
                           pcl::ModelCoefficients::Ptr backPlaneCoeff);

    void filterOutliersFromCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr & inputCloud,
                                 const int & meanNeighbours, 
                                 const float & distanceThreshold,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud);

    void computeCloudNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr & inputCloud,
                             const float & searchRadius,
                             pcl::PointCloud<pcl::PointNormal>::Ptr cloudNormals);

    void computeCloudGeometry(const pcl::PointCloud<pcl::PointXYZ>::Ptr & inputCloud,
                              pcl::ModelCoefficients::Ptr objAxisCoeff, 
                              pcl::PointXYZ & objCenterMass);

    template<typename T, typename U>
    void extractInliersCloud(const T & inputCloud,
                             const pcl::PointIndices::Ptr & inputCloudInliers, 
                             T outputCloud);
      
    void drawCalculus(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                      std::vector < pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator <pcl::PointCloud<pcl::PointXYZ>::Ptr > > chosenPoints);
                      
    void drawPossiblePoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, bool voxel);

    void findInitialPointsInSideView(bool *chooseX);

    void findInitialPointsInTopView(bool *chooseX, bool *topView);

    void buildGraspingPlane(const pcl::PointXYZ & planePoint,
                            const Eigen::Vector3f & planeNormalVector, 
                            const float & distanceThreshold,
                            const pcl::PointCloud<pcl::PointNormal>::Ptr & inputCloud,
                            pcl::ModelCoefficients::Ptr graspPlaneCoeff,
                            pcl::PointCloud<pcl::PointNormal>::Ptr graspPlaneCloud);

    void getClosestPointsByRadius(const pcl::PointNormal & point,
                                  const float & radius,
                                  const pcl::PointCloud<pcl::PointXYZ>::Ptr & inputCloud,
                                  const pcl::PointCloud<pcl::PointNormal>::Ptr & inputNormalCloud,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr & chosenPoints, 
                                  pcl::PointCloud<pcl::PointNormal>::Ptr & chosenPointsNormal);
                                  
    template<typename T, typename U>
    void voxelizeCloud(const T & inputCloud, 
                       const float & leafSize,
                       T outputCloud);
                       
    void candidatePointsDraw(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
	void candidatePointsDraw(pcl::visualization::PCLVisualizer::Ptr & viewer2, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
                       
    void getPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                 pcl::PointNormal point,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr & chosenPoints, 
                 pcl::PointCloud<pcl::PointNormal>::Ptr & chosenPointsNormal, 
                 bool chooseX,
                 bool topView, 
                 int apertureNumber);
                       
    void getPointInfoNormalization(pcl::PointCloud<pcl::PointNormal> normalCloud, 
                                   Eigen::Vector3f centroidVector,
                                   Eigen::Hyperplane<float,3> graspHyperplane,
                                   float *centroidDistanceMin, 
                                   float *centroidDistanceMax, 
                                   float *graspPlaneDistanceMin, 
                                   float *graspPlaneDistanceMax, 
                                   float *curvatureMin, 
                                   float *curvatureMax);

    void getBestGraspingPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,     
                               const pcl::ModelCoefficients::Ptr & graspPlaneCoeff,
                               const pcl::PointXYZ & centroidPoint,
                               const int & numGrasps,  
                               std::vector<GraspEvoContacts> & bestGrasps,
                               std::vector<float> & bestRanks,
                               bool chooseX,
                               bool topView);
    float voxelRadius;
    float graspRadius;
};

#endif
