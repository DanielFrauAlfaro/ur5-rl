cmake_minimum_required(VERSION 2.8.3)
project(geograspevo)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS pcl_conversions pcl_ros roscpp sensor_msgs
  INCLUDE_DIRS include
  LIBRARIES GeoGraspEvo
)

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_library(GeoGraspEvo
  lib/${PROJECT_NAME}/GeoGraspEvo.cpp
)

add_executable(cloud_processor src/cloud_processor.cpp)
add_executable(cloud_processor2 src/cloud_processor2.cpp)


target_link_libraries(cloud_processor
  GeoGraspEvo
  ${catkin_LIBRARIES}
  pcl_visualization
)

target_link_libraries(cloud_processor2
  GeoGraspEvo
  ${catkin_LIBRARIES}
  pcl_visualization
)


install(TARGETS GeoGraspEvo
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE)
