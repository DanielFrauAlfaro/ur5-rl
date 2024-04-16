cmake_minimum_required(VERSION 3.1.3)
project(experiment_settings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS actionlib_msgs geometry_msgs)
add_action_files(
  DIRECTORY action
  FILES IK.action
)



find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(xmlrpcpp REQUIRED)

if(NOT EIGEN3_INCLUDE_DIRS)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

find_package(catkin REQUIRED COMPONENTS
    #image_transport
    cv_bridge
    geograspevo
    message_generation
    moveit_ros_planning_interface
    OpenCV
    pcl_conversions
    pcl_ros
    roscpp
    sensor_msgs
    std_msgs
    #interactive_markers
    #moveit_core
    #moveit_visual_tools
    #moveit_ros_planning
    #moveit_ros_planning_interface
    #pluginlib
    #geometric_shapes
    #rosbag
    #tf2_ros
    #tf2_eigen
    #tf2_geometry_msgs
    #robotiq_2f_gripper_control
)

add_message_files(
  FILES
  Grasp.msg
  GraspEvoContacts.msg
  GraspEvoPose.msg
  Object.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
  moveit_msgs
  geometry_msgs
  actionlib_msgs 
  geometry_msgs
)

catkin_package(
  # LIBRARIES
    #interactivity_utils
  INCLUDE_DIRS
    ${THIS_PACKAGE_INCLUDE_DIRS}
  CATKIN_DEPENDS
    #image_transport
    geograspevo
    message_runtime
    moveit_ros_planning_interface
    pcl_conversions
    pcl_ros
    roscpp
    sensor_msgs
    std_msgs
    moveit_msgs
    xmlrpcpp
    actionlib_msgs
    #moveit_core
    #moveit_visual_tools
    #moveit_ros_planning_interface
    #interactive_markers
  DEPENDS
    EIGEN3
)

add_executable(object src/object.cpp)
add_executable(load_delete_objects src/load_delete_objects.cpp)
add_executable(object_gestion_capture src/object_gestion_capture.cpp)
add_executable(geograsp_evo src/geograsp_evo.cpp)
add_executable(moveit_grasp src/moveit_grasp.cpp)
add_executable(moveit_allegro_bridge src/moveit_allegro_bridge.cpp)
add_executable(joint_states src/joint_states.cpp)
add_executable(moveit_grasp_4f src/moveit_grasp_4f.cpp)
add_executable(gripper_moveit src/gripper_moveit.cpp)
add_executable(moveit_grasp_real src/moveit_grasp_real.cpp)
#add_executable(move_planner_ur5e_sim src/move_planner_ur5e_sim.cpp)
#add_executable(move_planner_ur5e_real src/move_planner_ur5e_real.cpp)
#add_executable(inverse_trajectory src/inverse_trajectory.cpp)


include_directories(include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})

target_link_libraries(object ${catkin_LIBRARIES})
target_link_libraries(load_delete_objects ${catkin_LIBRARIES})
target_link_libraries(object_gestion_capture ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(geograsp_evo ${catkin_LIBRARIES} pcl_visualization)
target_link_libraries(moveit_grasp ${catkin_LIBRARIES})
target_link_libraries(moveit_allegro_bridge ${catkin_LIBRARIES})
target_link_libraries(joint_states ${catkin_LIBRARIES})
target_link_libraries(moveit_grasp_4f ${catkin_LIBRARIES})
target_link_libraries(gripper_moveit ${catkin_LIBRARIES})
target_link_libraries(moveit_grasp_real ${catkin_LIBRARIES})
#target_link_libraries(move_planner_ur5e_sim ${catkin_LIBRARIES} ${Boost_LIBRARIES})
#target_link_libraries(move_planner_ur5e_real ${catkin_LIBRARIES} ${Boost_LIBRARIES})
#target_link_libraries(inverse_trajectory ${catkin_LIBRARIES} ${Boost_LIBRARIES})


add_dependencies(object experiment_settings_generate_messages_cpp)
add_dependencies(load_delete_objects experiment_settings_generate_messages_cpp)
add_dependencies(object_gestion_capture experiment_settings_generate_messages_cpp)
add_dependencies(geograsp_evo experiment_settings_generate_messages_cpp)
add_dependencies(moveit_grasp experiment_settings_generate_messages_cpp)
add_dependencies(moveit_allegro_bridge experiment_settings_generate_messages_cpp)
add_dependencies(joint_states experiment_settings_generate_messages_cpp)
add_dependencies(moveit_grasp experiment_settings_generate_messages_cpp)
add_dependencies(gripper_moveit experiment_settings_generate_messages_cpp)
add_dependencies(moveit_grasp_real experiment_settings_generate_messages_cpp)

#install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#  PATTERN "setup_assistant.launch" EXCLUDE)
#install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
