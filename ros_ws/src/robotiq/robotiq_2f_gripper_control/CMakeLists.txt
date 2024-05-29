cmake_minimum_required(VERSION 3.0.2)
project(robotiq_2f_gripper_control)
find_package(catkin REQUIRED COMPONENTS robotiq_ethercat roscpp rospy message_generation)


add_message_files(
  FILES
  Robotiq2FGripper_robot_input.msg
  Robotiq2FGripper_robot_output.msg
)

catkin_python_setup()

generate_messages()

catkin_package(
    CATKIN_DEPENDS rospy message_runtime roscpp robotiq_ethercat
    INCLUDE_DIRS include
)

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(robotiq_2f_gripper_ethercat_node
  src/robotiq_2f_gripper_control/robotiq_2f_gripper_ethercat_node.cpp
  src/robotiq_2f_gripper_control/robotiq_2f_gripper_ethercat_client.cpp
)

target_link_libraries(robotiq_2f_gripper_ethercat_node
  ${robotiq_ethercat_LIBRARIES}
  ${catkin_LIBRARIES}

)

add_dependencies(robotiq_2f_gripper_ethercat_node robotiq_2f_gripper_control_generate_messages_cpp)

install(PROGRAMS nodes/Robotiq2FGripperSimpleController.py nodes/Robotiq2FGripperStatusListener.py
  nodes/Robotiq2FGripperTcpNode.py nodes/Robotiq2FGripperRtuNode.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
