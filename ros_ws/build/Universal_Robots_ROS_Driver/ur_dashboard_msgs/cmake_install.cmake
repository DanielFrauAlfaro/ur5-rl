# Install script for directory: /daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/daniel/Desktop/ur5-rl/ros_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur_dashboard_msgs/msg" TYPE FILE FILES
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/msg/ProgramState.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/msg/RobotMode.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/msg/SafetyMode.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur_dashboard_msgs/srv" TYPE FILE FILES
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/srv/AddToLog.srv"
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/srv/GetLoadedProgram.srv"
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/srv/GetProgramState.srv"
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/srv/GetRobotMode.srv"
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/srv/GetSafetyMode.srv"
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/srv/IsInRemoteControl.srv"
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/srv/IsProgramRunning.srv"
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/srv/IsProgramSaved.srv"
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/srv/Load.srv"
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/srv/Popup.srv"
    "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/srv/RawRequest.srv"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur_dashboard_msgs/action" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/action/SetMode.action")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur_dashboard_msgs/msg" TYPE FILE FILES
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur_dashboard_msgs/msg/SetModeAction.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur_dashboard_msgs/msg/SetModeActionGoal.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur_dashboard_msgs/msg/SetModeActionResult.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur_dashboard_msgs/msg/SetModeActionFeedback.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur_dashboard_msgs/msg/SetModeGoal.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur_dashboard_msgs/msg/SetModeResult.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur_dashboard_msgs/msg/SetModeFeedback.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur_dashboard_msgs/cmake" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/build/Universal_Robots_ROS_Driver/ur_dashboard_msgs/catkin_generated/installspace/ur_dashboard_msgs-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/include/ur_dashboard_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/share/roseus/ros/ur_dashboard_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/share/common-lisp/ros/ur_dashboard_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/share/gennodejs/ros/ur_dashboard_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/daniel/Desktop/ur5-rl/ros_ws/devel/lib/python3/dist-packages/ur_dashboard_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/lib/python3/dist-packages/ur_dashboard_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/build/Universal_Robots_ROS_Driver/ur_dashboard_msgs/catkin_generated/installspace/ur_dashboard_msgs.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur_dashboard_msgs/cmake" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/build/Universal_Robots_ROS_Driver/ur_dashboard_msgs/catkin_generated/installspace/ur_dashboard_msgs-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur_dashboard_msgs/cmake" TYPE FILE FILES
    "/daniel/Desktop/ur5-rl/ros_ws/build/Universal_Robots_ROS_Driver/ur_dashboard_msgs/catkin_generated/installspace/ur_dashboard_msgsConfig.cmake"
    "/daniel/Desktop/ur5-rl/ros_ws/build/Universal_Robots_ROS_Driver/ur_dashboard_msgs/catkin_generated/installspace/ur_dashboard_msgsConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur_dashboard_msgs" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/src/Universal_Robots_ROS_Driver/ur_dashboard_msgs/package.xml")
endif()
