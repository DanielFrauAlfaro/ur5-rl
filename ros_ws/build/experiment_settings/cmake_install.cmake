# Install script for directory: /daniel/Desktop/ur5-rl/ros_ws/src/experiment_settings

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/experiment_settings/action" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/src/experiment_settings/action/IK.action")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/experiment_settings/msg" TYPE FILE FILES
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/experiment_settings/msg/IKAction.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/experiment_settings/msg/IKActionGoal.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/experiment_settings/msg/IKActionResult.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/experiment_settings/msg/IKActionFeedback.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/experiment_settings/msg/IKGoal.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/experiment_settings/msg/IKResult.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/experiment_settings/msg/IKFeedback.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/experiment_settings/msg" TYPE FILE FILES
    "/daniel/Desktop/ur5-rl/ros_ws/src/experiment_settings/msg/Grasp.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/src/experiment_settings/msg/GraspEvoContacts.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/src/experiment_settings/msg/GraspEvoPose.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/src/experiment_settings/msg/Object.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/experiment_settings/cmake" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/build/experiment_settings/catkin_generated/installspace/experiment_settings-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/include/experiment_settings")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/share/roseus/ros/experiment_settings")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/share/common-lisp/ros/experiment_settings")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/share/gennodejs/ros/experiment_settings")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/daniel/Desktop/ur5-rl/ros_ws/devel/lib/python3/dist-packages/experiment_settings")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/lib/python3/dist-packages/experiment_settings")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/build/experiment_settings/catkin_generated/installspace/experiment_settings.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/experiment_settings/cmake" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/build/experiment_settings/catkin_generated/installspace/experiment_settings-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/experiment_settings/cmake" TYPE FILE FILES
    "/daniel/Desktop/ur5-rl/ros_ws/build/experiment_settings/catkin_generated/installspace/experiment_settingsConfig.cmake"
    "/daniel/Desktop/ur5-rl/ros_ws/build/experiment_settings/catkin_generated/installspace/experiment_settingsConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/experiment_settings" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/src/experiment_settings/package.xml")
endif()

