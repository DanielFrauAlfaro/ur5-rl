# Install script for directory: /daniel/Desktop/ur5-rl/ros_ws/src/ur5_rl

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur5_rl/action" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/src/ur5_rl/action/ManageObject.action")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur5_rl/msg" TYPE FILE FILES
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg"
    "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur5_rl/cmake" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/build/ur5_rl/catkin_generated/installspace/ur5_rl-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/include/ur5_rl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/share/roseus/ros/ur5_rl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/share/common-lisp/ros/ur5_rl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/share/gennodejs/ros/ur5_rl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/daniel/Desktop/ur5-rl/ros_ws/devel/lib/python3/dist-packages/ur5_rl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/lib/python3/dist-packages/ur5_rl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/build/ur5_rl/catkin_generated/installspace/ur5_rl.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur5_rl/cmake" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/build/ur5_rl/catkin_generated/installspace/ur5_rl-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur5_rl/cmake" TYPE FILE FILES
    "/daniel/Desktop/ur5-rl/ros_ws/build/ur5_rl/catkin_generated/installspace/ur5_rlConfig.cmake"
    "/daniel/Desktop/ur5-rl/ros_ws/build/ur5_rl/catkin_generated/installspace/ur5_rlConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ur5_rl" TYPE FILE FILES "/daniel/Desktop/ur5-rl/ros_ws/src/ur5_rl/package.xml")
endif()

