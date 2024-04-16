# Install script for directory: /daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/daniel/Desktop/ur5-rl/ros_ws/devel/lib/libsoem.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/soem" TYPE FILE FILES
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/soem/ethercat.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/soem/ethercatbase.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/soem/ethercatcoe.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/soem/ethercatconfig.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/soem/ethercatconfiglist.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/soem/ethercatdc.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/soem/ethercateoe.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/soem/ethercatfoe.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/soem/ethercatmain.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/soem/ethercatprint.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/soem/ethercatsoe.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/soem/ethercattype.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/osal/linux/osal_defs.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/osal/osal.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/oshw/linux/nicdrv.h"
    "/daniel/Desktop/ur5-rl/ros_ws/src/soem/SOEM/oshw/linux/oshw.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/daniel/Desktop/ur5-rl/ros_ws/build/soem/SOEM/test/linux/slaveinfo/cmake_install.cmake")
  include("/daniel/Desktop/ur5-rl/ros_ws/build/soem/SOEM/test/linux/eepromtool/cmake_install.cmake")
  include("/daniel/Desktop/ur5-rl/ros_ws/build/soem/SOEM/test/linux/simple_test/cmake_install.cmake")

endif()

