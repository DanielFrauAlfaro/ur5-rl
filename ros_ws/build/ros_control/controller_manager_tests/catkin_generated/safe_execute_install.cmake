execute_process(COMMAND "/daniel/Desktop/ur5-rl/ros_ws/build/ros_control/controller_manager_tests/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/daniel/Desktop/ur5-rl/ros_ws/build/ros_control/controller_manager_tests/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
