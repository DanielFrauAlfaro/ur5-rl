# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "ur5_rl: 7 messages, 0 services")

set(MSG_I_FLAGS "-Iur5_rl:/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Iactionlib_msgs:/opt/ros/noetic/share/actionlib_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(ur5_rl_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg" NAME_WE)
add_custom_target(_ur5_rl_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ur5_rl" "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg" "ur5_rl/ManageObjectFeedback:std_msgs/Header:ur5_rl/ManageObjectResult:ur5_rl/ManageObjectActionResult:ur5_rl/ManageObjectGoal:ur5_rl/ManageObjectActionGoal:actionlib_msgs/GoalStatus:ur5_rl/ManageObjectActionFeedback:actionlib_msgs/GoalID"
)

get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg" NAME_WE)
add_custom_target(_ur5_rl_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ur5_rl" "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg" "actionlib_msgs/GoalID:ur5_rl/ManageObjectGoal:std_msgs/Header"
)

get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg" NAME_WE)
add_custom_target(_ur5_rl_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ur5_rl" "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg" "ur5_rl/ManageObjectResult:actionlib_msgs/GoalID:actionlib_msgs/GoalStatus:std_msgs/Header"
)

get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg" NAME_WE)
add_custom_target(_ur5_rl_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ur5_rl" "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg" "actionlib_msgs/GoalID:ur5_rl/ManageObjectFeedback:actionlib_msgs/GoalStatus:std_msgs/Header"
)

get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg" NAME_WE)
add_custom_target(_ur5_rl_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ur5_rl" "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg" ""
)

get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg" NAME_WE)
add_custom_target(_ur5_rl_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ur5_rl" "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg" ""
)

get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg" NAME_WE)
add_custom_target(_ur5_rl_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ur5_rl" "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg"
  "${MSG_I_FLAGS}"
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ur5_rl
)
_generate_msg_cpp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ur5_rl
)
_generate_msg_cpp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg"
  "${MSG_I_FLAGS}"
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ur5_rl
)
_generate_msg_cpp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ur5_rl
)
_generate_msg_cpp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ur5_rl
)
_generate_msg_cpp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ur5_rl
)
_generate_msg_cpp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ur5_rl
)

### Generating Services

### Generating Module File
_generate_module_cpp(ur5_rl
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ur5_rl
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(ur5_rl_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(ur5_rl_generate_messages ur5_rl_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_cpp _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_cpp _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_cpp _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_cpp _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_cpp _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_cpp _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_cpp _ur5_rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ur5_rl_gencpp)
add_dependencies(ur5_rl_gencpp ur5_rl_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ur5_rl_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg"
  "${MSG_I_FLAGS}"
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ur5_rl
)
_generate_msg_eus(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ur5_rl
)
_generate_msg_eus(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg"
  "${MSG_I_FLAGS}"
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ur5_rl
)
_generate_msg_eus(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ur5_rl
)
_generate_msg_eus(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ur5_rl
)
_generate_msg_eus(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ur5_rl
)
_generate_msg_eus(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ur5_rl
)

### Generating Services

### Generating Module File
_generate_module_eus(ur5_rl
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ur5_rl
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(ur5_rl_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(ur5_rl_generate_messages ur5_rl_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_eus _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_eus _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_eus _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_eus _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_eus _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_eus _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_eus _ur5_rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ur5_rl_geneus)
add_dependencies(ur5_rl_geneus ur5_rl_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ur5_rl_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg"
  "${MSG_I_FLAGS}"
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ur5_rl
)
_generate_msg_lisp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ur5_rl
)
_generate_msg_lisp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg"
  "${MSG_I_FLAGS}"
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ur5_rl
)
_generate_msg_lisp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ur5_rl
)
_generate_msg_lisp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ur5_rl
)
_generate_msg_lisp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ur5_rl
)
_generate_msg_lisp(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ur5_rl
)

### Generating Services

### Generating Module File
_generate_module_lisp(ur5_rl
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ur5_rl
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(ur5_rl_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(ur5_rl_generate_messages ur5_rl_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_lisp _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_lisp _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_lisp _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_lisp _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_lisp _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_lisp _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_lisp _ur5_rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ur5_rl_genlisp)
add_dependencies(ur5_rl_genlisp ur5_rl_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ur5_rl_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg"
  "${MSG_I_FLAGS}"
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ur5_rl
)
_generate_msg_nodejs(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ur5_rl
)
_generate_msg_nodejs(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg"
  "${MSG_I_FLAGS}"
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ur5_rl
)
_generate_msg_nodejs(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ur5_rl
)
_generate_msg_nodejs(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ur5_rl
)
_generate_msg_nodejs(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ur5_rl
)
_generate_msg_nodejs(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ur5_rl
)

### Generating Services

### Generating Module File
_generate_module_nodejs(ur5_rl
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ur5_rl
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(ur5_rl_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(ur5_rl_generate_messages ur5_rl_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_nodejs _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_nodejs _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_nodejs _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_nodejs _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_nodejs _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_nodejs _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_nodejs _ur5_rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ur5_rl_gennodejs)
add_dependencies(ur5_rl_gennodejs ur5_rl_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ur5_rl_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg"
  "${MSG_I_FLAGS}"
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ur5_rl
)
_generate_msg_py(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ur5_rl
)
_generate_msg_py(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg"
  "${MSG_I_FLAGS}"
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ur5_rl
)
_generate_msg_py(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalID.msg;/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg;/opt/ros/noetic/share/actionlib_msgs/cmake/../msg/GoalStatus.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ur5_rl
)
_generate_msg_py(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ur5_rl
)
_generate_msg_py(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ur5_rl
)
_generate_msg_py(ur5_rl
  "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ur5_rl
)

### Generating Services

### Generating Module File
_generate_module_py(ur5_rl
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ur5_rl
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(ur5_rl_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(ur5_rl_generate_messages ur5_rl_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectAction.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_py _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionGoal.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_py _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionResult.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_py _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectActionFeedback.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_py _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectGoal.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_py _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectResult.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_py _ur5_rl_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/devel/share/ur5_rl/msg/ManageObjectFeedback.msg" NAME_WE)
add_dependencies(ur5_rl_generate_messages_py _ur5_rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ur5_rl_genpy)
add_dependencies(ur5_rl_genpy ur5_rl_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ur5_rl_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ur5_rl)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ur5_rl
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(ur5_rl_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET actionlib_msgs_generate_messages_cpp)
  add_dependencies(ur5_rl_generate_messages_cpp actionlib_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ur5_rl)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ur5_rl
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(ur5_rl_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET actionlib_msgs_generate_messages_eus)
  add_dependencies(ur5_rl_generate_messages_eus actionlib_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ur5_rl)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ur5_rl
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(ur5_rl_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET actionlib_msgs_generate_messages_lisp)
  add_dependencies(ur5_rl_generate_messages_lisp actionlib_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ur5_rl)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ur5_rl
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(ur5_rl_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET actionlib_msgs_generate_messages_nodejs)
  add_dependencies(ur5_rl_generate_messages_nodejs actionlib_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ur5_rl)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ur5_rl\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ur5_rl
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(ur5_rl_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET actionlib_msgs_generate_messages_py)
  add_dependencies(ur5_rl_generate_messages_py actionlib_msgs_generate_messages_py)
endif()
