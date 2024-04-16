# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "canbus_interface: 1 messages, 1 services")

set(MSG_I_FLAGS "-Icanbus_interface:/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(canbus_interface_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg/CanFrame.msg" NAME_WE)
add_custom_target(_canbus_interface_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "canbus_interface" "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg/CanFrame.msg" ""
)

get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/srv/CanFrameSrv.srv" NAME_WE)
add_custom_target(_canbus_interface_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "canbus_interface" "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/srv/CanFrameSrv.srv" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(canbus_interface
  "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg/CanFrame.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/canbus_interface
)

### Generating Services
_generate_srv_cpp(canbus_interface
  "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/srv/CanFrameSrv.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/canbus_interface
)

### Generating Module File
_generate_module_cpp(canbus_interface
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/canbus_interface
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(canbus_interface_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(canbus_interface_generate_messages canbus_interface_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg/CanFrame.msg" NAME_WE)
add_dependencies(canbus_interface_generate_messages_cpp _canbus_interface_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/srv/CanFrameSrv.srv" NAME_WE)
add_dependencies(canbus_interface_generate_messages_cpp _canbus_interface_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(canbus_interface_gencpp)
add_dependencies(canbus_interface_gencpp canbus_interface_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS canbus_interface_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(canbus_interface
  "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg/CanFrame.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/canbus_interface
)

### Generating Services
_generate_srv_eus(canbus_interface
  "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/srv/CanFrameSrv.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/canbus_interface
)

### Generating Module File
_generate_module_eus(canbus_interface
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/canbus_interface
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(canbus_interface_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(canbus_interface_generate_messages canbus_interface_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg/CanFrame.msg" NAME_WE)
add_dependencies(canbus_interface_generate_messages_eus _canbus_interface_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/srv/CanFrameSrv.srv" NAME_WE)
add_dependencies(canbus_interface_generate_messages_eus _canbus_interface_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(canbus_interface_geneus)
add_dependencies(canbus_interface_geneus canbus_interface_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS canbus_interface_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(canbus_interface
  "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg/CanFrame.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/canbus_interface
)

### Generating Services
_generate_srv_lisp(canbus_interface
  "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/srv/CanFrameSrv.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/canbus_interface
)

### Generating Module File
_generate_module_lisp(canbus_interface
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/canbus_interface
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(canbus_interface_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(canbus_interface_generate_messages canbus_interface_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg/CanFrame.msg" NAME_WE)
add_dependencies(canbus_interface_generate_messages_lisp _canbus_interface_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/srv/CanFrameSrv.srv" NAME_WE)
add_dependencies(canbus_interface_generate_messages_lisp _canbus_interface_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(canbus_interface_genlisp)
add_dependencies(canbus_interface_genlisp canbus_interface_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS canbus_interface_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(canbus_interface
  "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg/CanFrame.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/canbus_interface
)

### Generating Services
_generate_srv_nodejs(canbus_interface
  "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/srv/CanFrameSrv.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/canbus_interface
)

### Generating Module File
_generate_module_nodejs(canbus_interface
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/canbus_interface
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(canbus_interface_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(canbus_interface_generate_messages canbus_interface_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg/CanFrame.msg" NAME_WE)
add_dependencies(canbus_interface_generate_messages_nodejs _canbus_interface_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/srv/CanFrameSrv.srv" NAME_WE)
add_dependencies(canbus_interface_generate_messages_nodejs _canbus_interface_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(canbus_interface_gennodejs)
add_dependencies(canbus_interface_gennodejs canbus_interface_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS canbus_interface_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(canbus_interface
  "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg/CanFrame.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/canbus_interface
)

### Generating Services
_generate_srv_py(canbus_interface
  "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/srv/CanFrameSrv.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/canbus_interface
)

### Generating Module File
_generate_module_py(canbus_interface
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/canbus_interface
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(canbus_interface_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(canbus_interface_generate_messages canbus_interface_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/msg/CanFrame.msg" NAME_WE)
add_dependencies(canbus_interface_generate_messages_py _canbus_interface_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/srv/CanFrameSrv.srv" NAME_WE)
add_dependencies(canbus_interface_generate_messages_py _canbus_interface_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(canbus_interface_genpy)
add_dependencies(canbus_interface_genpy canbus_interface_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS canbus_interface_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/canbus_interface)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/canbus_interface
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(canbus_interface_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/canbus_interface)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/canbus_interface
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(canbus_interface_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/canbus_interface)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/canbus_interface
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(canbus_interface_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/canbus_interface)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/canbus_interface
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(canbus_interface_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/canbus_interface)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/canbus_interface\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/canbus_interface
    DESTINATION ${genpy_INSTALL_DIR}
    # skip all init files
    PATTERN "__init__.py" EXCLUDE
    PATTERN "__init__.pyc" EXCLUDE
  )
  # install init files which are not in the root folder of the generated code
  string(REGEX REPLACE "([][+.*()^])" "\\\\\\1" ESCAPED_PATH "${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/canbus_interface")
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/canbus_interface
    DESTINATION ${genpy_INSTALL_DIR}
    FILES_MATCHING
    REGEX "${ESCAPED_PATH}/.+/__init__.pyc?$"
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(canbus_interface_generate_messages_py std_msgs_generate_messages_py)
endif()
