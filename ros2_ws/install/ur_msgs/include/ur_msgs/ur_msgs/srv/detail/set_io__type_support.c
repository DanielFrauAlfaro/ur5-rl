// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from ur_msgs:srv/SetIO.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "ur_msgs/srv/detail/set_io__rosidl_typesupport_introspection_c.h"
#include "ur_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "ur_msgs/srv/detail/set_io__functions.h"
#include "ur_msgs/srv/detail/set_io__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void ur_msgs__srv__SetIO_Request__rosidl_typesupport_introspection_c__SetIO_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ur_msgs__srv__SetIO_Request__init(message_memory);
}

void ur_msgs__srv__SetIO_Request__rosidl_typesupport_introspection_c__SetIO_Request_fini_function(void * message_memory)
{
  ur_msgs__srv__SetIO_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember ur_msgs__srv__SetIO_Request__rosidl_typesupport_introspection_c__SetIO_Request_message_member_array[3] = {
  {
    "fun",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__srv__SetIO_Request, fun),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "pin",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__srv__SetIO_Request, pin),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__srv__SetIO_Request, state),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ur_msgs__srv__SetIO_Request__rosidl_typesupport_introspection_c__SetIO_Request_message_members = {
  "ur_msgs__srv",  // message namespace
  "SetIO_Request",  // message name
  3,  // number of fields
  sizeof(ur_msgs__srv__SetIO_Request),
  ur_msgs__srv__SetIO_Request__rosidl_typesupport_introspection_c__SetIO_Request_message_member_array,  // message members
  ur_msgs__srv__SetIO_Request__rosidl_typesupport_introspection_c__SetIO_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  ur_msgs__srv__SetIO_Request__rosidl_typesupport_introspection_c__SetIO_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ur_msgs__srv__SetIO_Request__rosidl_typesupport_introspection_c__SetIO_Request_message_type_support_handle = {
  0,
  &ur_msgs__srv__SetIO_Request__rosidl_typesupport_introspection_c__SetIO_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ur_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ur_msgs, srv, SetIO_Request)() {
  if (!ur_msgs__srv__SetIO_Request__rosidl_typesupport_introspection_c__SetIO_Request_message_type_support_handle.typesupport_identifier) {
    ur_msgs__srv__SetIO_Request__rosidl_typesupport_introspection_c__SetIO_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ur_msgs__srv__SetIO_Request__rosidl_typesupport_introspection_c__SetIO_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "ur_msgs/srv/detail/set_io__rosidl_typesupport_introspection_c.h"
// already included above
// #include "ur_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "ur_msgs/srv/detail/set_io__functions.h"
// already included above
// #include "ur_msgs/srv/detail/set_io__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void ur_msgs__srv__SetIO_Response__rosidl_typesupport_introspection_c__SetIO_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ur_msgs__srv__SetIO_Response__init(message_memory);
}

void ur_msgs__srv__SetIO_Response__rosidl_typesupport_introspection_c__SetIO_Response_fini_function(void * message_memory)
{
  ur_msgs__srv__SetIO_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember ur_msgs__srv__SetIO_Response__rosidl_typesupport_introspection_c__SetIO_Response_message_member_array[1] = {
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__srv__SetIO_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ur_msgs__srv__SetIO_Response__rosidl_typesupport_introspection_c__SetIO_Response_message_members = {
  "ur_msgs__srv",  // message namespace
  "SetIO_Response",  // message name
  1,  // number of fields
  sizeof(ur_msgs__srv__SetIO_Response),
  ur_msgs__srv__SetIO_Response__rosidl_typesupport_introspection_c__SetIO_Response_message_member_array,  // message members
  ur_msgs__srv__SetIO_Response__rosidl_typesupport_introspection_c__SetIO_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  ur_msgs__srv__SetIO_Response__rosidl_typesupport_introspection_c__SetIO_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ur_msgs__srv__SetIO_Response__rosidl_typesupport_introspection_c__SetIO_Response_message_type_support_handle = {
  0,
  &ur_msgs__srv__SetIO_Response__rosidl_typesupport_introspection_c__SetIO_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ur_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ur_msgs, srv, SetIO_Response)() {
  if (!ur_msgs__srv__SetIO_Response__rosidl_typesupport_introspection_c__SetIO_Response_message_type_support_handle.typesupport_identifier) {
    ur_msgs__srv__SetIO_Response__rosidl_typesupport_introspection_c__SetIO_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ur_msgs__srv__SetIO_Response__rosidl_typesupport_introspection_c__SetIO_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "ur_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "ur_msgs/srv/detail/set_io__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers ur_msgs__srv__detail__set_io__rosidl_typesupport_introspection_c__SetIO_service_members = {
  "ur_msgs__srv",  // service namespace
  "SetIO",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // ur_msgs__srv__detail__set_io__rosidl_typesupport_introspection_c__SetIO_Request_message_type_support_handle,
  NULL  // response message
  // ur_msgs__srv__detail__set_io__rosidl_typesupport_introspection_c__SetIO_Response_message_type_support_handle
};

static rosidl_service_type_support_t ur_msgs__srv__detail__set_io__rosidl_typesupport_introspection_c__SetIO_service_type_support_handle = {
  0,
  &ur_msgs__srv__detail__set_io__rosidl_typesupport_introspection_c__SetIO_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ur_msgs, srv, SetIO_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ur_msgs, srv, SetIO_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ur_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ur_msgs, srv, SetIO)() {
  if (!ur_msgs__srv__detail__set_io__rosidl_typesupport_introspection_c__SetIO_service_type_support_handle.typesupport_identifier) {
    ur_msgs__srv__detail__set_io__rosidl_typesupport_introspection_c__SetIO_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)ur_msgs__srv__detail__set_io__rosidl_typesupport_introspection_c__SetIO_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ur_msgs, srv, SetIO_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ur_msgs, srv, SetIO_Response)()->data;
  }

  return &ur_msgs__srv__detail__set_io__rosidl_typesupport_introspection_c__SetIO_service_type_support_handle;
}
