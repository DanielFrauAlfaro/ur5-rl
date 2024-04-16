// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from ur_msgs:msg/Digital.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "ur_msgs/msg/detail/digital__rosidl_typesupport_introspection_c.h"
#include "ur_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "ur_msgs/msg/detail/digital__functions.h"
#include "ur_msgs/msg/detail/digital__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void ur_msgs__msg__Digital__rosidl_typesupport_introspection_c__Digital_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ur_msgs__msg__Digital__init(message_memory);
}

void ur_msgs__msg__Digital__rosidl_typesupport_introspection_c__Digital_fini_function(void * message_memory)
{
  ur_msgs__msg__Digital__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember ur_msgs__msg__Digital__rosidl_typesupport_introspection_c__Digital_message_member_array[2] = {
  {
    "pin",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__Digital, pin),  // bytes offset in struct
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
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__Digital, state),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ur_msgs__msg__Digital__rosidl_typesupport_introspection_c__Digital_message_members = {
  "ur_msgs__msg",  // message namespace
  "Digital",  // message name
  2,  // number of fields
  sizeof(ur_msgs__msg__Digital),
  ur_msgs__msg__Digital__rosidl_typesupport_introspection_c__Digital_message_member_array,  // message members
  ur_msgs__msg__Digital__rosidl_typesupport_introspection_c__Digital_init_function,  // function to initialize message memory (memory has to be allocated)
  ur_msgs__msg__Digital__rosidl_typesupport_introspection_c__Digital_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ur_msgs__msg__Digital__rosidl_typesupport_introspection_c__Digital_message_type_support_handle = {
  0,
  &ur_msgs__msg__Digital__rosidl_typesupport_introspection_c__Digital_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ur_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ur_msgs, msg, Digital)() {
  if (!ur_msgs__msg__Digital__rosidl_typesupport_introspection_c__Digital_message_type_support_handle.typesupport_identifier) {
    ur_msgs__msg__Digital__rosidl_typesupport_introspection_c__Digital_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ur_msgs__msg__Digital__rosidl_typesupport_introspection_c__Digital_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
