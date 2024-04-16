// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ur_msgs:msg/IOStates.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__IO_STATES__STRUCT_H_
#define UR_MSGS__MSG__DETAIL__IO_STATES__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'digital_in_states'
// Member 'digital_out_states'
// Member 'flag_states'
#include "ur_msgs/msg/detail/digital__struct.h"
// Member 'analog_in_states'
// Member 'analog_out_states'
#include "ur_msgs/msg/detail/analog__struct.h"

/// Struct defined in msg/IOStates in the package ur_msgs.
typedef struct ur_msgs__msg__IOStates
{
  ur_msgs__msg__Digital__Sequence digital_in_states;
  ur_msgs__msg__Digital__Sequence digital_out_states;
  ur_msgs__msg__Digital__Sequence flag_states;
  ur_msgs__msg__Analog__Sequence analog_in_states;
  ur_msgs__msg__Analog__Sequence analog_out_states;
} ur_msgs__msg__IOStates;

// Struct for a sequence of ur_msgs__msg__IOStates.
typedef struct ur_msgs__msg__IOStates__Sequence
{
  ur_msgs__msg__IOStates * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ur_msgs__msg__IOStates__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // UR_MSGS__MSG__DETAIL__IO_STATES__STRUCT_H_
