// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ur_msgs:msg/Digital.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__DIGITAL__STRUCT_H_
#define UR_MSGS__MSG__DETAIL__DIGITAL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Digital in the package ur_msgs.
typedef struct ur_msgs__msg__Digital
{
  uint8_t pin;
  bool state;
} ur_msgs__msg__Digital;

// Struct for a sequence of ur_msgs__msg__Digital.
typedef struct ur_msgs__msg__Digital__Sequence
{
  ur_msgs__msg__Digital * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ur_msgs__msg__Digital__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // UR_MSGS__MSG__DETAIL__DIGITAL__STRUCT_H_
