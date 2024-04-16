// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ur_msgs:msg/Analog.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__ANALOG__STRUCT_H_
#define UR_MSGS__MSG__DETAIL__ANALOG__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'CURRENT'.
enum
{
  ur_msgs__msg__Analog__CURRENT = 0
};

/// Constant 'VOLTAGE'.
enum
{
  ur_msgs__msg__Analog__VOLTAGE = 1
};

/// Struct defined in msg/Analog in the package ur_msgs.
typedef struct ur_msgs__msg__Analog
{
  uint8_t pin;
  /// can be VOLTAGE or CURRENT
  uint8_t domain;
  float state;
} ur_msgs__msg__Analog;

// Struct for a sequence of ur_msgs__msg__Analog.
typedef struct ur_msgs__msg__Analog__Sequence
{
  ur_msgs__msg__Analog * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ur_msgs__msg__Analog__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // UR_MSGS__MSG__DETAIL__ANALOG__STRUCT_H_
