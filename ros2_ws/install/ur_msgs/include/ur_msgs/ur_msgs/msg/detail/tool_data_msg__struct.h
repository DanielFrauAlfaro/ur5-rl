// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ur_msgs:msg/ToolDataMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__TOOL_DATA_MSG__STRUCT_H_
#define UR_MSGS__MSG__DETAIL__TOOL_DATA_MSG__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'ANALOG_INPUT_RANGE_CURRENT'.
enum
{
  ur_msgs__msg__ToolDataMsg__ANALOG_INPUT_RANGE_CURRENT = 0
};

/// Constant 'ANALOG_INPUT_RANGE_VOLTAGE'.
enum
{
  ur_msgs__msg__ToolDataMsg__ANALOG_INPUT_RANGE_VOLTAGE = 1
};

/// Constant 'TOOL_BOOTLOADER_MODE'.
enum
{
  ur_msgs__msg__ToolDataMsg__TOOL_BOOTLOADER_MODE = 249
};

/// Constant 'TOOL_RUNNING_MODE'.
enum
{
  ur_msgs__msg__ToolDataMsg__TOOL_RUNNING_MODE = 253
};

/// Constant 'TOOL_IDLE_MODE'.
enum
{
  ur_msgs__msg__ToolDataMsg__TOOL_IDLE_MODE = 255
};

/// Struct defined in msg/ToolDataMsg in the package ur_msgs.
/**
  * This data structure contains the ToolData structure
  * used by the Universal Robots controller
 */
typedef struct ur_msgs__msg__ToolDataMsg
{
  /// one of ANALOG_INPUT_RANGE_*
  int8_t analog_input_range2;
  /// one of ANALOG_INPUT_RANGE_*
  int8_t analog_input_range3;
  double analog_input2;
  double analog_input3;
  float tool_voltage_48v;
  uint8_t tool_output_voltage;
  float tool_current;
  float tool_temperature;
  /// one of TOOL_*
  uint8_t tool_mode;
} ur_msgs__msg__ToolDataMsg;

// Struct for a sequence of ur_msgs__msg__ToolDataMsg.
typedef struct ur_msgs__msg__ToolDataMsg__Sequence
{
  ur_msgs__msg__ToolDataMsg * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ur_msgs__msg__ToolDataMsg__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // UR_MSGS__MSG__DETAIL__TOOL_DATA_MSG__STRUCT_H_
