// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ur_msgs:msg/RobotModeDataMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__ROBOT_MODE_DATA_MSG__STRUCT_H_
#define UR_MSGS__MSG__DETAIL__ROBOT_MODE_DATA_MSG__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/RobotModeDataMsg in the package ur_msgs.
/**
  * This data structure contains the RobotModeData structure
  * used by the Universal Robots controller
  *
  * This data structure is send at 10 Hz on TCP port 30002
  *
  * Note: this message does not carry all fields from the RobotModeData structure as broadcast by the robot controller, but a subset.
 */
typedef struct ur_msgs__msg__RobotModeDataMsg
{
  uint64_t timestamp;
  bool is_robot_connected;
  bool is_real_robot_enabled;
  bool is_power_on_robot;
  bool is_emergency_stopped;
  bool is_protective_stopped;
  bool is_program_running;
  bool is_program_paused;
} ur_msgs__msg__RobotModeDataMsg;

// Struct for a sequence of ur_msgs__msg__RobotModeDataMsg.
typedef struct ur_msgs__msg__RobotModeDataMsg__Sequence
{
  ur_msgs__msg__RobotModeDataMsg * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ur_msgs__msg__RobotModeDataMsg__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // UR_MSGS__MSG__DETAIL__ROBOT_MODE_DATA_MSG__STRUCT_H_
