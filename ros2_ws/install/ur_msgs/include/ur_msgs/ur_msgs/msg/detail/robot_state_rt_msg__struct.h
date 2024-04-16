// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ur_msgs:msg/RobotStateRTMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__ROBOT_STATE_RT_MSG__STRUCT_H_
#define UR_MSGS__MSG__DETAIL__ROBOT_STATE_RT_MSG__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'q_target'
// Member 'qd_target'
// Member 'qdd_target'
// Member 'i_target'
// Member 'm_target'
// Member 'q_actual'
// Member 'qd_actual'
// Member 'i_actual'
// Member 'tool_acc_values'
// Member 'tcp_force'
// Member 'tool_vector'
// Member 'tcp_speed'
// Member 'motor_temperatures'
// Member 'joint_modes'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/RobotStateRTMsg in the package ur_msgs.
/**
  * Data structure for the realtime communications interface (aka Matlab interface)
  * used by the Universal Robots controller
  *
  * This data structure is send at 125 Hz on TCP port 30003
  *
  * Dokumentation can be found on the Universal Robots Support Wiki
  * (http://wiki03.lynero.net/Technical/RealTimeClientInterface?rev=9)
 */
typedef struct ur_msgs__msg__RobotStateRTMsg
{
  double time;
  rosidl_runtime_c__double__Sequence q_target;
  rosidl_runtime_c__double__Sequence qd_target;
  rosidl_runtime_c__double__Sequence qdd_target;
  rosidl_runtime_c__double__Sequence i_target;
  rosidl_runtime_c__double__Sequence m_target;
  rosidl_runtime_c__double__Sequence q_actual;
  rosidl_runtime_c__double__Sequence qd_actual;
  rosidl_runtime_c__double__Sequence i_actual;
  rosidl_runtime_c__double__Sequence tool_acc_values;
  rosidl_runtime_c__double__Sequence tcp_force;
  rosidl_runtime_c__double__Sequence tool_vector;
  rosidl_runtime_c__double__Sequence tcp_speed;
  double digital_input_bits;
  rosidl_runtime_c__double__Sequence motor_temperatures;
  double controller_timer;
  double test_value;
  double robot_mode;
  rosidl_runtime_c__double__Sequence joint_modes;
} ur_msgs__msg__RobotStateRTMsg;

// Struct for a sequence of ur_msgs__msg__RobotStateRTMsg.
typedef struct ur_msgs__msg__RobotStateRTMsg__Sequence
{
  ur_msgs__msg__RobotStateRTMsg * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ur_msgs__msg__RobotStateRTMsg__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // UR_MSGS__MSG__DETAIL__ROBOT_STATE_RT_MSG__STRUCT_H_
