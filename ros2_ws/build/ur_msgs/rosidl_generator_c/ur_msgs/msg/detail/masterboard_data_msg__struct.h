// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ur_msgs:msg/MasterboardDataMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__MASTERBOARD_DATA_MSG__STRUCT_H_
#define UR_MSGS__MSG__DETAIL__MASTERBOARD_DATA_MSG__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/MasterboardDataMsg in the package ur_msgs.
/**
  * This data structure contains the MasterboardData structure
  * used by the Universal Robots controller
  *
  * MasterboardData is part of the data structure being send on the 
  * secondary client communications interface
  *
  * This data structure is send at 10 Hz on TCP port 30002
  *
  * Documentation can be found on the Universal Robots Support site, article
  * number 16496.
 */
typedef struct ur_msgs__msg__MasterboardDataMsg
{
  uint32_t digital_input_bits;
  uint32_t digital_output_bits;
  int8_t analog_input_range0;
  int8_t analog_input_range1;
  double analog_input0;
  double analog_input1;
  int8_t analog_output_domain0;
  int8_t analog_output_domain1;
  double analog_output0;
  double analog_output1;
  float masterboard_temperature;
  float robot_voltage_48v;
  float robot_current;
  float master_io_current;
  uint8_t master_safety_state;
  uint8_t master_onoff_state;
} ur_msgs__msg__MasterboardDataMsg;

// Struct for a sequence of ur_msgs__msg__MasterboardDataMsg.
typedef struct ur_msgs__msg__MasterboardDataMsg__Sequence
{
  ur_msgs__msg__MasterboardDataMsg * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ur_msgs__msg__MasterboardDataMsg__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // UR_MSGS__MSG__DETAIL__MASTERBOARD_DATA_MSG__STRUCT_H_
