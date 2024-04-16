// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from ur_msgs:msg/RobotModeDataMsg.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/robot_mode_data_msg__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "ur_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "ur_msgs/msg/detail/robot_mode_data_msg__struct.h"
#include "ur_msgs/msg/detail/robot_mode_data_msg__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif


// forward declare type support functions


using _RobotModeDataMsg__ros_msg_type = ur_msgs__msg__RobotModeDataMsg;

static bool _RobotModeDataMsg__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _RobotModeDataMsg__ros_msg_type * ros_message = static_cast<const _RobotModeDataMsg__ros_msg_type *>(untyped_ros_message);
  // Field name: timestamp
  {
    cdr << ros_message->timestamp;
  }

  // Field name: is_robot_connected
  {
    cdr << (ros_message->is_robot_connected ? true : false);
  }

  // Field name: is_real_robot_enabled
  {
    cdr << (ros_message->is_real_robot_enabled ? true : false);
  }

  // Field name: is_power_on_robot
  {
    cdr << (ros_message->is_power_on_robot ? true : false);
  }

  // Field name: is_emergency_stopped
  {
    cdr << (ros_message->is_emergency_stopped ? true : false);
  }

  // Field name: is_protective_stopped
  {
    cdr << (ros_message->is_protective_stopped ? true : false);
  }

  // Field name: is_program_running
  {
    cdr << (ros_message->is_program_running ? true : false);
  }

  // Field name: is_program_paused
  {
    cdr << (ros_message->is_program_paused ? true : false);
  }

  return true;
}

static bool _RobotModeDataMsg__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _RobotModeDataMsg__ros_msg_type * ros_message = static_cast<_RobotModeDataMsg__ros_msg_type *>(untyped_ros_message);
  // Field name: timestamp
  {
    cdr >> ros_message->timestamp;
  }

  // Field name: is_robot_connected
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->is_robot_connected = tmp ? true : false;
  }

  // Field name: is_real_robot_enabled
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->is_real_robot_enabled = tmp ? true : false;
  }

  // Field name: is_power_on_robot
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->is_power_on_robot = tmp ? true : false;
  }

  // Field name: is_emergency_stopped
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->is_emergency_stopped = tmp ? true : false;
  }

  // Field name: is_protective_stopped
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->is_protective_stopped = tmp ? true : false;
  }

  // Field name: is_program_running
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->is_program_running = tmp ? true : false;
  }

  // Field name: is_program_paused
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->is_program_paused = tmp ? true : false;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_ur_msgs
size_t get_serialized_size_ur_msgs__msg__RobotModeDataMsg(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _RobotModeDataMsg__ros_msg_type * ros_message = static_cast<const _RobotModeDataMsg__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name timestamp
  {
    size_t item_size = sizeof(ros_message->timestamp);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name is_robot_connected
  {
    size_t item_size = sizeof(ros_message->is_robot_connected);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name is_real_robot_enabled
  {
    size_t item_size = sizeof(ros_message->is_real_robot_enabled);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name is_power_on_robot
  {
    size_t item_size = sizeof(ros_message->is_power_on_robot);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name is_emergency_stopped
  {
    size_t item_size = sizeof(ros_message->is_emergency_stopped);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name is_protective_stopped
  {
    size_t item_size = sizeof(ros_message->is_protective_stopped);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name is_program_running
  {
    size_t item_size = sizeof(ros_message->is_program_running);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name is_program_paused
  {
    size_t item_size = sizeof(ros_message->is_program_paused);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _RobotModeDataMsg__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_ur_msgs__msg__RobotModeDataMsg(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_ur_msgs
size_t max_serialized_size_ur_msgs__msg__RobotModeDataMsg(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: timestamp
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: is_robot_connected
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: is_real_robot_enabled
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: is_power_on_robot
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: is_emergency_stopped
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: is_protective_stopped
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: is_program_running
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: is_program_paused
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = ur_msgs__msg__RobotModeDataMsg;
    is_plain =
      (
      offsetof(DataType, is_program_paused) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _RobotModeDataMsg__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_ur_msgs__msg__RobotModeDataMsg(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_RobotModeDataMsg = {
  "ur_msgs::msg",
  "RobotModeDataMsg",
  _RobotModeDataMsg__cdr_serialize,
  _RobotModeDataMsg__cdr_deserialize,
  _RobotModeDataMsg__get_serialized_size,
  _RobotModeDataMsg__max_serialized_size
};

static rosidl_message_type_support_t _RobotModeDataMsg__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_RobotModeDataMsg,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, ur_msgs, msg, RobotModeDataMsg)() {
  return &_RobotModeDataMsg__type_support;
}

#if defined(__cplusplus)
}
#endif
