// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from ur_msgs:msg/RobotModeDataMsg.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/robot_mode_data_msg__rosidl_typesupport_fastrtps_cpp.hpp"
#include "ur_msgs/msg/detail/robot_mode_data_msg__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace ur_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ur_msgs
cdr_serialize(
  const ur_msgs::msg::RobotModeDataMsg & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: timestamp
  cdr << ros_message.timestamp;
  // Member: is_robot_connected
  cdr << (ros_message.is_robot_connected ? true : false);
  // Member: is_real_robot_enabled
  cdr << (ros_message.is_real_robot_enabled ? true : false);
  // Member: is_power_on_robot
  cdr << (ros_message.is_power_on_robot ? true : false);
  // Member: is_emergency_stopped
  cdr << (ros_message.is_emergency_stopped ? true : false);
  // Member: is_protective_stopped
  cdr << (ros_message.is_protective_stopped ? true : false);
  // Member: is_program_running
  cdr << (ros_message.is_program_running ? true : false);
  // Member: is_program_paused
  cdr << (ros_message.is_program_paused ? true : false);
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ur_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  ur_msgs::msg::RobotModeDataMsg & ros_message)
{
  // Member: timestamp
  cdr >> ros_message.timestamp;

  // Member: is_robot_connected
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.is_robot_connected = tmp ? true : false;
  }

  // Member: is_real_robot_enabled
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.is_real_robot_enabled = tmp ? true : false;
  }

  // Member: is_power_on_robot
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.is_power_on_robot = tmp ? true : false;
  }

  // Member: is_emergency_stopped
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.is_emergency_stopped = tmp ? true : false;
  }

  // Member: is_protective_stopped
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.is_protective_stopped = tmp ? true : false;
  }

  // Member: is_program_running
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.is_program_running = tmp ? true : false;
  }

  // Member: is_program_paused
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.is_program_paused = tmp ? true : false;
  }

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ur_msgs
get_serialized_size(
  const ur_msgs::msg::RobotModeDataMsg & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: timestamp
  {
    size_t item_size = sizeof(ros_message.timestamp);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: is_robot_connected
  {
    size_t item_size = sizeof(ros_message.is_robot_connected);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: is_real_robot_enabled
  {
    size_t item_size = sizeof(ros_message.is_real_robot_enabled);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: is_power_on_robot
  {
    size_t item_size = sizeof(ros_message.is_power_on_robot);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: is_emergency_stopped
  {
    size_t item_size = sizeof(ros_message.is_emergency_stopped);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: is_protective_stopped
  {
    size_t item_size = sizeof(ros_message.is_protective_stopped);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: is_program_running
  {
    size_t item_size = sizeof(ros_message.is_program_running);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: is_program_paused
  {
    size_t item_size = sizeof(ros_message.is_program_paused);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ur_msgs
max_serialized_size_RobotModeDataMsg(
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


  // Member: timestamp
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: is_robot_connected
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: is_real_robot_enabled
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: is_power_on_robot
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: is_emergency_stopped
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: is_protective_stopped
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: is_program_running
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: is_program_paused
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
    using DataType = ur_msgs::msg::RobotModeDataMsg;
    is_plain =
      (
      offsetof(DataType, is_program_paused) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _RobotModeDataMsg__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const ur_msgs::msg::RobotModeDataMsg *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _RobotModeDataMsg__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<ur_msgs::msg::RobotModeDataMsg *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _RobotModeDataMsg__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const ur_msgs::msg::RobotModeDataMsg *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _RobotModeDataMsg__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_RobotModeDataMsg(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _RobotModeDataMsg__callbacks = {
  "ur_msgs::msg",
  "RobotModeDataMsg",
  _RobotModeDataMsg__cdr_serialize,
  _RobotModeDataMsg__cdr_deserialize,
  _RobotModeDataMsg__get_serialized_size,
  _RobotModeDataMsg__max_serialized_size
};

static rosidl_message_type_support_t _RobotModeDataMsg__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_RobotModeDataMsg__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace ur_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_ur_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<ur_msgs::msg::RobotModeDataMsg>()
{
  return &ur_msgs::msg::typesupport_fastrtps_cpp::_RobotModeDataMsg__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, ur_msgs, msg, RobotModeDataMsg)() {
  return &ur_msgs::msg::typesupport_fastrtps_cpp::_RobotModeDataMsg__handle;
}

#ifdef __cplusplus
}
#endif
