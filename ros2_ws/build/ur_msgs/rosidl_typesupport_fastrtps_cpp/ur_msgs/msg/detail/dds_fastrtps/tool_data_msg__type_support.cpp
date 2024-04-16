// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from ur_msgs:msg/ToolDataMsg.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/tool_data_msg__rosidl_typesupport_fastrtps_cpp.hpp"
#include "ur_msgs/msg/detail/tool_data_msg__struct.hpp"

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
  const ur_msgs::msg::ToolDataMsg & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: analog_input_range2
  cdr << ros_message.analog_input_range2;
  // Member: analog_input_range3
  cdr << ros_message.analog_input_range3;
  // Member: analog_input2
  cdr << ros_message.analog_input2;
  // Member: analog_input3
  cdr << ros_message.analog_input3;
  // Member: tool_voltage_48v
  cdr << ros_message.tool_voltage_48v;
  // Member: tool_output_voltage
  cdr << ros_message.tool_output_voltage;
  // Member: tool_current
  cdr << ros_message.tool_current;
  // Member: tool_temperature
  cdr << ros_message.tool_temperature;
  // Member: tool_mode
  cdr << ros_message.tool_mode;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ur_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  ur_msgs::msg::ToolDataMsg & ros_message)
{
  // Member: analog_input_range2
  cdr >> ros_message.analog_input_range2;

  // Member: analog_input_range3
  cdr >> ros_message.analog_input_range3;

  // Member: analog_input2
  cdr >> ros_message.analog_input2;

  // Member: analog_input3
  cdr >> ros_message.analog_input3;

  // Member: tool_voltage_48v
  cdr >> ros_message.tool_voltage_48v;

  // Member: tool_output_voltage
  cdr >> ros_message.tool_output_voltage;

  // Member: tool_current
  cdr >> ros_message.tool_current;

  // Member: tool_temperature
  cdr >> ros_message.tool_temperature;

  // Member: tool_mode
  cdr >> ros_message.tool_mode;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ur_msgs
get_serialized_size(
  const ur_msgs::msg::ToolDataMsg & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: analog_input_range2
  {
    size_t item_size = sizeof(ros_message.analog_input_range2);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: analog_input_range3
  {
    size_t item_size = sizeof(ros_message.analog_input_range3);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: analog_input2
  {
    size_t item_size = sizeof(ros_message.analog_input2);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: analog_input3
  {
    size_t item_size = sizeof(ros_message.analog_input3);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: tool_voltage_48v
  {
    size_t item_size = sizeof(ros_message.tool_voltage_48v);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: tool_output_voltage
  {
    size_t item_size = sizeof(ros_message.tool_output_voltage);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: tool_current
  {
    size_t item_size = sizeof(ros_message.tool_current);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: tool_temperature
  {
    size_t item_size = sizeof(ros_message.tool_temperature);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: tool_mode
  {
    size_t item_size = sizeof(ros_message.tool_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ur_msgs
max_serialized_size_ToolDataMsg(
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


  // Member: analog_input_range2
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: analog_input_range3
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: analog_input2
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: analog_input3
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: tool_voltage_48v
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: tool_output_voltage
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: tool_current
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: tool_temperature
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: tool_mode
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
    using DataType = ur_msgs::msg::ToolDataMsg;
    is_plain =
      (
      offsetof(DataType, tool_mode) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _ToolDataMsg__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const ur_msgs::msg::ToolDataMsg *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _ToolDataMsg__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<ur_msgs::msg::ToolDataMsg *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _ToolDataMsg__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const ur_msgs::msg::ToolDataMsg *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _ToolDataMsg__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_ToolDataMsg(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _ToolDataMsg__callbacks = {
  "ur_msgs::msg",
  "ToolDataMsg",
  _ToolDataMsg__cdr_serialize,
  _ToolDataMsg__cdr_deserialize,
  _ToolDataMsg__get_serialized_size,
  _ToolDataMsg__max_serialized_size
};

static rosidl_message_type_support_t _ToolDataMsg__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_ToolDataMsg__callbacks,
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
get_message_type_support_handle<ur_msgs::msg::ToolDataMsg>()
{
  return &ur_msgs::msg::typesupport_fastrtps_cpp::_ToolDataMsg__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, ur_msgs, msg, ToolDataMsg)() {
  return &ur_msgs::msg::typesupport_fastrtps_cpp::_ToolDataMsg__handle;
}

#ifdef __cplusplus
}
#endif
