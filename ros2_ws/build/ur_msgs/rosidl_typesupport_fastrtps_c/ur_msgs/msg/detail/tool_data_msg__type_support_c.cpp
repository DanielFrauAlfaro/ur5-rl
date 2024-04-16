// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from ur_msgs:msg/ToolDataMsg.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/tool_data_msg__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "ur_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "ur_msgs/msg/detail/tool_data_msg__struct.h"
#include "ur_msgs/msg/detail/tool_data_msg__functions.h"
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


using _ToolDataMsg__ros_msg_type = ur_msgs__msg__ToolDataMsg;

static bool _ToolDataMsg__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _ToolDataMsg__ros_msg_type * ros_message = static_cast<const _ToolDataMsg__ros_msg_type *>(untyped_ros_message);
  // Field name: analog_input_range2
  {
    cdr << ros_message->analog_input_range2;
  }

  // Field name: analog_input_range3
  {
    cdr << ros_message->analog_input_range3;
  }

  // Field name: analog_input2
  {
    cdr << ros_message->analog_input2;
  }

  // Field name: analog_input3
  {
    cdr << ros_message->analog_input3;
  }

  // Field name: tool_voltage_48v
  {
    cdr << ros_message->tool_voltage_48v;
  }

  // Field name: tool_output_voltage
  {
    cdr << ros_message->tool_output_voltage;
  }

  // Field name: tool_current
  {
    cdr << ros_message->tool_current;
  }

  // Field name: tool_temperature
  {
    cdr << ros_message->tool_temperature;
  }

  // Field name: tool_mode
  {
    cdr << ros_message->tool_mode;
  }

  return true;
}

static bool _ToolDataMsg__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _ToolDataMsg__ros_msg_type * ros_message = static_cast<_ToolDataMsg__ros_msg_type *>(untyped_ros_message);
  // Field name: analog_input_range2
  {
    cdr >> ros_message->analog_input_range2;
  }

  // Field name: analog_input_range3
  {
    cdr >> ros_message->analog_input_range3;
  }

  // Field name: analog_input2
  {
    cdr >> ros_message->analog_input2;
  }

  // Field name: analog_input3
  {
    cdr >> ros_message->analog_input3;
  }

  // Field name: tool_voltage_48v
  {
    cdr >> ros_message->tool_voltage_48v;
  }

  // Field name: tool_output_voltage
  {
    cdr >> ros_message->tool_output_voltage;
  }

  // Field name: tool_current
  {
    cdr >> ros_message->tool_current;
  }

  // Field name: tool_temperature
  {
    cdr >> ros_message->tool_temperature;
  }

  // Field name: tool_mode
  {
    cdr >> ros_message->tool_mode;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_ur_msgs
size_t get_serialized_size_ur_msgs__msg__ToolDataMsg(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _ToolDataMsg__ros_msg_type * ros_message = static_cast<const _ToolDataMsg__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name analog_input_range2
  {
    size_t item_size = sizeof(ros_message->analog_input_range2);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name analog_input_range3
  {
    size_t item_size = sizeof(ros_message->analog_input_range3);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name analog_input2
  {
    size_t item_size = sizeof(ros_message->analog_input2);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name analog_input3
  {
    size_t item_size = sizeof(ros_message->analog_input3);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name tool_voltage_48v
  {
    size_t item_size = sizeof(ros_message->tool_voltage_48v);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name tool_output_voltage
  {
    size_t item_size = sizeof(ros_message->tool_output_voltage);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name tool_current
  {
    size_t item_size = sizeof(ros_message->tool_current);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name tool_temperature
  {
    size_t item_size = sizeof(ros_message->tool_temperature);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name tool_mode
  {
    size_t item_size = sizeof(ros_message->tool_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _ToolDataMsg__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_ur_msgs__msg__ToolDataMsg(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_ur_msgs
size_t max_serialized_size_ur_msgs__msg__ToolDataMsg(
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

  // member: analog_input_range2
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: analog_input_range3
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: analog_input2
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: analog_input3
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: tool_voltage_48v
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: tool_output_voltage
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: tool_current
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: tool_temperature
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: tool_mode
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
    using DataType = ur_msgs__msg__ToolDataMsg;
    is_plain =
      (
      offsetof(DataType, tool_mode) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _ToolDataMsg__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_ur_msgs__msg__ToolDataMsg(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_ToolDataMsg = {
  "ur_msgs::msg",
  "ToolDataMsg",
  _ToolDataMsg__cdr_serialize,
  _ToolDataMsg__cdr_deserialize,
  _ToolDataMsg__get_serialized_size,
  _ToolDataMsg__max_serialized_size
};

static rosidl_message_type_support_t _ToolDataMsg__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_ToolDataMsg,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, ur_msgs, msg, ToolDataMsg)() {
  return &_ToolDataMsg__type_support;
}

#if defined(__cplusplus)
}
#endif
