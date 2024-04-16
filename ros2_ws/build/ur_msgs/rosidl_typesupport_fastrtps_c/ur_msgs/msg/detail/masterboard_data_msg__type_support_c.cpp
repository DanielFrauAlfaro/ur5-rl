// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from ur_msgs:msg/MasterboardDataMsg.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/masterboard_data_msg__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "ur_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "ur_msgs/msg/detail/masterboard_data_msg__struct.h"
#include "ur_msgs/msg/detail/masterboard_data_msg__functions.h"
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


using _MasterboardDataMsg__ros_msg_type = ur_msgs__msg__MasterboardDataMsg;

static bool _MasterboardDataMsg__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _MasterboardDataMsg__ros_msg_type * ros_message = static_cast<const _MasterboardDataMsg__ros_msg_type *>(untyped_ros_message);
  // Field name: digital_input_bits
  {
    cdr << ros_message->digital_input_bits;
  }

  // Field name: digital_output_bits
  {
    cdr << ros_message->digital_output_bits;
  }

  // Field name: analog_input_range0
  {
    cdr << ros_message->analog_input_range0;
  }

  // Field name: analog_input_range1
  {
    cdr << ros_message->analog_input_range1;
  }

  // Field name: analog_input0
  {
    cdr << ros_message->analog_input0;
  }

  // Field name: analog_input1
  {
    cdr << ros_message->analog_input1;
  }

  // Field name: analog_output_domain0
  {
    cdr << ros_message->analog_output_domain0;
  }

  // Field name: analog_output_domain1
  {
    cdr << ros_message->analog_output_domain1;
  }

  // Field name: analog_output0
  {
    cdr << ros_message->analog_output0;
  }

  // Field name: analog_output1
  {
    cdr << ros_message->analog_output1;
  }

  // Field name: masterboard_temperature
  {
    cdr << ros_message->masterboard_temperature;
  }

  // Field name: robot_voltage_48v
  {
    cdr << ros_message->robot_voltage_48v;
  }

  // Field name: robot_current
  {
    cdr << ros_message->robot_current;
  }

  // Field name: master_io_current
  {
    cdr << ros_message->master_io_current;
  }

  // Field name: master_safety_state
  {
    cdr << ros_message->master_safety_state;
  }

  // Field name: master_onoff_state
  {
    cdr << ros_message->master_onoff_state;
  }

  return true;
}

static bool _MasterboardDataMsg__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _MasterboardDataMsg__ros_msg_type * ros_message = static_cast<_MasterboardDataMsg__ros_msg_type *>(untyped_ros_message);
  // Field name: digital_input_bits
  {
    cdr >> ros_message->digital_input_bits;
  }

  // Field name: digital_output_bits
  {
    cdr >> ros_message->digital_output_bits;
  }

  // Field name: analog_input_range0
  {
    cdr >> ros_message->analog_input_range0;
  }

  // Field name: analog_input_range1
  {
    cdr >> ros_message->analog_input_range1;
  }

  // Field name: analog_input0
  {
    cdr >> ros_message->analog_input0;
  }

  // Field name: analog_input1
  {
    cdr >> ros_message->analog_input1;
  }

  // Field name: analog_output_domain0
  {
    cdr >> ros_message->analog_output_domain0;
  }

  // Field name: analog_output_domain1
  {
    cdr >> ros_message->analog_output_domain1;
  }

  // Field name: analog_output0
  {
    cdr >> ros_message->analog_output0;
  }

  // Field name: analog_output1
  {
    cdr >> ros_message->analog_output1;
  }

  // Field name: masterboard_temperature
  {
    cdr >> ros_message->masterboard_temperature;
  }

  // Field name: robot_voltage_48v
  {
    cdr >> ros_message->robot_voltage_48v;
  }

  // Field name: robot_current
  {
    cdr >> ros_message->robot_current;
  }

  // Field name: master_io_current
  {
    cdr >> ros_message->master_io_current;
  }

  // Field name: master_safety_state
  {
    cdr >> ros_message->master_safety_state;
  }

  // Field name: master_onoff_state
  {
    cdr >> ros_message->master_onoff_state;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_ur_msgs
size_t get_serialized_size_ur_msgs__msg__MasterboardDataMsg(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _MasterboardDataMsg__ros_msg_type * ros_message = static_cast<const _MasterboardDataMsg__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name digital_input_bits
  {
    size_t item_size = sizeof(ros_message->digital_input_bits);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name digital_output_bits
  {
    size_t item_size = sizeof(ros_message->digital_output_bits);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name analog_input_range0
  {
    size_t item_size = sizeof(ros_message->analog_input_range0);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name analog_input_range1
  {
    size_t item_size = sizeof(ros_message->analog_input_range1);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name analog_input0
  {
    size_t item_size = sizeof(ros_message->analog_input0);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name analog_input1
  {
    size_t item_size = sizeof(ros_message->analog_input1);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name analog_output_domain0
  {
    size_t item_size = sizeof(ros_message->analog_output_domain0);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name analog_output_domain1
  {
    size_t item_size = sizeof(ros_message->analog_output_domain1);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name analog_output0
  {
    size_t item_size = sizeof(ros_message->analog_output0);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name analog_output1
  {
    size_t item_size = sizeof(ros_message->analog_output1);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name masterboard_temperature
  {
    size_t item_size = sizeof(ros_message->masterboard_temperature);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name robot_voltage_48v
  {
    size_t item_size = sizeof(ros_message->robot_voltage_48v);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name robot_current
  {
    size_t item_size = sizeof(ros_message->robot_current);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name master_io_current
  {
    size_t item_size = sizeof(ros_message->master_io_current);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name master_safety_state
  {
    size_t item_size = sizeof(ros_message->master_safety_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name master_onoff_state
  {
    size_t item_size = sizeof(ros_message->master_onoff_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _MasterboardDataMsg__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_ur_msgs__msg__MasterboardDataMsg(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_ur_msgs
size_t max_serialized_size_ur_msgs__msg__MasterboardDataMsg(
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

  // member: digital_input_bits
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: digital_output_bits
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: analog_input_range0
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: analog_input_range1
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: analog_input0
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: analog_input1
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: analog_output_domain0
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: analog_output_domain1
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: analog_output0
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: analog_output1
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: masterboard_temperature
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: robot_voltage_48v
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: robot_current
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: master_io_current
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: master_safety_state
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: master_onoff_state
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
    using DataType = ur_msgs__msg__MasterboardDataMsg;
    is_plain =
      (
      offsetof(DataType, master_onoff_state) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _MasterboardDataMsg__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_ur_msgs__msg__MasterboardDataMsg(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_MasterboardDataMsg = {
  "ur_msgs::msg",
  "MasterboardDataMsg",
  _MasterboardDataMsg__cdr_serialize,
  _MasterboardDataMsg__cdr_deserialize,
  _MasterboardDataMsg__get_serialized_size,
  _MasterboardDataMsg__max_serialized_size
};

static rosidl_message_type_support_t _MasterboardDataMsg__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_MasterboardDataMsg,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, ur_msgs, msg, MasterboardDataMsg)() {
  return &_MasterboardDataMsg__type_support;
}

#if defined(__cplusplus)
}
#endif
