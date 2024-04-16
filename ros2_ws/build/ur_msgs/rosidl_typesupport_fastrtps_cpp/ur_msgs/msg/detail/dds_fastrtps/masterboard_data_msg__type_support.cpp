// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from ur_msgs:msg/MasterboardDataMsg.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/masterboard_data_msg__rosidl_typesupport_fastrtps_cpp.hpp"
#include "ur_msgs/msg/detail/masterboard_data_msg__struct.hpp"

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
  const ur_msgs::msg::MasterboardDataMsg & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: digital_input_bits
  cdr << ros_message.digital_input_bits;
  // Member: digital_output_bits
  cdr << ros_message.digital_output_bits;
  // Member: analog_input_range0
  cdr << ros_message.analog_input_range0;
  // Member: analog_input_range1
  cdr << ros_message.analog_input_range1;
  // Member: analog_input0
  cdr << ros_message.analog_input0;
  // Member: analog_input1
  cdr << ros_message.analog_input1;
  // Member: analog_output_domain0
  cdr << ros_message.analog_output_domain0;
  // Member: analog_output_domain1
  cdr << ros_message.analog_output_domain1;
  // Member: analog_output0
  cdr << ros_message.analog_output0;
  // Member: analog_output1
  cdr << ros_message.analog_output1;
  // Member: masterboard_temperature
  cdr << ros_message.masterboard_temperature;
  // Member: robot_voltage_48v
  cdr << ros_message.robot_voltage_48v;
  // Member: robot_current
  cdr << ros_message.robot_current;
  // Member: master_io_current
  cdr << ros_message.master_io_current;
  // Member: master_safety_state
  cdr << ros_message.master_safety_state;
  // Member: master_onoff_state
  cdr << ros_message.master_onoff_state;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ur_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  ur_msgs::msg::MasterboardDataMsg & ros_message)
{
  // Member: digital_input_bits
  cdr >> ros_message.digital_input_bits;

  // Member: digital_output_bits
  cdr >> ros_message.digital_output_bits;

  // Member: analog_input_range0
  cdr >> ros_message.analog_input_range0;

  // Member: analog_input_range1
  cdr >> ros_message.analog_input_range1;

  // Member: analog_input0
  cdr >> ros_message.analog_input0;

  // Member: analog_input1
  cdr >> ros_message.analog_input1;

  // Member: analog_output_domain0
  cdr >> ros_message.analog_output_domain0;

  // Member: analog_output_domain1
  cdr >> ros_message.analog_output_domain1;

  // Member: analog_output0
  cdr >> ros_message.analog_output0;

  // Member: analog_output1
  cdr >> ros_message.analog_output1;

  // Member: masterboard_temperature
  cdr >> ros_message.masterboard_temperature;

  // Member: robot_voltage_48v
  cdr >> ros_message.robot_voltage_48v;

  // Member: robot_current
  cdr >> ros_message.robot_current;

  // Member: master_io_current
  cdr >> ros_message.master_io_current;

  // Member: master_safety_state
  cdr >> ros_message.master_safety_state;

  // Member: master_onoff_state
  cdr >> ros_message.master_onoff_state;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ur_msgs
get_serialized_size(
  const ur_msgs::msg::MasterboardDataMsg & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: digital_input_bits
  {
    size_t item_size = sizeof(ros_message.digital_input_bits);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: digital_output_bits
  {
    size_t item_size = sizeof(ros_message.digital_output_bits);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: analog_input_range0
  {
    size_t item_size = sizeof(ros_message.analog_input_range0);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: analog_input_range1
  {
    size_t item_size = sizeof(ros_message.analog_input_range1);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: analog_input0
  {
    size_t item_size = sizeof(ros_message.analog_input0);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: analog_input1
  {
    size_t item_size = sizeof(ros_message.analog_input1);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: analog_output_domain0
  {
    size_t item_size = sizeof(ros_message.analog_output_domain0);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: analog_output_domain1
  {
    size_t item_size = sizeof(ros_message.analog_output_domain1);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: analog_output0
  {
    size_t item_size = sizeof(ros_message.analog_output0);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: analog_output1
  {
    size_t item_size = sizeof(ros_message.analog_output1);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: masterboard_temperature
  {
    size_t item_size = sizeof(ros_message.masterboard_temperature);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: robot_voltage_48v
  {
    size_t item_size = sizeof(ros_message.robot_voltage_48v);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: robot_current
  {
    size_t item_size = sizeof(ros_message.robot_current);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: master_io_current
  {
    size_t item_size = sizeof(ros_message.master_io_current);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: master_safety_state
  {
    size_t item_size = sizeof(ros_message.master_safety_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: master_onoff_state
  {
    size_t item_size = sizeof(ros_message.master_onoff_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_ur_msgs
max_serialized_size_MasterboardDataMsg(
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


  // Member: digital_input_bits
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: digital_output_bits
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: analog_input_range0
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: analog_input_range1
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: analog_input0
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: analog_input1
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: analog_output_domain0
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: analog_output_domain1
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: analog_output0
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: analog_output1
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: masterboard_temperature
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: robot_voltage_48v
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: robot_current
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: master_io_current
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: master_safety_state
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: master_onoff_state
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
    using DataType = ur_msgs::msg::MasterboardDataMsg;
    is_plain =
      (
      offsetof(DataType, master_onoff_state) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _MasterboardDataMsg__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const ur_msgs::msg::MasterboardDataMsg *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _MasterboardDataMsg__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<ur_msgs::msg::MasterboardDataMsg *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _MasterboardDataMsg__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const ur_msgs::msg::MasterboardDataMsg *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _MasterboardDataMsg__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_MasterboardDataMsg(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _MasterboardDataMsg__callbacks = {
  "ur_msgs::msg",
  "MasterboardDataMsg",
  _MasterboardDataMsg__cdr_serialize,
  _MasterboardDataMsg__cdr_deserialize,
  _MasterboardDataMsg__get_serialized_size,
  _MasterboardDataMsg__max_serialized_size
};

static rosidl_message_type_support_t _MasterboardDataMsg__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_MasterboardDataMsg__callbacks,
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
get_message_type_support_handle<ur_msgs::msg::MasterboardDataMsg>()
{
  return &ur_msgs::msg::typesupport_fastrtps_cpp::_MasterboardDataMsg__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, ur_msgs, msg, MasterboardDataMsg)() {
  return &ur_msgs::msg::typesupport_fastrtps_cpp::_MasterboardDataMsg__handle;
}

#ifdef __cplusplus
}
#endif
