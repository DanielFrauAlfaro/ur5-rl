// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from ur_msgs:msg/RobotStateRTMsg.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/robot_state_rt_msg__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "ur_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "ur_msgs/msg/detail/robot_state_rt_msg__struct.h"
#include "ur_msgs/msg/detail/robot_state_rt_msg__functions.h"
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

#include "rosidl_runtime_c/primitives_sequence.h"  // i_actual, i_target, joint_modes, m_target, motor_temperatures, q_actual, q_target, qd_actual, qd_target, qdd_target, tcp_force, tcp_speed, tool_acc_values, tool_vector
#include "rosidl_runtime_c/primitives_sequence_functions.h"  // i_actual, i_target, joint_modes, m_target, motor_temperatures, q_actual, q_target, qd_actual, qd_target, qdd_target, tcp_force, tcp_speed, tool_acc_values, tool_vector

// forward declare type support functions


using _RobotStateRTMsg__ros_msg_type = ur_msgs__msg__RobotStateRTMsg;

static bool _RobotStateRTMsg__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _RobotStateRTMsg__ros_msg_type * ros_message = static_cast<const _RobotStateRTMsg__ros_msg_type *>(untyped_ros_message);
  // Field name: time
  {
    cdr << ros_message->time;
  }

  // Field name: q_target
  {
    size_t size = ros_message->q_target.size;
    auto array_ptr = ros_message->q_target.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: qd_target
  {
    size_t size = ros_message->qd_target.size;
    auto array_ptr = ros_message->qd_target.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: qdd_target
  {
    size_t size = ros_message->qdd_target.size;
    auto array_ptr = ros_message->qdd_target.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: i_target
  {
    size_t size = ros_message->i_target.size;
    auto array_ptr = ros_message->i_target.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: m_target
  {
    size_t size = ros_message->m_target.size;
    auto array_ptr = ros_message->m_target.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: q_actual
  {
    size_t size = ros_message->q_actual.size;
    auto array_ptr = ros_message->q_actual.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: qd_actual
  {
    size_t size = ros_message->qd_actual.size;
    auto array_ptr = ros_message->qd_actual.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: i_actual
  {
    size_t size = ros_message->i_actual.size;
    auto array_ptr = ros_message->i_actual.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: tool_acc_values
  {
    size_t size = ros_message->tool_acc_values.size;
    auto array_ptr = ros_message->tool_acc_values.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: tcp_force
  {
    size_t size = ros_message->tcp_force.size;
    auto array_ptr = ros_message->tcp_force.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: tool_vector
  {
    size_t size = ros_message->tool_vector.size;
    auto array_ptr = ros_message->tool_vector.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: tcp_speed
  {
    size_t size = ros_message->tcp_speed.size;
    auto array_ptr = ros_message->tcp_speed.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: digital_input_bits
  {
    cdr << ros_message->digital_input_bits;
  }

  // Field name: motor_temperatures
  {
    size_t size = ros_message->motor_temperatures.size;
    auto array_ptr = ros_message->motor_temperatures.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: controller_timer
  {
    cdr << ros_message->controller_timer;
  }

  // Field name: test_value
  {
    cdr << ros_message->test_value;
  }

  // Field name: robot_mode
  {
    cdr << ros_message->robot_mode;
  }

  // Field name: joint_modes
  {
    size_t size = ros_message->joint_modes.size;
    auto array_ptr = ros_message->joint_modes.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _RobotStateRTMsg__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _RobotStateRTMsg__ros_msg_type * ros_message = static_cast<_RobotStateRTMsg__ros_msg_type *>(untyped_ros_message);
  // Field name: time
  {
    cdr >> ros_message->time;
  }

  // Field name: q_target
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->q_target.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->q_target);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->q_target, size)) {
      fprintf(stderr, "failed to create array for field 'q_target'");
      return false;
    }
    auto array_ptr = ros_message->q_target.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: qd_target
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->qd_target.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->qd_target);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->qd_target, size)) {
      fprintf(stderr, "failed to create array for field 'qd_target'");
      return false;
    }
    auto array_ptr = ros_message->qd_target.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: qdd_target
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->qdd_target.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->qdd_target);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->qdd_target, size)) {
      fprintf(stderr, "failed to create array for field 'qdd_target'");
      return false;
    }
    auto array_ptr = ros_message->qdd_target.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: i_target
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->i_target.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->i_target);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->i_target, size)) {
      fprintf(stderr, "failed to create array for field 'i_target'");
      return false;
    }
    auto array_ptr = ros_message->i_target.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: m_target
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->m_target.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->m_target);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->m_target, size)) {
      fprintf(stderr, "failed to create array for field 'm_target'");
      return false;
    }
    auto array_ptr = ros_message->m_target.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: q_actual
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->q_actual.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->q_actual);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->q_actual, size)) {
      fprintf(stderr, "failed to create array for field 'q_actual'");
      return false;
    }
    auto array_ptr = ros_message->q_actual.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: qd_actual
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->qd_actual.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->qd_actual);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->qd_actual, size)) {
      fprintf(stderr, "failed to create array for field 'qd_actual'");
      return false;
    }
    auto array_ptr = ros_message->qd_actual.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: i_actual
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->i_actual.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->i_actual);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->i_actual, size)) {
      fprintf(stderr, "failed to create array for field 'i_actual'");
      return false;
    }
    auto array_ptr = ros_message->i_actual.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: tool_acc_values
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->tool_acc_values.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->tool_acc_values);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->tool_acc_values, size)) {
      fprintf(stderr, "failed to create array for field 'tool_acc_values'");
      return false;
    }
    auto array_ptr = ros_message->tool_acc_values.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: tcp_force
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->tcp_force.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->tcp_force);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->tcp_force, size)) {
      fprintf(stderr, "failed to create array for field 'tcp_force'");
      return false;
    }
    auto array_ptr = ros_message->tcp_force.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: tool_vector
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->tool_vector.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->tool_vector);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->tool_vector, size)) {
      fprintf(stderr, "failed to create array for field 'tool_vector'");
      return false;
    }
    auto array_ptr = ros_message->tool_vector.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: tcp_speed
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->tcp_speed.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->tcp_speed);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->tcp_speed, size)) {
      fprintf(stderr, "failed to create array for field 'tcp_speed'");
      return false;
    }
    auto array_ptr = ros_message->tcp_speed.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: digital_input_bits
  {
    cdr >> ros_message->digital_input_bits;
  }

  // Field name: motor_temperatures
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->motor_temperatures.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->motor_temperatures);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->motor_temperatures, size)) {
      fprintf(stderr, "failed to create array for field 'motor_temperatures'");
      return false;
    }
    auto array_ptr = ros_message->motor_temperatures.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: controller_timer
  {
    cdr >> ros_message->controller_timer;
  }

  // Field name: test_value
  {
    cdr >> ros_message->test_value;
  }

  // Field name: robot_mode
  {
    cdr >> ros_message->robot_mode;
  }

  // Field name: joint_modes
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->joint_modes.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->joint_modes);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->joint_modes, size)) {
      fprintf(stderr, "failed to create array for field 'joint_modes'");
      return false;
    }
    auto array_ptr = ros_message->joint_modes.data;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_ur_msgs
size_t get_serialized_size_ur_msgs__msg__RobotStateRTMsg(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _RobotStateRTMsg__ros_msg_type * ros_message = static_cast<const _RobotStateRTMsg__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name time
  {
    size_t item_size = sizeof(ros_message->time);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name q_target
  {
    size_t array_size = ros_message->q_target.size;
    auto array_ptr = ros_message->q_target.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name qd_target
  {
    size_t array_size = ros_message->qd_target.size;
    auto array_ptr = ros_message->qd_target.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name qdd_target
  {
    size_t array_size = ros_message->qdd_target.size;
    auto array_ptr = ros_message->qdd_target.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name i_target
  {
    size_t array_size = ros_message->i_target.size;
    auto array_ptr = ros_message->i_target.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name m_target
  {
    size_t array_size = ros_message->m_target.size;
    auto array_ptr = ros_message->m_target.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name q_actual
  {
    size_t array_size = ros_message->q_actual.size;
    auto array_ptr = ros_message->q_actual.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name qd_actual
  {
    size_t array_size = ros_message->qd_actual.size;
    auto array_ptr = ros_message->qd_actual.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name i_actual
  {
    size_t array_size = ros_message->i_actual.size;
    auto array_ptr = ros_message->i_actual.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name tool_acc_values
  {
    size_t array_size = ros_message->tool_acc_values.size;
    auto array_ptr = ros_message->tool_acc_values.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name tcp_force
  {
    size_t array_size = ros_message->tcp_force.size;
    auto array_ptr = ros_message->tcp_force.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name tool_vector
  {
    size_t array_size = ros_message->tool_vector.size;
    auto array_ptr = ros_message->tool_vector.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name tcp_speed
  {
    size_t array_size = ros_message->tcp_speed.size;
    auto array_ptr = ros_message->tcp_speed.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name digital_input_bits
  {
    size_t item_size = sizeof(ros_message->digital_input_bits);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name motor_temperatures
  {
    size_t array_size = ros_message->motor_temperatures.size;
    auto array_ptr = ros_message->motor_temperatures.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name controller_timer
  {
    size_t item_size = sizeof(ros_message->controller_timer);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name test_value
  {
    size_t item_size = sizeof(ros_message->test_value);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name robot_mode
  {
    size_t item_size = sizeof(ros_message->robot_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name joint_modes
  {
    size_t array_size = ros_message->joint_modes.size;
    auto array_ptr = ros_message->joint_modes.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _RobotStateRTMsg__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_ur_msgs__msg__RobotStateRTMsg(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_ur_msgs
size_t max_serialized_size_ur_msgs__msg__RobotStateRTMsg(
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

  // member: time
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: q_target
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: qd_target
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: qdd_target
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: i_target
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: m_target
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: q_actual
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: qd_actual
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: i_actual
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: tool_acc_values
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: tcp_force
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: tool_vector
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: tcp_speed
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: digital_input_bits
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: motor_temperatures
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: controller_timer
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: test_value
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: robot_mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: joint_modes
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = ur_msgs__msg__RobotStateRTMsg;
    is_plain =
      (
      offsetof(DataType, joint_modes) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _RobotStateRTMsg__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_ur_msgs__msg__RobotStateRTMsg(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_RobotStateRTMsg = {
  "ur_msgs::msg",
  "RobotStateRTMsg",
  _RobotStateRTMsg__cdr_serialize,
  _RobotStateRTMsg__cdr_deserialize,
  _RobotStateRTMsg__get_serialized_size,
  _RobotStateRTMsg__max_serialized_size
};

static rosidl_message_type_support_t _RobotStateRTMsg__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_RobotStateRTMsg,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, ur_msgs, msg, RobotStateRTMsg)() {
  return &_RobotStateRTMsg__type_support;
}

#if defined(__cplusplus)
}
#endif
