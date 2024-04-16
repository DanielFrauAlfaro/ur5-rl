// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from ur_msgs:msg/RobotStateRTMsg.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "ur_msgs/msg/detail/robot_state_rt_msg__rosidl_typesupport_introspection_c.h"
#include "ur_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "ur_msgs/msg/detail/robot_state_rt_msg__functions.h"
#include "ur_msgs/msg/detail/robot_state_rt_msg__struct.h"


// Include directives for member types
// Member `q_target`
// Member `qd_target`
// Member `qdd_target`
// Member `i_target`
// Member `m_target`
// Member `q_actual`
// Member `qd_actual`
// Member `i_actual`
// Member `tool_acc_values`
// Member `tcp_force`
// Member `tool_vector`
// Member `tcp_speed`
// Member `motor_temperatures`
// Member `joint_modes`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__RobotStateRTMsg_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ur_msgs__msg__RobotStateRTMsg__init(message_memory);
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__RobotStateRTMsg_fini_function(void * message_memory)
{
  ur_msgs__msg__RobotStateRTMsg__fini(message_memory);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__q_target(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__q_target(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__q_target(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__q_target(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__q_target(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__q_target(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__q_target(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__q_target(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__qd_target(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__qd_target(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__qd_target(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__qd_target(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__qd_target(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__qd_target(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__qd_target(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__qd_target(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__qdd_target(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__qdd_target(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__qdd_target(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__qdd_target(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__qdd_target(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__qdd_target(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__qdd_target(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__qdd_target(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__i_target(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__i_target(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__i_target(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__i_target(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__i_target(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__i_target(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__i_target(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__i_target(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__m_target(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__m_target(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__m_target(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__m_target(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__m_target(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__m_target(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__m_target(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__m_target(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__q_actual(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__q_actual(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__q_actual(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__q_actual(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__q_actual(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__q_actual(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__q_actual(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__q_actual(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__qd_actual(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__qd_actual(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__qd_actual(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__qd_actual(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__qd_actual(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__qd_actual(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__qd_actual(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__qd_actual(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__i_actual(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__i_actual(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__i_actual(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__i_actual(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__i_actual(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__i_actual(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__i_actual(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__i_actual(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__tool_acc_values(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__tool_acc_values(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__tool_acc_values(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__tool_acc_values(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__tool_acc_values(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__tool_acc_values(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__tool_acc_values(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__tool_acc_values(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__tcp_force(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__tcp_force(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__tcp_force(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__tcp_force(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__tcp_force(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__tcp_force(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__tcp_force(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__tcp_force(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__tool_vector(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__tool_vector(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__tool_vector(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__tool_vector(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__tool_vector(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__tool_vector(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__tool_vector(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__tool_vector(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__tcp_speed(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__tcp_speed(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__tcp_speed(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__tcp_speed(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__tcp_speed(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__tcp_speed(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__tcp_speed(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__tcp_speed(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__motor_temperatures(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__motor_temperatures(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__motor_temperatures(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__motor_temperatures(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__motor_temperatures(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__motor_temperatures(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__motor_temperatures(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__motor_temperatures(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__joint_modes(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__joint_modes(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__joint_modes(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__joint_modes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__joint_modes(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__joint_modes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__joint_modes(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__joint_modes(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__RobotStateRTMsg_message_member_array[19] = {
  {
    "time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "q_target",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, q_target),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__q_target,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__q_target,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__q_target,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__q_target,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__q_target,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__q_target  // resize(index) function pointer
  },
  {
    "qd_target",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, qd_target),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__qd_target,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__qd_target,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__qd_target,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__qd_target,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__qd_target,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__qd_target  // resize(index) function pointer
  },
  {
    "qdd_target",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, qdd_target),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__qdd_target,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__qdd_target,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__qdd_target,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__qdd_target,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__qdd_target,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__qdd_target  // resize(index) function pointer
  },
  {
    "i_target",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, i_target),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__i_target,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__i_target,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__i_target,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__i_target,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__i_target,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__i_target  // resize(index) function pointer
  },
  {
    "m_target",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, m_target),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__m_target,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__m_target,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__m_target,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__m_target,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__m_target,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__m_target  // resize(index) function pointer
  },
  {
    "q_actual",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, q_actual),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__q_actual,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__q_actual,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__q_actual,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__q_actual,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__q_actual,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__q_actual  // resize(index) function pointer
  },
  {
    "qd_actual",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, qd_actual),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__qd_actual,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__qd_actual,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__qd_actual,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__qd_actual,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__qd_actual,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__qd_actual  // resize(index) function pointer
  },
  {
    "i_actual",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, i_actual),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__i_actual,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__i_actual,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__i_actual,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__i_actual,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__i_actual,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__i_actual  // resize(index) function pointer
  },
  {
    "tool_acc_values",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, tool_acc_values),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__tool_acc_values,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__tool_acc_values,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__tool_acc_values,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__tool_acc_values,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__tool_acc_values,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__tool_acc_values  // resize(index) function pointer
  },
  {
    "tcp_force",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, tcp_force),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__tcp_force,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__tcp_force,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__tcp_force,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__tcp_force,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__tcp_force,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__tcp_force  // resize(index) function pointer
  },
  {
    "tool_vector",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, tool_vector),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__tool_vector,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__tool_vector,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__tool_vector,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__tool_vector,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__tool_vector,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__tool_vector  // resize(index) function pointer
  },
  {
    "tcp_speed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, tcp_speed),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__tcp_speed,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__tcp_speed,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__tcp_speed,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__tcp_speed,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__tcp_speed,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__tcp_speed  // resize(index) function pointer
  },
  {
    "digital_input_bits",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, digital_input_bits),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "motor_temperatures",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, motor_temperatures),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__motor_temperatures,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__motor_temperatures,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__motor_temperatures,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__motor_temperatures,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__motor_temperatures,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__motor_temperatures  // resize(index) function pointer
  },
  {
    "controller_timer",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, controller_timer),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "test_value",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, test_value),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "robot_mode",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, robot_mode),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "joint_modes",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs__msg__RobotStateRTMsg, joint_modes),  // bytes offset in struct
    NULL,  // default value
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__size_function__RobotStateRTMsg__joint_modes,  // size() function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_const_function__RobotStateRTMsg__joint_modes,  // get_const(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__get_function__RobotStateRTMsg__joint_modes,  // get(index) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__fetch_function__RobotStateRTMsg__joint_modes,  // fetch(index, &value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__assign_function__RobotStateRTMsg__joint_modes,  // assign(index, value) function pointer
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__resize_function__RobotStateRTMsg__joint_modes  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__RobotStateRTMsg_message_members = {
  "ur_msgs__msg",  // message namespace
  "RobotStateRTMsg",  // message name
  19,  // number of fields
  sizeof(ur_msgs__msg__RobotStateRTMsg),
  ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__RobotStateRTMsg_message_member_array,  // message members
  ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__RobotStateRTMsg_init_function,  // function to initialize message memory (memory has to be allocated)
  ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__RobotStateRTMsg_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__RobotStateRTMsg_message_type_support_handle = {
  0,
  &ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__RobotStateRTMsg_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ur_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ur_msgs, msg, RobotStateRTMsg)() {
  if (!ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__RobotStateRTMsg_message_type_support_handle.typesupport_identifier) {
    ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__RobotStateRTMsg_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ur_msgs__msg__RobotStateRTMsg__rosidl_typesupport_introspection_c__RobotStateRTMsg_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
