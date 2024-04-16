// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from ur_msgs:msg/RobotStateRTMsg.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "ur_msgs/msg/detail/robot_state_rt_msg__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace ur_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void RobotStateRTMsg_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) ur_msgs::msg::RobotStateRTMsg(_init);
}

void RobotStateRTMsg_fini_function(void * message_memory)
{
  auto typed_message = static_cast<ur_msgs::msg::RobotStateRTMsg *>(message_memory);
  typed_message->~RobotStateRTMsg();
}

size_t size_function__RobotStateRTMsg__q_target(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__q_target(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__q_target(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__q_target(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__q_target(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__q_target(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__q_target(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__q_target(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__qd_target(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__qd_target(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__qd_target(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__qd_target(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__qd_target(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__qd_target(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__qd_target(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__qd_target(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__qdd_target(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__qdd_target(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__qdd_target(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__qdd_target(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__qdd_target(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__qdd_target(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__qdd_target(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__qdd_target(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__i_target(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__i_target(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__i_target(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__i_target(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__i_target(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__i_target(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__i_target(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__i_target(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__m_target(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__m_target(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__m_target(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__m_target(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__m_target(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__m_target(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__m_target(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__m_target(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__q_actual(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__q_actual(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__q_actual(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__q_actual(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__q_actual(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__q_actual(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__q_actual(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__q_actual(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__qd_actual(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__qd_actual(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__qd_actual(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__qd_actual(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__qd_actual(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__qd_actual(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__qd_actual(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__qd_actual(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__i_actual(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__i_actual(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__i_actual(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__i_actual(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__i_actual(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__i_actual(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__i_actual(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__i_actual(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__tool_acc_values(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__tool_acc_values(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__tool_acc_values(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__tool_acc_values(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__tool_acc_values(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__tool_acc_values(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__tool_acc_values(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__tool_acc_values(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__tcp_force(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__tcp_force(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__tcp_force(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__tcp_force(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__tcp_force(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__tcp_force(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__tcp_force(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__tcp_force(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__tool_vector(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__tool_vector(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__tool_vector(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__tool_vector(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__tool_vector(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__tool_vector(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__tool_vector(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__tool_vector(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__tcp_speed(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__tcp_speed(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__tcp_speed(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__tcp_speed(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__tcp_speed(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__tcp_speed(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__tcp_speed(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__tcp_speed(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__motor_temperatures(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__motor_temperatures(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__motor_temperatures(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__motor_temperatures(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__motor_temperatures(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__motor_temperatures(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__motor_temperatures(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__motor_temperatures(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__RobotStateRTMsg__joint_modes(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__RobotStateRTMsg__joint_modes(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotStateRTMsg__joint_modes(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotStateRTMsg__joint_modes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotStateRTMsg__joint_modes(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotStateRTMsg__joint_modes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotStateRTMsg__joint_modes(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__RobotStateRTMsg__joint_modes(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember RobotStateRTMsg_message_member_array[19] = {
  {
    "time",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, time),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "q_target",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, q_target),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__q_target,  // size() function pointer
    get_const_function__RobotStateRTMsg__q_target,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__q_target,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__q_target,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__q_target,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__q_target  // resize(index) function pointer
  },
  {
    "qd_target",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, qd_target),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__qd_target,  // size() function pointer
    get_const_function__RobotStateRTMsg__qd_target,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__qd_target,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__qd_target,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__qd_target,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__qd_target  // resize(index) function pointer
  },
  {
    "qdd_target",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, qdd_target),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__qdd_target,  // size() function pointer
    get_const_function__RobotStateRTMsg__qdd_target,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__qdd_target,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__qdd_target,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__qdd_target,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__qdd_target  // resize(index) function pointer
  },
  {
    "i_target",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, i_target),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__i_target,  // size() function pointer
    get_const_function__RobotStateRTMsg__i_target,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__i_target,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__i_target,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__i_target,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__i_target  // resize(index) function pointer
  },
  {
    "m_target",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, m_target),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__m_target,  // size() function pointer
    get_const_function__RobotStateRTMsg__m_target,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__m_target,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__m_target,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__m_target,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__m_target  // resize(index) function pointer
  },
  {
    "q_actual",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, q_actual),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__q_actual,  // size() function pointer
    get_const_function__RobotStateRTMsg__q_actual,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__q_actual,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__q_actual,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__q_actual,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__q_actual  // resize(index) function pointer
  },
  {
    "qd_actual",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, qd_actual),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__qd_actual,  // size() function pointer
    get_const_function__RobotStateRTMsg__qd_actual,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__qd_actual,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__qd_actual,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__qd_actual,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__qd_actual  // resize(index) function pointer
  },
  {
    "i_actual",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, i_actual),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__i_actual,  // size() function pointer
    get_const_function__RobotStateRTMsg__i_actual,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__i_actual,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__i_actual,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__i_actual,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__i_actual  // resize(index) function pointer
  },
  {
    "tool_acc_values",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, tool_acc_values),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__tool_acc_values,  // size() function pointer
    get_const_function__RobotStateRTMsg__tool_acc_values,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__tool_acc_values,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__tool_acc_values,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__tool_acc_values,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__tool_acc_values  // resize(index) function pointer
  },
  {
    "tcp_force",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, tcp_force),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__tcp_force,  // size() function pointer
    get_const_function__RobotStateRTMsg__tcp_force,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__tcp_force,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__tcp_force,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__tcp_force,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__tcp_force  // resize(index) function pointer
  },
  {
    "tool_vector",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, tool_vector),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__tool_vector,  // size() function pointer
    get_const_function__RobotStateRTMsg__tool_vector,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__tool_vector,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__tool_vector,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__tool_vector,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__tool_vector  // resize(index) function pointer
  },
  {
    "tcp_speed",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, tcp_speed),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__tcp_speed,  // size() function pointer
    get_const_function__RobotStateRTMsg__tcp_speed,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__tcp_speed,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__tcp_speed,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__tcp_speed,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__tcp_speed  // resize(index) function pointer
  },
  {
    "digital_input_bits",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, digital_input_bits),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "motor_temperatures",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, motor_temperatures),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__motor_temperatures,  // size() function pointer
    get_const_function__RobotStateRTMsg__motor_temperatures,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__motor_temperatures,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__motor_temperatures,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__motor_temperatures,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__motor_temperatures  // resize(index) function pointer
  },
  {
    "controller_timer",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, controller_timer),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "test_value",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, test_value),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "robot_mode",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, robot_mode),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "joint_modes",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::RobotStateRTMsg, joint_modes),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotStateRTMsg__joint_modes,  // size() function pointer
    get_const_function__RobotStateRTMsg__joint_modes,  // get_const(index) function pointer
    get_function__RobotStateRTMsg__joint_modes,  // get(index) function pointer
    fetch_function__RobotStateRTMsg__joint_modes,  // fetch(index, &value) function pointer
    assign_function__RobotStateRTMsg__joint_modes,  // assign(index, value) function pointer
    resize_function__RobotStateRTMsg__joint_modes  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers RobotStateRTMsg_message_members = {
  "ur_msgs::msg",  // message namespace
  "RobotStateRTMsg",  // message name
  19,  // number of fields
  sizeof(ur_msgs::msg::RobotStateRTMsg),
  RobotStateRTMsg_message_member_array,  // message members
  RobotStateRTMsg_init_function,  // function to initialize message memory (memory has to be allocated)
  RobotStateRTMsg_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t RobotStateRTMsg_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &RobotStateRTMsg_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace ur_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<ur_msgs::msg::RobotStateRTMsg>()
{
  return &::ur_msgs::msg::rosidl_typesupport_introspection_cpp::RobotStateRTMsg_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, ur_msgs, msg, RobotStateRTMsg)() {
  return &::ur_msgs::msg::rosidl_typesupport_introspection_cpp::RobotStateRTMsg_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
