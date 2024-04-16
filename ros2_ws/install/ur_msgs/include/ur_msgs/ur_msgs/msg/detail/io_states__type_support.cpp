// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from ur_msgs:msg/IOStates.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "ur_msgs/msg/detail/io_states__struct.hpp"
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

void IOStates_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) ur_msgs::msg::IOStates(_init);
}

void IOStates_fini_function(void * message_memory)
{
  auto typed_message = static_cast<ur_msgs::msg::IOStates *>(message_memory);
  typed_message->~IOStates();
}

size_t size_function__IOStates__digital_in_states(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<ur_msgs::msg::Digital> *>(untyped_member);
  return member->size();
}

const void * get_const_function__IOStates__digital_in_states(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<ur_msgs::msg::Digital> *>(untyped_member);
  return &member[index];
}

void * get_function__IOStates__digital_in_states(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<ur_msgs::msg::Digital> *>(untyped_member);
  return &member[index];
}

void fetch_function__IOStates__digital_in_states(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const ur_msgs::msg::Digital *>(
    get_const_function__IOStates__digital_in_states(untyped_member, index));
  auto & value = *reinterpret_cast<ur_msgs::msg::Digital *>(untyped_value);
  value = item;
}

void assign_function__IOStates__digital_in_states(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<ur_msgs::msg::Digital *>(
    get_function__IOStates__digital_in_states(untyped_member, index));
  const auto & value = *reinterpret_cast<const ur_msgs::msg::Digital *>(untyped_value);
  item = value;
}

void resize_function__IOStates__digital_in_states(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<ur_msgs::msg::Digital> *>(untyped_member);
  member->resize(size);
}

size_t size_function__IOStates__digital_out_states(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<ur_msgs::msg::Digital> *>(untyped_member);
  return member->size();
}

const void * get_const_function__IOStates__digital_out_states(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<ur_msgs::msg::Digital> *>(untyped_member);
  return &member[index];
}

void * get_function__IOStates__digital_out_states(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<ur_msgs::msg::Digital> *>(untyped_member);
  return &member[index];
}

void fetch_function__IOStates__digital_out_states(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const ur_msgs::msg::Digital *>(
    get_const_function__IOStates__digital_out_states(untyped_member, index));
  auto & value = *reinterpret_cast<ur_msgs::msg::Digital *>(untyped_value);
  value = item;
}

void assign_function__IOStates__digital_out_states(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<ur_msgs::msg::Digital *>(
    get_function__IOStates__digital_out_states(untyped_member, index));
  const auto & value = *reinterpret_cast<const ur_msgs::msg::Digital *>(untyped_value);
  item = value;
}

void resize_function__IOStates__digital_out_states(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<ur_msgs::msg::Digital> *>(untyped_member);
  member->resize(size);
}

size_t size_function__IOStates__flag_states(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<ur_msgs::msg::Digital> *>(untyped_member);
  return member->size();
}

const void * get_const_function__IOStates__flag_states(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<ur_msgs::msg::Digital> *>(untyped_member);
  return &member[index];
}

void * get_function__IOStates__flag_states(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<ur_msgs::msg::Digital> *>(untyped_member);
  return &member[index];
}

void fetch_function__IOStates__flag_states(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const ur_msgs::msg::Digital *>(
    get_const_function__IOStates__flag_states(untyped_member, index));
  auto & value = *reinterpret_cast<ur_msgs::msg::Digital *>(untyped_value);
  value = item;
}

void assign_function__IOStates__flag_states(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<ur_msgs::msg::Digital *>(
    get_function__IOStates__flag_states(untyped_member, index));
  const auto & value = *reinterpret_cast<const ur_msgs::msg::Digital *>(untyped_value);
  item = value;
}

void resize_function__IOStates__flag_states(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<ur_msgs::msg::Digital> *>(untyped_member);
  member->resize(size);
}

size_t size_function__IOStates__analog_in_states(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<ur_msgs::msg::Analog> *>(untyped_member);
  return member->size();
}

const void * get_const_function__IOStates__analog_in_states(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<ur_msgs::msg::Analog> *>(untyped_member);
  return &member[index];
}

void * get_function__IOStates__analog_in_states(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<ur_msgs::msg::Analog> *>(untyped_member);
  return &member[index];
}

void fetch_function__IOStates__analog_in_states(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const ur_msgs::msg::Analog *>(
    get_const_function__IOStates__analog_in_states(untyped_member, index));
  auto & value = *reinterpret_cast<ur_msgs::msg::Analog *>(untyped_value);
  value = item;
}

void assign_function__IOStates__analog_in_states(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<ur_msgs::msg::Analog *>(
    get_function__IOStates__analog_in_states(untyped_member, index));
  const auto & value = *reinterpret_cast<const ur_msgs::msg::Analog *>(untyped_value);
  item = value;
}

void resize_function__IOStates__analog_in_states(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<ur_msgs::msg::Analog> *>(untyped_member);
  member->resize(size);
}

size_t size_function__IOStates__analog_out_states(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<ur_msgs::msg::Analog> *>(untyped_member);
  return member->size();
}

const void * get_const_function__IOStates__analog_out_states(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<ur_msgs::msg::Analog> *>(untyped_member);
  return &member[index];
}

void * get_function__IOStates__analog_out_states(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<ur_msgs::msg::Analog> *>(untyped_member);
  return &member[index];
}

void fetch_function__IOStates__analog_out_states(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const ur_msgs::msg::Analog *>(
    get_const_function__IOStates__analog_out_states(untyped_member, index));
  auto & value = *reinterpret_cast<ur_msgs::msg::Analog *>(untyped_value);
  value = item;
}

void assign_function__IOStates__analog_out_states(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<ur_msgs::msg::Analog *>(
    get_function__IOStates__analog_out_states(untyped_member, index));
  const auto & value = *reinterpret_cast<const ur_msgs::msg::Analog *>(untyped_value);
  item = value;
}

void resize_function__IOStates__analog_out_states(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<ur_msgs::msg::Analog> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember IOStates_message_member_array[5] = {
  {
    "digital_in_states",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<ur_msgs::msg::Digital>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::IOStates, digital_in_states),  // bytes offset in struct
    nullptr,  // default value
    size_function__IOStates__digital_in_states,  // size() function pointer
    get_const_function__IOStates__digital_in_states,  // get_const(index) function pointer
    get_function__IOStates__digital_in_states,  // get(index) function pointer
    fetch_function__IOStates__digital_in_states,  // fetch(index, &value) function pointer
    assign_function__IOStates__digital_in_states,  // assign(index, value) function pointer
    resize_function__IOStates__digital_in_states  // resize(index) function pointer
  },
  {
    "digital_out_states",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<ur_msgs::msg::Digital>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::IOStates, digital_out_states),  // bytes offset in struct
    nullptr,  // default value
    size_function__IOStates__digital_out_states,  // size() function pointer
    get_const_function__IOStates__digital_out_states,  // get_const(index) function pointer
    get_function__IOStates__digital_out_states,  // get(index) function pointer
    fetch_function__IOStates__digital_out_states,  // fetch(index, &value) function pointer
    assign_function__IOStates__digital_out_states,  // assign(index, value) function pointer
    resize_function__IOStates__digital_out_states  // resize(index) function pointer
  },
  {
    "flag_states",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<ur_msgs::msg::Digital>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::IOStates, flag_states),  // bytes offset in struct
    nullptr,  // default value
    size_function__IOStates__flag_states,  // size() function pointer
    get_const_function__IOStates__flag_states,  // get_const(index) function pointer
    get_function__IOStates__flag_states,  // get(index) function pointer
    fetch_function__IOStates__flag_states,  // fetch(index, &value) function pointer
    assign_function__IOStates__flag_states,  // assign(index, value) function pointer
    resize_function__IOStates__flag_states  // resize(index) function pointer
  },
  {
    "analog_in_states",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<ur_msgs::msg::Analog>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::IOStates, analog_in_states),  // bytes offset in struct
    nullptr,  // default value
    size_function__IOStates__analog_in_states,  // size() function pointer
    get_const_function__IOStates__analog_in_states,  // get_const(index) function pointer
    get_function__IOStates__analog_in_states,  // get(index) function pointer
    fetch_function__IOStates__analog_in_states,  // fetch(index, &value) function pointer
    assign_function__IOStates__analog_in_states,  // assign(index, value) function pointer
    resize_function__IOStates__analog_in_states  // resize(index) function pointer
  },
  {
    "analog_out_states",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<ur_msgs::msg::Analog>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ur_msgs::msg::IOStates, analog_out_states),  // bytes offset in struct
    nullptr,  // default value
    size_function__IOStates__analog_out_states,  // size() function pointer
    get_const_function__IOStates__analog_out_states,  // get_const(index) function pointer
    get_function__IOStates__analog_out_states,  // get(index) function pointer
    fetch_function__IOStates__analog_out_states,  // fetch(index, &value) function pointer
    assign_function__IOStates__analog_out_states,  // assign(index, value) function pointer
    resize_function__IOStates__analog_out_states  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers IOStates_message_members = {
  "ur_msgs::msg",  // message namespace
  "IOStates",  // message name
  5,  // number of fields
  sizeof(ur_msgs::msg::IOStates),
  IOStates_message_member_array,  // message members
  IOStates_init_function,  // function to initialize message memory (memory has to be allocated)
  IOStates_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t IOStates_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &IOStates_message_members,
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
get_message_type_support_handle<ur_msgs::msg::IOStates>()
{
  return &::ur_msgs::msg::rosidl_typesupport_introspection_cpp::IOStates_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, ur_msgs, msg, IOStates)() {
  return &::ur_msgs::msg::rosidl_typesupport_introspection_cpp::IOStates_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
