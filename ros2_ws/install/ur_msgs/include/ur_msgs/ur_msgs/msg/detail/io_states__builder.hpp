// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ur_msgs:msg/IOStates.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__IO_STATES__BUILDER_HPP_
#define UR_MSGS__MSG__DETAIL__IO_STATES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ur_msgs/msg/detail/io_states__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ur_msgs
{

namespace msg
{

namespace builder
{

class Init_IOStates_analog_out_states
{
public:
  explicit Init_IOStates_analog_out_states(::ur_msgs::msg::IOStates & msg)
  : msg_(msg)
  {}
  ::ur_msgs::msg::IOStates analog_out_states(::ur_msgs::msg::IOStates::_analog_out_states_type arg)
  {
    msg_.analog_out_states = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ur_msgs::msg::IOStates msg_;
};

class Init_IOStates_analog_in_states
{
public:
  explicit Init_IOStates_analog_in_states(::ur_msgs::msg::IOStates & msg)
  : msg_(msg)
  {}
  Init_IOStates_analog_out_states analog_in_states(::ur_msgs::msg::IOStates::_analog_in_states_type arg)
  {
    msg_.analog_in_states = std::move(arg);
    return Init_IOStates_analog_out_states(msg_);
  }

private:
  ::ur_msgs::msg::IOStates msg_;
};

class Init_IOStates_flag_states
{
public:
  explicit Init_IOStates_flag_states(::ur_msgs::msg::IOStates & msg)
  : msg_(msg)
  {}
  Init_IOStates_analog_in_states flag_states(::ur_msgs::msg::IOStates::_flag_states_type arg)
  {
    msg_.flag_states = std::move(arg);
    return Init_IOStates_analog_in_states(msg_);
  }

private:
  ::ur_msgs::msg::IOStates msg_;
};

class Init_IOStates_digital_out_states
{
public:
  explicit Init_IOStates_digital_out_states(::ur_msgs::msg::IOStates & msg)
  : msg_(msg)
  {}
  Init_IOStates_flag_states digital_out_states(::ur_msgs::msg::IOStates::_digital_out_states_type arg)
  {
    msg_.digital_out_states = std::move(arg);
    return Init_IOStates_flag_states(msg_);
  }

private:
  ::ur_msgs::msg::IOStates msg_;
};

class Init_IOStates_digital_in_states
{
public:
  Init_IOStates_digital_in_states()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_IOStates_digital_out_states digital_in_states(::ur_msgs::msg::IOStates::_digital_in_states_type arg)
  {
    msg_.digital_in_states = std::move(arg);
    return Init_IOStates_digital_out_states(msg_);
  }

private:
  ::ur_msgs::msg::IOStates msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ur_msgs::msg::IOStates>()
{
  return ur_msgs::msg::builder::Init_IOStates_digital_in_states();
}

}  // namespace ur_msgs

#endif  // UR_MSGS__MSG__DETAIL__IO_STATES__BUILDER_HPP_
