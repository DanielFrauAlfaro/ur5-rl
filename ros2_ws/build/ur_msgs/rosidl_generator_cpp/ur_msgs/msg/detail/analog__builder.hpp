// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ur_msgs:msg/Analog.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__ANALOG__BUILDER_HPP_
#define UR_MSGS__MSG__DETAIL__ANALOG__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ur_msgs/msg/detail/analog__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ur_msgs
{

namespace msg
{

namespace builder
{

class Init_Analog_state
{
public:
  explicit Init_Analog_state(::ur_msgs::msg::Analog & msg)
  : msg_(msg)
  {}
  ::ur_msgs::msg::Analog state(::ur_msgs::msg::Analog::_state_type arg)
  {
    msg_.state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ur_msgs::msg::Analog msg_;
};

class Init_Analog_domain
{
public:
  explicit Init_Analog_domain(::ur_msgs::msg::Analog & msg)
  : msg_(msg)
  {}
  Init_Analog_state domain(::ur_msgs::msg::Analog::_domain_type arg)
  {
    msg_.domain = std::move(arg);
    return Init_Analog_state(msg_);
  }

private:
  ::ur_msgs::msg::Analog msg_;
};

class Init_Analog_pin
{
public:
  Init_Analog_pin()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Analog_domain pin(::ur_msgs::msg::Analog::_pin_type arg)
  {
    msg_.pin = std::move(arg);
    return Init_Analog_domain(msg_);
  }

private:
  ::ur_msgs::msg::Analog msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ur_msgs::msg::Analog>()
{
  return ur_msgs::msg::builder::Init_Analog_pin();
}

}  // namespace ur_msgs

#endif  // UR_MSGS__MSG__DETAIL__ANALOG__BUILDER_HPP_
