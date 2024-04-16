// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ur_msgs:msg/ToolDataMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__TOOL_DATA_MSG__BUILDER_HPP_
#define UR_MSGS__MSG__DETAIL__TOOL_DATA_MSG__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ur_msgs/msg/detail/tool_data_msg__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ur_msgs
{

namespace msg
{

namespace builder
{

class Init_ToolDataMsg_tool_mode
{
public:
  explicit Init_ToolDataMsg_tool_mode(::ur_msgs::msg::ToolDataMsg & msg)
  : msg_(msg)
  {}
  ::ur_msgs::msg::ToolDataMsg tool_mode(::ur_msgs::msg::ToolDataMsg::_tool_mode_type arg)
  {
    msg_.tool_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ur_msgs::msg::ToolDataMsg msg_;
};

class Init_ToolDataMsg_tool_temperature
{
public:
  explicit Init_ToolDataMsg_tool_temperature(::ur_msgs::msg::ToolDataMsg & msg)
  : msg_(msg)
  {}
  Init_ToolDataMsg_tool_mode tool_temperature(::ur_msgs::msg::ToolDataMsg::_tool_temperature_type arg)
  {
    msg_.tool_temperature = std::move(arg);
    return Init_ToolDataMsg_tool_mode(msg_);
  }

private:
  ::ur_msgs::msg::ToolDataMsg msg_;
};

class Init_ToolDataMsg_tool_current
{
public:
  explicit Init_ToolDataMsg_tool_current(::ur_msgs::msg::ToolDataMsg & msg)
  : msg_(msg)
  {}
  Init_ToolDataMsg_tool_temperature tool_current(::ur_msgs::msg::ToolDataMsg::_tool_current_type arg)
  {
    msg_.tool_current = std::move(arg);
    return Init_ToolDataMsg_tool_temperature(msg_);
  }

private:
  ::ur_msgs::msg::ToolDataMsg msg_;
};

class Init_ToolDataMsg_tool_output_voltage
{
public:
  explicit Init_ToolDataMsg_tool_output_voltage(::ur_msgs::msg::ToolDataMsg & msg)
  : msg_(msg)
  {}
  Init_ToolDataMsg_tool_current tool_output_voltage(::ur_msgs::msg::ToolDataMsg::_tool_output_voltage_type arg)
  {
    msg_.tool_output_voltage = std::move(arg);
    return Init_ToolDataMsg_tool_current(msg_);
  }

private:
  ::ur_msgs::msg::ToolDataMsg msg_;
};

class Init_ToolDataMsg_tool_voltage_48v
{
public:
  explicit Init_ToolDataMsg_tool_voltage_48v(::ur_msgs::msg::ToolDataMsg & msg)
  : msg_(msg)
  {}
  Init_ToolDataMsg_tool_output_voltage tool_voltage_48v(::ur_msgs::msg::ToolDataMsg::_tool_voltage_48v_type arg)
  {
    msg_.tool_voltage_48v = std::move(arg);
    return Init_ToolDataMsg_tool_output_voltage(msg_);
  }

private:
  ::ur_msgs::msg::ToolDataMsg msg_;
};

class Init_ToolDataMsg_analog_input3
{
public:
  explicit Init_ToolDataMsg_analog_input3(::ur_msgs::msg::ToolDataMsg & msg)
  : msg_(msg)
  {}
  Init_ToolDataMsg_tool_voltage_48v analog_input3(::ur_msgs::msg::ToolDataMsg::_analog_input3_type arg)
  {
    msg_.analog_input3 = std::move(arg);
    return Init_ToolDataMsg_tool_voltage_48v(msg_);
  }

private:
  ::ur_msgs::msg::ToolDataMsg msg_;
};

class Init_ToolDataMsg_analog_input2
{
public:
  explicit Init_ToolDataMsg_analog_input2(::ur_msgs::msg::ToolDataMsg & msg)
  : msg_(msg)
  {}
  Init_ToolDataMsg_analog_input3 analog_input2(::ur_msgs::msg::ToolDataMsg::_analog_input2_type arg)
  {
    msg_.analog_input2 = std::move(arg);
    return Init_ToolDataMsg_analog_input3(msg_);
  }

private:
  ::ur_msgs::msg::ToolDataMsg msg_;
};

class Init_ToolDataMsg_analog_input_range3
{
public:
  explicit Init_ToolDataMsg_analog_input_range3(::ur_msgs::msg::ToolDataMsg & msg)
  : msg_(msg)
  {}
  Init_ToolDataMsg_analog_input2 analog_input_range3(::ur_msgs::msg::ToolDataMsg::_analog_input_range3_type arg)
  {
    msg_.analog_input_range3 = std::move(arg);
    return Init_ToolDataMsg_analog_input2(msg_);
  }

private:
  ::ur_msgs::msg::ToolDataMsg msg_;
};

class Init_ToolDataMsg_analog_input_range2
{
public:
  Init_ToolDataMsg_analog_input_range2()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ToolDataMsg_analog_input_range3 analog_input_range2(::ur_msgs::msg::ToolDataMsg::_analog_input_range2_type arg)
  {
    msg_.analog_input_range2 = std::move(arg);
    return Init_ToolDataMsg_analog_input_range3(msg_);
  }

private:
  ::ur_msgs::msg::ToolDataMsg msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ur_msgs::msg::ToolDataMsg>()
{
  return ur_msgs::msg::builder::Init_ToolDataMsg_analog_input_range2();
}

}  // namespace ur_msgs

#endif  // UR_MSGS__MSG__DETAIL__TOOL_DATA_MSG__BUILDER_HPP_
