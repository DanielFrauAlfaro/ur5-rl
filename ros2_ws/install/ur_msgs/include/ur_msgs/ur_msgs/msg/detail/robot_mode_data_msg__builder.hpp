// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ur_msgs:msg/RobotModeDataMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__ROBOT_MODE_DATA_MSG__BUILDER_HPP_
#define UR_MSGS__MSG__DETAIL__ROBOT_MODE_DATA_MSG__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ur_msgs/msg/detail/robot_mode_data_msg__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ur_msgs
{

namespace msg
{

namespace builder
{

class Init_RobotModeDataMsg_is_program_paused
{
public:
  explicit Init_RobotModeDataMsg_is_program_paused(::ur_msgs::msg::RobotModeDataMsg & msg)
  : msg_(msg)
  {}
  ::ur_msgs::msg::RobotModeDataMsg is_program_paused(::ur_msgs::msg::RobotModeDataMsg::_is_program_paused_type arg)
  {
    msg_.is_program_paused = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ur_msgs::msg::RobotModeDataMsg msg_;
};

class Init_RobotModeDataMsg_is_program_running
{
public:
  explicit Init_RobotModeDataMsg_is_program_running(::ur_msgs::msg::RobotModeDataMsg & msg)
  : msg_(msg)
  {}
  Init_RobotModeDataMsg_is_program_paused is_program_running(::ur_msgs::msg::RobotModeDataMsg::_is_program_running_type arg)
  {
    msg_.is_program_running = std::move(arg);
    return Init_RobotModeDataMsg_is_program_paused(msg_);
  }

private:
  ::ur_msgs::msg::RobotModeDataMsg msg_;
};

class Init_RobotModeDataMsg_is_protective_stopped
{
public:
  explicit Init_RobotModeDataMsg_is_protective_stopped(::ur_msgs::msg::RobotModeDataMsg & msg)
  : msg_(msg)
  {}
  Init_RobotModeDataMsg_is_program_running is_protective_stopped(::ur_msgs::msg::RobotModeDataMsg::_is_protective_stopped_type arg)
  {
    msg_.is_protective_stopped = std::move(arg);
    return Init_RobotModeDataMsg_is_program_running(msg_);
  }

private:
  ::ur_msgs::msg::RobotModeDataMsg msg_;
};

class Init_RobotModeDataMsg_is_emergency_stopped
{
public:
  explicit Init_RobotModeDataMsg_is_emergency_stopped(::ur_msgs::msg::RobotModeDataMsg & msg)
  : msg_(msg)
  {}
  Init_RobotModeDataMsg_is_protective_stopped is_emergency_stopped(::ur_msgs::msg::RobotModeDataMsg::_is_emergency_stopped_type arg)
  {
    msg_.is_emergency_stopped = std::move(arg);
    return Init_RobotModeDataMsg_is_protective_stopped(msg_);
  }

private:
  ::ur_msgs::msg::RobotModeDataMsg msg_;
};

class Init_RobotModeDataMsg_is_power_on_robot
{
public:
  explicit Init_RobotModeDataMsg_is_power_on_robot(::ur_msgs::msg::RobotModeDataMsg & msg)
  : msg_(msg)
  {}
  Init_RobotModeDataMsg_is_emergency_stopped is_power_on_robot(::ur_msgs::msg::RobotModeDataMsg::_is_power_on_robot_type arg)
  {
    msg_.is_power_on_robot = std::move(arg);
    return Init_RobotModeDataMsg_is_emergency_stopped(msg_);
  }

private:
  ::ur_msgs::msg::RobotModeDataMsg msg_;
};

class Init_RobotModeDataMsg_is_real_robot_enabled
{
public:
  explicit Init_RobotModeDataMsg_is_real_robot_enabled(::ur_msgs::msg::RobotModeDataMsg & msg)
  : msg_(msg)
  {}
  Init_RobotModeDataMsg_is_power_on_robot is_real_robot_enabled(::ur_msgs::msg::RobotModeDataMsg::_is_real_robot_enabled_type arg)
  {
    msg_.is_real_robot_enabled = std::move(arg);
    return Init_RobotModeDataMsg_is_power_on_robot(msg_);
  }

private:
  ::ur_msgs::msg::RobotModeDataMsg msg_;
};

class Init_RobotModeDataMsg_is_robot_connected
{
public:
  explicit Init_RobotModeDataMsg_is_robot_connected(::ur_msgs::msg::RobotModeDataMsg & msg)
  : msg_(msg)
  {}
  Init_RobotModeDataMsg_is_real_robot_enabled is_robot_connected(::ur_msgs::msg::RobotModeDataMsg::_is_robot_connected_type arg)
  {
    msg_.is_robot_connected = std::move(arg);
    return Init_RobotModeDataMsg_is_real_robot_enabled(msg_);
  }

private:
  ::ur_msgs::msg::RobotModeDataMsg msg_;
};

class Init_RobotModeDataMsg_timestamp
{
public:
  Init_RobotModeDataMsg_timestamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RobotModeDataMsg_is_robot_connected timestamp(::ur_msgs::msg::RobotModeDataMsg::_timestamp_type arg)
  {
    msg_.timestamp = std::move(arg);
    return Init_RobotModeDataMsg_is_robot_connected(msg_);
  }

private:
  ::ur_msgs::msg::RobotModeDataMsg msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ur_msgs::msg::RobotModeDataMsg>()
{
  return ur_msgs::msg::builder::Init_RobotModeDataMsg_timestamp();
}

}  // namespace ur_msgs

#endif  // UR_MSGS__MSG__DETAIL__ROBOT_MODE_DATA_MSG__BUILDER_HPP_
