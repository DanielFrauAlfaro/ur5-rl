// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ur_msgs:msg/RobotStateRTMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__ROBOT_STATE_RT_MSG__BUILDER_HPP_
#define UR_MSGS__MSG__DETAIL__ROBOT_STATE_RT_MSG__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ur_msgs/msg/detail/robot_state_rt_msg__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ur_msgs
{

namespace msg
{

namespace builder
{

class Init_RobotStateRTMsg_joint_modes
{
public:
  explicit Init_RobotStateRTMsg_joint_modes(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  ::ur_msgs::msg::RobotStateRTMsg joint_modes(::ur_msgs::msg::RobotStateRTMsg::_joint_modes_type arg)
  {
    msg_.joint_modes = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_robot_mode
{
public:
  explicit Init_RobotStateRTMsg_robot_mode(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_joint_modes robot_mode(::ur_msgs::msg::RobotStateRTMsg::_robot_mode_type arg)
  {
    msg_.robot_mode = std::move(arg);
    return Init_RobotStateRTMsg_joint_modes(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_test_value
{
public:
  explicit Init_RobotStateRTMsg_test_value(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_robot_mode test_value(::ur_msgs::msg::RobotStateRTMsg::_test_value_type arg)
  {
    msg_.test_value = std::move(arg);
    return Init_RobotStateRTMsg_robot_mode(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_controller_timer
{
public:
  explicit Init_RobotStateRTMsg_controller_timer(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_test_value controller_timer(::ur_msgs::msg::RobotStateRTMsg::_controller_timer_type arg)
  {
    msg_.controller_timer = std::move(arg);
    return Init_RobotStateRTMsg_test_value(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_motor_temperatures
{
public:
  explicit Init_RobotStateRTMsg_motor_temperatures(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_controller_timer motor_temperatures(::ur_msgs::msg::RobotStateRTMsg::_motor_temperatures_type arg)
  {
    msg_.motor_temperatures = std::move(arg);
    return Init_RobotStateRTMsg_controller_timer(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_digital_input_bits
{
public:
  explicit Init_RobotStateRTMsg_digital_input_bits(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_motor_temperatures digital_input_bits(::ur_msgs::msg::RobotStateRTMsg::_digital_input_bits_type arg)
  {
    msg_.digital_input_bits = std::move(arg);
    return Init_RobotStateRTMsg_motor_temperatures(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_tcp_speed
{
public:
  explicit Init_RobotStateRTMsg_tcp_speed(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_digital_input_bits tcp_speed(::ur_msgs::msg::RobotStateRTMsg::_tcp_speed_type arg)
  {
    msg_.tcp_speed = std::move(arg);
    return Init_RobotStateRTMsg_digital_input_bits(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_tool_vector
{
public:
  explicit Init_RobotStateRTMsg_tool_vector(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_tcp_speed tool_vector(::ur_msgs::msg::RobotStateRTMsg::_tool_vector_type arg)
  {
    msg_.tool_vector = std::move(arg);
    return Init_RobotStateRTMsg_tcp_speed(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_tcp_force
{
public:
  explicit Init_RobotStateRTMsg_tcp_force(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_tool_vector tcp_force(::ur_msgs::msg::RobotStateRTMsg::_tcp_force_type arg)
  {
    msg_.tcp_force = std::move(arg);
    return Init_RobotStateRTMsg_tool_vector(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_tool_acc_values
{
public:
  explicit Init_RobotStateRTMsg_tool_acc_values(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_tcp_force tool_acc_values(::ur_msgs::msg::RobotStateRTMsg::_tool_acc_values_type arg)
  {
    msg_.tool_acc_values = std::move(arg);
    return Init_RobotStateRTMsg_tcp_force(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_i_actual
{
public:
  explicit Init_RobotStateRTMsg_i_actual(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_tool_acc_values i_actual(::ur_msgs::msg::RobotStateRTMsg::_i_actual_type arg)
  {
    msg_.i_actual = std::move(arg);
    return Init_RobotStateRTMsg_tool_acc_values(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_qd_actual
{
public:
  explicit Init_RobotStateRTMsg_qd_actual(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_i_actual qd_actual(::ur_msgs::msg::RobotStateRTMsg::_qd_actual_type arg)
  {
    msg_.qd_actual = std::move(arg);
    return Init_RobotStateRTMsg_i_actual(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_q_actual
{
public:
  explicit Init_RobotStateRTMsg_q_actual(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_qd_actual q_actual(::ur_msgs::msg::RobotStateRTMsg::_q_actual_type arg)
  {
    msg_.q_actual = std::move(arg);
    return Init_RobotStateRTMsg_qd_actual(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_m_target
{
public:
  explicit Init_RobotStateRTMsg_m_target(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_q_actual m_target(::ur_msgs::msg::RobotStateRTMsg::_m_target_type arg)
  {
    msg_.m_target = std::move(arg);
    return Init_RobotStateRTMsg_q_actual(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_i_target
{
public:
  explicit Init_RobotStateRTMsg_i_target(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_m_target i_target(::ur_msgs::msg::RobotStateRTMsg::_i_target_type arg)
  {
    msg_.i_target = std::move(arg);
    return Init_RobotStateRTMsg_m_target(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_qdd_target
{
public:
  explicit Init_RobotStateRTMsg_qdd_target(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_i_target qdd_target(::ur_msgs::msg::RobotStateRTMsg::_qdd_target_type arg)
  {
    msg_.qdd_target = std::move(arg);
    return Init_RobotStateRTMsg_i_target(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_qd_target
{
public:
  explicit Init_RobotStateRTMsg_qd_target(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_qdd_target qd_target(::ur_msgs::msg::RobotStateRTMsg::_qd_target_type arg)
  {
    msg_.qd_target = std::move(arg);
    return Init_RobotStateRTMsg_qdd_target(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_q_target
{
public:
  explicit Init_RobotStateRTMsg_q_target(::ur_msgs::msg::RobotStateRTMsg & msg)
  : msg_(msg)
  {}
  Init_RobotStateRTMsg_qd_target q_target(::ur_msgs::msg::RobotStateRTMsg::_q_target_type arg)
  {
    msg_.q_target = std::move(arg);
    return Init_RobotStateRTMsg_qd_target(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

class Init_RobotStateRTMsg_time
{
public:
  Init_RobotStateRTMsg_time()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RobotStateRTMsg_q_target time(::ur_msgs::msg::RobotStateRTMsg::_time_type arg)
  {
    msg_.time = std::move(arg);
    return Init_RobotStateRTMsg_q_target(msg_);
  }

private:
  ::ur_msgs::msg::RobotStateRTMsg msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ur_msgs::msg::RobotStateRTMsg>()
{
  return ur_msgs::msg::builder::Init_RobotStateRTMsg_time();
}

}  // namespace ur_msgs

#endif  // UR_MSGS__MSG__DETAIL__ROBOT_STATE_RT_MSG__BUILDER_HPP_
