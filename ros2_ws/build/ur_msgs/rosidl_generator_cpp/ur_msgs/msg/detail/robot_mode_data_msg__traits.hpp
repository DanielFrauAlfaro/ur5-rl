// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ur_msgs:msg/RobotModeDataMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__ROBOT_MODE_DATA_MSG__TRAITS_HPP_
#define UR_MSGS__MSG__DETAIL__ROBOT_MODE_DATA_MSG__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ur_msgs/msg/detail/robot_mode_data_msg__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace ur_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const RobotModeDataMsg & msg,
  std::ostream & out)
{
  out << "{";
  // member: timestamp
  {
    out << "timestamp: ";
    rosidl_generator_traits::value_to_yaml(msg.timestamp, out);
    out << ", ";
  }

  // member: is_robot_connected
  {
    out << "is_robot_connected: ";
    rosidl_generator_traits::value_to_yaml(msg.is_robot_connected, out);
    out << ", ";
  }

  // member: is_real_robot_enabled
  {
    out << "is_real_robot_enabled: ";
    rosidl_generator_traits::value_to_yaml(msg.is_real_robot_enabled, out);
    out << ", ";
  }

  // member: is_power_on_robot
  {
    out << "is_power_on_robot: ";
    rosidl_generator_traits::value_to_yaml(msg.is_power_on_robot, out);
    out << ", ";
  }

  // member: is_emergency_stopped
  {
    out << "is_emergency_stopped: ";
    rosidl_generator_traits::value_to_yaml(msg.is_emergency_stopped, out);
    out << ", ";
  }

  // member: is_protective_stopped
  {
    out << "is_protective_stopped: ";
    rosidl_generator_traits::value_to_yaml(msg.is_protective_stopped, out);
    out << ", ";
  }

  // member: is_program_running
  {
    out << "is_program_running: ";
    rosidl_generator_traits::value_to_yaml(msg.is_program_running, out);
    out << ", ";
  }

  // member: is_program_paused
  {
    out << "is_program_paused: ";
    rosidl_generator_traits::value_to_yaml(msg.is_program_paused, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RobotModeDataMsg & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: timestamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "timestamp: ";
    rosidl_generator_traits::value_to_yaml(msg.timestamp, out);
    out << "\n";
  }

  // member: is_robot_connected
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_robot_connected: ";
    rosidl_generator_traits::value_to_yaml(msg.is_robot_connected, out);
    out << "\n";
  }

  // member: is_real_robot_enabled
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_real_robot_enabled: ";
    rosidl_generator_traits::value_to_yaml(msg.is_real_robot_enabled, out);
    out << "\n";
  }

  // member: is_power_on_robot
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_power_on_robot: ";
    rosidl_generator_traits::value_to_yaml(msg.is_power_on_robot, out);
    out << "\n";
  }

  // member: is_emergency_stopped
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_emergency_stopped: ";
    rosidl_generator_traits::value_to_yaml(msg.is_emergency_stopped, out);
    out << "\n";
  }

  // member: is_protective_stopped
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_protective_stopped: ";
    rosidl_generator_traits::value_to_yaml(msg.is_protective_stopped, out);
    out << "\n";
  }

  // member: is_program_running
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_program_running: ";
    rosidl_generator_traits::value_to_yaml(msg.is_program_running, out);
    out << "\n";
  }

  // member: is_program_paused
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_program_paused: ";
    rosidl_generator_traits::value_to_yaml(msg.is_program_paused, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RobotModeDataMsg & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace ur_msgs

namespace rosidl_generator_traits
{

[[deprecated("use ur_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const ur_msgs::msg::RobotModeDataMsg & msg,
  std::ostream & out, size_t indentation = 0)
{
  ur_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ur_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ur_msgs::msg::RobotModeDataMsg & msg)
{
  return ur_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ur_msgs::msg::RobotModeDataMsg>()
{
  return "ur_msgs::msg::RobotModeDataMsg";
}

template<>
inline const char * name<ur_msgs::msg::RobotModeDataMsg>()
{
  return "ur_msgs/msg/RobotModeDataMsg";
}

template<>
struct has_fixed_size<ur_msgs::msg::RobotModeDataMsg>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<ur_msgs::msg::RobotModeDataMsg>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<ur_msgs::msg::RobotModeDataMsg>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // UR_MSGS__MSG__DETAIL__ROBOT_MODE_DATA_MSG__TRAITS_HPP_
