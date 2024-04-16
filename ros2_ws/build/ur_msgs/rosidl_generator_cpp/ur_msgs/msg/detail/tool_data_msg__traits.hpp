// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ur_msgs:msg/ToolDataMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__TOOL_DATA_MSG__TRAITS_HPP_
#define UR_MSGS__MSG__DETAIL__TOOL_DATA_MSG__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ur_msgs/msg/detail/tool_data_msg__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace ur_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ToolDataMsg & msg,
  std::ostream & out)
{
  out << "{";
  // member: analog_input_range2
  {
    out << "analog_input_range2: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input_range2, out);
    out << ", ";
  }

  // member: analog_input_range3
  {
    out << "analog_input_range3: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input_range3, out);
    out << ", ";
  }

  // member: analog_input2
  {
    out << "analog_input2: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input2, out);
    out << ", ";
  }

  // member: analog_input3
  {
    out << "analog_input3: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input3, out);
    out << ", ";
  }

  // member: tool_voltage_48v
  {
    out << "tool_voltage_48v: ";
    rosidl_generator_traits::value_to_yaml(msg.tool_voltage_48v, out);
    out << ", ";
  }

  // member: tool_output_voltage
  {
    out << "tool_output_voltage: ";
    rosidl_generator_traits::value_to_yaml(msg.tool_output_voltage, out);
    out << ", ";
  }

  // member: tool_current
  {
    out << "tool_current: ";
    rosidl_generator_traits::value_to_yaml(msg.tool_current, out);
    out << ", ";
  }

  // member: tool_temperature
  {
    out << "tool_temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.tool_temperature, out);
    out << ", ";
  }

  // member: tool_mode
  {
    out << "tool_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.tool_mode, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ToolDataMsg & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: analog_input_range2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "analog_input_range2: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input_range2, out);
    out << "\n";
  }

  // member: analog_input_range3
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "analog_input_range3: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input_range3, out);
    out << "\n";
  }

  // member: analog_input2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "analog_input2: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input2, out);
    out << "\n";
  }

  // member: analog_input3
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "analog_input3: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input3, out);
    out << "\n";
  }

  // member: tool_voltage_48v
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tool_voltage_48v: ";
    rosidl_generator_traits::value_to_yaml(msg.tool_voltage_48v, out);
    out << "\n";
  }

  // member: tool_output_voltage
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tool_output_voltage: ";
    rosidl_generator_traits::value_to_yaml(msg.tool_output_voltage, out);
    out << "\n";
  }

  // member: tool_current
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tool_current: ";
    rosidl_generator_traits::value_to_yaml(msg.tool_current, out);
    out << "\n";
  }

  // member: tool_temperature
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tool_temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.tool_temperature, out);
    out << "\n";
  }

  // member: tool_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tool_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.tool_mode, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ToolDataMsg & msg, bool use_flow_style = false)
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
  const ur_msgs::msg::ToolDataMsg & msg,
  std::ostream & out, size_t indentation = 0)
{
  ur_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ur_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ur_msgs::msg::ToolDataMsg & msg)
{
  return ur_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ur_msgs::msg::ToolDataMsg>()
{
  return "ur_msgs::msg::ToolDataMsg";
}

template<>
inline const char * name<ur_msgs::msg::ToolDataMsg>()
{
  return "ur_msgs/msg/ToolDataMsg";
}

template<>
struct has_fixed_size<ur_msgs::msg::ToolDataMsg>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<ur_msgs::msg::ToolDataMsg>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<ur_msgs::msg::ToolDataMsg>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // UR_MSGS__MSG__DETAIL__TOOL_DATA_MSG__TRAITS_HPP_
