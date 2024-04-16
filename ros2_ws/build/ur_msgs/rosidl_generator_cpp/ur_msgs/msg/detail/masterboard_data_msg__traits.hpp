// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ur_msgs:msg/MasterboardDataMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__MASTERBOARD_DATA_MSG__TRAITS_HPP_
#define UR_MSGS__MSG__DETAIL__MASTERBOARD_DATA_MSG__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ur_msgs/msg/detail/masterboard_data_msg__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace ur_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const MasterboardDataMsg & msg,
  std::ostream & out)
{
  out << "{";
  // member: digital_input_bits
  {
    out << "digital_input_bits: ";
    rosidl_generator_traits::value_to_yaml(msg.digital_input_bits, out);
    out << ", ";
  }

  // member: digital_output_bits
  {
    out << "digital_output_bits: ";
    rosidl_generator_traits::value_to_yaml(msg.digital_output_bits, out);
    out << ", ";
  }

  // member: analog_input_range0
  {
    out << "analog_input_range0: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input_range0, out);
    out << ", ";
  }

  // member: analog_input_range1
  {
    out << "analog_input_range1: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input_range1, out);
    out << ", ";
  }

  // member: analog_input0
  {
    out << "analog_input0: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input0, out);
    out << ", ";
  }

  // member: analog_input1
  {
    out << "analog_input1: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input1, out);
    out << ", ";
  }

  // member: analog_output_domain0
  {
    out << "analog_output_domain0: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_output_domain0, out);
    out << ", ";
  }

  // member: analog_output_domain1
  {
    out << "analog_output_domain1: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_output_domain1, out);
    out << ", ";
  }

  // member: analog_output0
  {
    out << "analog_output0: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_output0, out);
    out << ", ";
  }

  // member: analog_output1
  {
    out << "analog_output1: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_output1, out);
    out << ", ";
  }

  // member: masterboard_temperature
  {
    out << "masterboard_temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.masterboard_temperature, out);
    out << ", ";
  }

  // member: robot_voltage_48v
  {
    out << "robot_voltage_48v: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_voltage_48v, out);
    out << ", ";
  }

  // member: robot_current
  {
    out << "robot_current: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_current, out);
    out << ", ";
  }

  // member: master_io_current
  {
    out << "master_io_current: ";
    rosidl_generator_traits::value_to_yaml(msg.master_io_current, out);
    out << ", ";
  }

  // member: master_safety_state
  {
    out << "master_safety_state: ";
    rosidl_generator_traits::value_to_yaml(msg.master_safety_state, out);
    out << ", ";
  }

  // member: master_onoff_state
  {
    out << "master_onoff_state: ";
    rosidl_generator_traits::value_to_yaml(msg.master_onoff_state, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MasterboardDataMsg & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: digital_input_bits
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "digital_input_bits: ";
    rosidl_generator_traits::value_to_yaml(msg.digital_input_bits, out);
    out << "\n";
  }

  // member: digital_output_bits
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "digital_output_bits: ";
    rosidl_generator_traits::value_to_yaml(msg.digital_output_bits, out);
    out << "\n";
  }

  // member: analog_input_range0
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "analog_input_range0: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input_range0, out);
    out << "\n";
  }

  // member: analog_input_range1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "analog_input_range1: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input_range1, out);
    out << "\n";
  }

  // member: analog_input0
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "analog_input0: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input0, out);
    out << "\n";
  }

  // member: analog_input1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "analog_input1: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_input1, out);
    out << "\n";
  }

  // member: analog_output_domain0
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "analog_output_domain0: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_output_domain0, out);
    out << "\n";
  }

  // member: analog_output_domain1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "analog_output_domain1: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_output_domain1, out);
    out << "\n";
  }

  // member: analog_output0
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "analog_output0: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_output0, out);
    out << "\n";
  }

  // member: analog_output1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "analog_output1: ";
    rosidl_generator_traits::value_to_yaml(msg.analog_output1, out);
    out << "\n";
  }

  // member: masterboard_temperature
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "masterboard_temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.masterboard_temperature, out);
    out << "\n";
  }

  // member: robot_voltage_48v
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "robot_voltage_48v: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_voltage_48v, out);
    out << "\n";
  }

  // member: robot_current
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "robot_current: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_current, out);
    out << "\n";
  }

  // member: master_io_current
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "master_io_current: ";
    rosidl_generator_traits::value_to_yaml(msg.master_io_current, out);
    out << "\n";
  }

  // member: master_safety_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "master_safety_state: ";
    rosidl_generator_traits::value_to_yaml(msg.master_safety_state, out);
    out << "\n";
  }

  // member: master_onoff_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "master_onoff_state: ";
    rosidl_generator_traits::value_to_yaml(msg.master_onoff_state, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MasterboardDataMsg & msg, bool use_flow_style = false)
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
  const ur_msgs::msg::MasterboardDataMsg & msg,
  std::ostream & out, size_t indentation = 0)
{
  ur_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ur_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ur_msgs::msg::MasterboardDataMsg & msg)
{
  return ur_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ur_msgs::msg::MasterboardDataMsg>()
{
  return "ur_msgs::msg::MasterboardDataMsg";
}

template<>
inline const char * name<ur_msgs::msg::MasterboardDataMsg>()
{
  return "ur_msgs/msg/MasterboardDataMsg";
}

template<>
struct has_fixed_size<ur_msgs::msg::MasterboardDataMsg>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<ur_msgs::msg::MasterboardDataMsg>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<ur_msgs::msg::MasterboardDataMsg>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // UR_MSGS__MSG__DETAIL__MASTERBOARD_DATA_MSG__TRAITS_HPP_
