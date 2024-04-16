// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ur_msgs:msg/Analog.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__ANALOG__TRAITS_HPP_
#define UR_MSGS__MSG__DETAIL__ANALOG__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ur_msgs/msg/detail/analog__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace ur_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Analog & msg,
  std::ostream & out)
{
  out << "{";
  // member: pin
  {
    out << "pin: ";
    rosidl_generator_traits::value_to_yaml(msg.pin, out);
    out << ", ";
  }

  // member: domain
  {
    out << "domain: ";
    rosidl_generator_traits::value_to_yaml(msg.domain, out);
    out << ", ";
  }

  // member: state
  {
    out << "state: ";
    rosidl_generator_traits::value_to_yaml(msg.state, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Analog & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: pin
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pin: ";
    rosidl_generator_traits::value_to_yaml(msg.pin, out);
    out << "\n";
  }

  // member: domain
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "domain: ";
    rosidl_generator_traits::value_to_yaml(msg.domain, out);
    out << "\n";
  }

  // member: state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "state: ";
    rosidl_generator_traits::value_to_yaml(msg.state, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Analog & msg, bool use_flow_style = false)
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
  const ur_msgs::msg::Analog & msg,
  std::ostream & out, size_t indentation = 0)
{
  ur_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ur_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ur_msgs::msg::Analog & msg)
{
  return ur_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ur_msgs::msg::Analog>()
{
  return "ur_msgs::msg::Analog";
}

template<>
inline const char * name<ur_msgs::msg::Analog>()
{
  return "ur_msgs/msg/Analog";
}

template<>
struct has_fixed_size<ur_msgs::msg::Analog>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<ur_msgs::msg::Analog>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<ur_msgs::msg::Analog>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // UR_MSGS__MSG__DETAIL__ANALOG__TRAITS_HPP_
