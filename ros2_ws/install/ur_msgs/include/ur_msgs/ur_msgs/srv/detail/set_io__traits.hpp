// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ur_msgs:srv/SetIO.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__SRV__DETAIL__SET_IO__TRAITS_HPP_
#define UR_MSGS__SRV__DETAIL__SET_IO__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ur_msgs/srv/detail/set_io__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace ur_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetIO_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: fun
  {
    out << "fun: ";
    rosidl_generator_traits::value_to_yaml(msg.fun, out);
    out << ", ";
  }

  // member: pin
  {
    out << "pin: ";
    rosidl_generator_traits::value_to_yaml(msg.pin, out);
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
  const SetIO_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: fun
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "fun: ";
    rosidl_generator_traits::value_to_yaml(msg.fun, out);
    out << "\n";
  }

  // member: pin
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pin: ";
    rosidl_generator_traits::value_to_yaml(msg.pin, out);
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

inline std::string to_yaml(const SetIO_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace ur_msgs

namespace rosidl_generator_traits
{

[[deprecated("use ur_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const ur_msgs::srv::SetIO_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  ur_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ur_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const ur_msgs::srv::SetIO_Request & msg)
{
  return ur_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<ur_msgs::srv::SetIO_Request>()
{
  return "ur_msgs::srv::SetIO_Request";
}

template<>
inline const char * name<ur_msgs::srv::SetIO_Request>()
{
  return "ur_msgs/srv/SetIO_Request";
}

template<>
struct has_fixed_size<ur_msgs::srv::SetIO_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<ur_msgs::srv::SetIO_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<ur_msgs::srv::SetIO_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace ur_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetIO_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetIO_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetIO_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace ur_msgs

namespace rosidl_generator_traits
{

[[deprecated("use ur_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const ur_msgs::srv::SetIO_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  ur_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ur_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const ur_msgs::srv::SetIO_Response & msg)
{
  return ur_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<ur_msgs::srv::SetIO_Response>()
{
  return "ur_msgs::srv::SetIO_Response";
}

template<>
inline const char * name<ur_msgs::srv::SetIO_Response>()
{
  return "ur_msgs/srv/SetIO_Response";
}

template<>
struct has_fixed_size<ur_msgs::srv::SetIO_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<ur_msgs::srv::SetIO_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<ur_msgs::srv::SetIO_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<ur_msgs::srv::SetIO>()
{
  return "ur_msgs::srv::SetIO";
}

template<>
inline const char * name<ur_msgs::srv::SetIO>()
{
  return "ur_msgs/srv/SetIO";
}

template<>
struct has_fixed_size<ur_msgs::srv::SetIO>
  : std::integral_constant<
    bool,
    has_fixed_size<ur_msgs::srv::SetIO_Request>::value &&
    has_fixed_size<ur_msgs::srv::SetIO_Response>::value
  >
{
};

template<>
struct has_bounded_size<ur_msgs::srv::SetIO>
  : std::integral_constant<
    bool,
    has_bounded_size<ur_msgs::srv::SetIO_Request>::value &&
    has_bounded_size<ur_msgs::srv::SetIO_Response>::value
  >
{
};

template<>
struct is_service<ur_msgs::srv::SetIO>
  : std::true_type
{
};

template<>
struct is_service_request<ur_msgs::srv::SetIO_Request>
  : std::true_type
{
};

template<>
struct is_service_response<ur_msgs::srv::SetIO_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // UR_MSGS__SRV__DETAIL__SET_IO__TRAITS_HPP_
