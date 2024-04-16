// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from ur_msgs:msg/RobotStateRTMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__ROBOT_STATE_RT_MSG__TRAITS_HPP_
#define UR_MSGS__MSG__DETAIL__ROBOT_STATE_RT_MSG__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "ur_msgs/msg/detail/robot_state_rt_msg__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace ur_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const RobotStateRTMsg & msg,
  std::ostream & out)
{
  out << "{";
  // member: time
  {
    out << "time: ";
    rosidl_generator_traits::value_to_yaml(msg.time, out);
    out << ", ";
  }

  // member: q_target
  {
    if (msg.q_target.size() == 0) {
      out << "q_target: []";
    } else {
      out << "q_target: [";
      size_t pending_items = msg.q_target.size();
      for (auto item : msg.q_target) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: qd_target
  {
    if (msg.qd_target.size() == 0) {
      out << "qd_target: []";
    } else {
      out << "qd_target: [";
      size_t pending_items = msg.qd_target.size();
      for (auto item : msg.qd_target) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: qdd_target
  {
    if (msg.qdd_target.size() == 0) {
      out << "qdd_target: []";
    } else {
      out << "qdd_target: [";
      size_t pending_items = msg.qdd_target.size();
      for (auto item : msg.qdd_target) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: i_target
  {
    if (msg.i_target.size() == 0) {
      out << "i_target: []";
    } else {
      out << "i_target: [";
      size_t pending_items = msg.i_target.size();
      for (auto item : msg.i_target) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: m_target
  {
    if (msg.m_target.size() == 0) {
      out << "m_target: []";
    } else {
      out << "m_target: [";
      size_t pending_items = msg.m_target.size();
      for (auto item : msg.m_target) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: q_actual
  {
    if (msg.q_actual.size() == 0) {
      out << "q_actual: []";
    } else {
      out << "q_actual: [";
      size_t pending_items = msg.q_actual.size();
      for (auto item : msg.q_actual) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: qd_actual
  {
    if (msg.qd_actual.size() == 0) {
      out << "qd_actual: []";
    } else {
      out << "qd_actual: [";
      size_t pending_items = msg.qd_actual.size();
      for (auto item : msg.qd_actual) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: i_actual
  {
    if (msg.i_actual.size() == 0) {
      out << "i_actual: []";
    } else {
      out << "i_actual: [";
      size_t pending_items = msg.i_actual.size();
      for (auto item : msg.i_actual) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: tool_acc_values
  {
    if (msg.tool_acc_values.size() == 0) {
      out << "tool_acc_values: []";
    } else {
      out << "tool_acc_values: [";
      size_t pending_items = msg.tool_acc_values.size();
      for (auto item : msg.tool_acc_values) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: tcp_force
  {
    if (msg.tcp_force.size() == 0) {
      out << "tcp_force: []";
    } else {
      out << "tcp_force: [";
      size_t pending_items = msg.tcp_force.size();
      for (auto item : msg.tcp_force) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: tool_vector
  {
    if (msg.tool_vector.size() == 0) {
      out << "tool_vector: []";
    } else {
      out << "tool_vector: [";
      size_t pending_items = msg.tool_vector.size();
      for (auto item : msg.tool_vector) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: tcp_speed
  {
    if (msg.tcp_speed.size() == 0) {
      out << "tcp_speed: []";
    } else {
      out << "tcp_speed: [";
      size_t pending_items = msg.tcp_speed.size();
      for (auto item : msg.tcp_speed) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: digital_input_bits
  {
    out << "digital_input_bits: ";
    rosidl_generator_traits::value_to_yaml(msg.digital_input_bits, out);
    out << ", ";
  }

  // member: motor_temperatures
  {
    if (msg.motor_temperatures.size() == 0) {
      out << "motor_temperatures: []";
    } else {
      out << "motor_temperatures: [";
      size_t pending_items = msg.motor_temperatures.size();
      for (auto item : msg.motor_temperatures) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: controller_timer
  {
    out << "controller_timer: ";
    rosidl_generator_traits::value_to_yaml(msg.controller_timer, out);
    out << ", ";
  }

  // member: test_value
  {
    out << "test_value: ";
    rosidl_generator_traits::value_to_yaml(msg.test_value, out);
    out << ", ";
  }

  // member: robot_mode
  {
    out << "robot_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_mode, out);
    out << ", ";
  }

  // member: joint_modes
  {
    if (msg.joint_modes.size() == 0) {
      out << "joint_modes: []";
    } else {
      out << "joint_modes: [";
      size_t pending_items = msg.joint_modes.size();
      for (auto item : msg.joint_modes) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RobotStateRTMsg & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "time: ";
    rosidl_generator_traits::value_to_yaml(msg.time, out);
    out << "\n";
  }

  // member: q_target
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.q_target.size() == 0) {
      out << "q_target: []\n";
    } else {
      out << "q_target:\n";
      for (auto item : msg.q_target) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: qd_target
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.qd_target.size() == 0) {
      out << "qd_target: []\n";
    } else {
      out << "qd_target:\n";
      for (auto item : msg.qd_target) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: qdd_target
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.qdd_target.size() == 0) {
      out << "qdd_target: []\n";
    } else {
      out << "qdd_target:\n";
      for (auto item : msg.qdd_target) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: i_target
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.i_target.size() == 0) {
      out << "i_target: []\n";
    } else {
      out << "i_target:\n";
      for (auto item : msg.i_target) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: m_target
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.m_target.size() == 0) {
      out << "m_target: []\n";
    } else {
      out << "m_target:\n";
      for (auto item : msg.m_target) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: q_actual
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.q_actual.size() == 0) {
      out << "q_actual: []\n";
    } else {
      out << "q_actual:\n";
      for (auto item : msg.q_actual) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: qd_actual
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.qd_actual.size() == 0) {
      out << "qd_actual: []\n";
    } else {
      out << "qd_actual:\n";
      for (auto item : msg.qd_actual) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: i_actual
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.i_actual.size() == 0) {
      out << "i_actual: []\n";
    } else {
      out << "i_actual:\n";
      for (auto item : msg.i_actual) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: tool_acc_values
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.tool_acc_values.size() == 0) {
      out << "tool_acc_values: []\n";
    } else {
      out << "tool_acc_values:\n";
      for (auto item : msg.tool_acc_values) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: tcp_force
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.tcp_force.size() == 0) {
      out << "tcp_force: []\n";
    } else {
      out << "tcp_force:\n";
      for (auto item : msg.tcp_force) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: tool_vector
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.tool_vector.size() == 0) {
      out << "tool_vector: []\n";
    } else {
      out << "tool_vector:\n";
      for (auto item : msg.tool_vector) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: tcp_speed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.tcp_speed.size() == 0) {
      out << "tcp_speed: []\n";
    } else {
      out << "tcp_speed:\n";
      for (auto item : msg.tcp_speed) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: digital_input_bits
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "digital_input_bits: ";
    rosidl_generator_traits::value_to_yaml(msg.digital_input_bits, out);
    out << "\n";
  }

  // member: motor_temperatures
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.motor_temperatures.size() == 0) {
      out << "motor_temperatures: []\n";
    } else {
      out << "motor_temperatures:\n";
      for (auto item : msg.motor_temperatures) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: controller_timer
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "controller_timer: ";
    rosidl_generator_traits::value_to_yaml(msg.controller_timer, out);
    out << "\n";
  }

  // member: test_value
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "test_value: ";
    rosidl_generator_traits::value_to_yaml(msg.test_value, out);
    out << "\n";
  }

  // member: robot_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "robot_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.robot_mode, out);
    out << "\n";
  }

  // member: joint_modes
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_modes.size() == 0) {
      out << "joint_modes: []\n";
    } else {
      out << "joint_modes:\n";
      for (auto item : msg.joint_modes) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RobotStateRTMsg & msg, bool use_flow_style = false)
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
  const ur_msgs::msg::RobotStateRTMsg & msg,
  std::ostream & out, size_t indentation = 0)
{
  ur_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use ur_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const ur_msgs::msg::RobotStateRTMsg & msg)
{
  return ur_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<ur_msgs::msg::RobotStateRTMsg>()
{
  return "ur_msgs::msg::RobotStateRTMsg";
}

template<>
inline const char * name<ur_msgs::msg::RobotStateRTMsg>()
{
  return "ur_msgs/msg/RobotStateRTMsg";
}

template<>
struct has_fixed_size<ur_msgs::msg::RobotStateRTMsg>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<ur_msgs::msg::RobotStateRTMsg>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<ur_msgs::msg::RobotStateRTMsg>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // UR_MSGS__MSG__DETAIL__ROBOT_STATE_RT_MSG__TRAITS_HPP_
