// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ur_msgs:msg/ToolDataMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__TOOL_DATA_MSG__STRUCT_HPP_
#define UR_MSGS__MSG__DETAIL__TOOL_DATA_MSG__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ur_msgs__msg__ToolDataMsg __attribute__((deprecated))
#else
# define DEPRECATED__ur_msgs__msg__ToolDataMsg __declspec(deprecated)
#endif

namespace ur_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ToolDataMsg_
{
  using Type = ToolDataMsg_<ContainerAllocator>;

  explicit ToolDataMsg_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->analog_input_range2 = 0;
      this->analog_input_range3 = 0;
      this->analog_input2 = 0.0;
      this->analog_input3 = 0.0;
      this->tool_voltage_48v = 0.0f;
      this->tool_output_voltage = 0;
      this->tool_current = 0.0f;
      this->tool_temperature = 0.0f;
      this->tool_mode = 0;
    }
  }

  explicit ToolDataMsg_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->analog_input_range2 = 0;
      this->analog_input_range3 = 0;
      this->analog_input2 = 0.0;
      this->analog_input3 = 0.0;
      this->tool_voltage_48v = 0.0f;
      this->tool_output_voltage = 0;
      this->tool_current = 0.0f;
      this->tool_temperature = 0.0f;
      this->tool_mode = 0;
    }
  }

  // field types and members
  using _analog_input_range2_type =
    int8_t;
  _analog_input_range2_type analog_input_range2;
  using _analog_input_range3_type =
    int8_t;
  _analog_input_range3_type analog_input_range3;
  using _analog_input2_type =
    double;
  _analog_input2_type analog_input2;
  using _analog_input3_type =
    double;
  _analog_input3_type analog_input3;
  using _tool_voltage_48v_type =
    float;
  _tool_voltage_48v_type tool_voltage_48v;
  using _tool_output_voltage_type =
    uint8_t;
  _tool_output_voltage_type tool_output_voltage;
  using _tool_current_type =
    float;
  _tool_current_type tool_current;
  using _tool_temperature_type =
    float;
  _tool_temperature_type tool_temperature;
  using _tool_mode_type =
    uint8_t;
  _tool_mode_type tool_mode;

  // setters for named parameter idiom
  Type & set__analog_input_range2(
    const int8_t & _arg)
  {
    this->analog_input_range2 = _arg;
    return *this;
  }
  Type & set__analog_input_range3(
    const int8_t & _arg)
  {
    this->analog_input_range3 = _arg;
    return *this;
  }
  Type & set__analog_input2(
    const double & _arg)
  {
    this->analog_input2 = _arg;
    return *this;
  }
  Type & set__analog_input3(
    const double & _arg)
  {
    this->analog_input3 = _arg;
    return *this;
  }
  Type & set__tool_voltage_48v(
    const float & _arg)
  {
    this->tool_voltage_48v = _arg;
    return *this;
  }
  Type & set__tool_output_voltage(
    const uint8_t & _arg)
  {
    this->tool_output_voltage = _arg;
    return *this;
  }
  Type & set__tool_current(
    const float & _arg)
  {
    this->tool_current = _arg;
    return *this;
  }
  Type & set__tool_temperature(
    const float & _arg)
  {
    this->tool_temperature = _arg;
    return *this;
  }
  Type & set__tool_mode(
    const uint8_t & _arg)
  {
    this->tool_mode = _arg;
    return *this;
  }

  // constant declarations
  static constexpr int8_t ANALOG_INPUT_RANGE_CURRENT =
    0;
  static constexpr int8_t ANALOG_INPUT_RANGE_VOLTAGE =
    1;
  static constexpr uint8_t TOOL_BOOTLOADER_MODE =
    249u;
  static constexpr uint8_t TOOL_RUNNING_MODE =
    253u;
  static constexpr uint8_t TOOL_IDLE_MODE =
    255u;

  // pointer types
  using RawPtr =
    ur_msgs::msg::ToolDataMsg_<ContainerAllocator> *;
  using ConstRawPtr =
    const ur_msgs::msg::ToolDataMsg_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ur_msgs::msg::ToolDataMsg_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ur_msgs::msg::ToolDataMsg_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::ToolDataMsg_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::ToolDataMsg_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::ToolDataMsg_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::ToolDataMsg_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ur_msgs::msg::ToolDataMsg_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ur_msgs::msg::ToolDataMsg_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ur_msgs__msg__ToolDataMsg
    std::shared_ptr<ur_msgs::msg::ToolDataMsg_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ur_msgs__msg__ToolDataMsg
    std::shared_ptr<ur_msgs::msg::ToolDataMsg_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ToolDataMsg_ & other) const
  {
    if (this->analog_input_range2 != other.analog_input_range2) {
      return false;
    }
    if (this->analog_input_range3 != other.analog_input_range3) {
      return false;
    }
    if (this->analog_input2 != other.analog_input2) {
      return false;
    }
    if (this->analog_input3 != other.analog_input3) {
      return false;
    }
    if (this->tool_voltage_48v != other.tool_voltage_48v) {
      return false;
    }
    if (this->tool_output_voltage != other.tool_output_voltage) {
      return false;
    }
    if (this->tool_current != other.tool_current) {
      return false;
    }
    if (this->tool_temperature != other.tool_temperature) {
      return false;
    }
    if (this->tool_mode != other.tool_mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const ToolDataMsg_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ToolDataMsg_

// alias to use template instance with default allocator
using ToolDataMsg =
  ur_msgs::msg::ToolDataMsg_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ToolDataMsg_<ContainerAllocator>::ANALOG_INPUT_RANGE_CURRENT;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ToolDataMsg_<ContainerAllocator>::ANALOG_INPUT_RANGE_VOLTAGE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t ToolDataMsg_<ContainerAllocator>::TOOL_BOOTLOADER_MODE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t ToolDataMsg_<ContainerAllocator>::TOOL_RUNNING_MODE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t ToolDataMsg_<ContainerAllocator>::TOOL_IDLE_MODE;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace ur_msgs

#endif  // UR_MSGS__MSG__DETAIL__TOOL_DATA_MSG__STRUCT_HPP_
