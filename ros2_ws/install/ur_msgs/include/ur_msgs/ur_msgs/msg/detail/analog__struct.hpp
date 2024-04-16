// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ur_msgs:msg/Analog.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__ANALOG__STRUCT_HPP_
#define UR_MSGS__MSG__DETAIL__ANALOG__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ur_msgs__msg__Analog __attribute__((deprecated))
#else
# define DEPRECATED__ur_msgs__msg__Analog __declspec(deprecated)
#endif

namespace ur_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Analog_
{
  using Type = Analog_<ContainerAllocator>;

  explicit Analog_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->pin = 0;
      this->domain = 0;
      this->state = 0.0f;
    }
  }

  explicit Analog_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->pin = 0;
      this->domain = 0;
      this->state = 0.0f;
    }
  }

  // field types and members
  using _pin_type =
    uint8_t;
  _pin_type pin;
  using _domain_type =
    uint8_t;
  _domain_type domain;
  using _state_type =
    float;
  _state_type state;

  // setters for named parameter idiom
  Type & set__pin(
    const uint8_t & _arg)
  {
    this->pin = _arg;
    return *this;
  }
  Type & set__domain(
    const uint8_t & _arg)
  {
    this->domain = _arg;
    return *this;
  }
  Type & set__state(
    const float & _arg)
  {
    this->state = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t CURRENT =
    0u;
  static constexpr uint8_t VOLTAGE =
    1u;

  // pointer types
  using RawPtr =
    ur_msgs::msg::Analog_<ContainerAllocator> *;
  using ConstRawPtr =
    const ur_msgs::msg::Analog_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ur_msgs::msg::Analog_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ur_msgs::msg::Analog_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::Analog_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::Analog_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::Analog_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::Analog_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ur_msgs::msg::Analog_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ur_msgs::msg::Analog_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ur_msgs__msg__Analog
    std::shared_ptr<ur_msgs::msg::Analog_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ur_msgs__msg__Analog
    std::shared_ptr<ur_msgs::msg::Analog_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Analog_ & other) const
  {
    if (this->pin != other.pin) {
      return false;
    }
    if (this->domain != other.domain) {
      return false;
    }
    if (this->state != other.state) {
      return false;
    }
    return true;
  }
  bool operator!=(const Analog_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Analog_

// alias to use template instance with default allocator
using Analog =
  ur_msgs::msg::Analog_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Analog_<ContainerAllocator>::CURRENT;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t Analog_<ContainerAllocator>::VOLTAGE;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace ur_msgs

#endif  // UR_MSGS__MSG__DETAIL__ANALOG__STRUCT_HPP_
