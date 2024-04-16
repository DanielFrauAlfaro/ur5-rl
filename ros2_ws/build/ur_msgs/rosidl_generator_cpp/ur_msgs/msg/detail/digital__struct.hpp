// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ur_msgs:msg/Digital.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__DIGITAL__STRUCT_HPP_
#define UR_MSGS__MSG__DETAIL__DIGITAL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ur_msgs__msg__Digital __attribute__((deprecated))
#else
# define DEPRECATED__ur_msgs__msg__Digital __declspec(deprecated)
#endif

namespace ur_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Digital_
{
  using Type = Digital_<ContainerAllocator>;

  explicit Digital_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->pin = 0;
      this->state = false;
    }
  }

  explicit Digital_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->pin = 0;
      this->state = false;
    }
  }

  // field types and members
  using _pin_type =
    uint8_t;
  _pin_type pin;
  using _state_type =
    bool;
  _state_type state;

  // setters for named parameter idiom
  Type & set__pin(
    const uint8_t & _arg)
  {
    this->pin = _arg;
    return *this;
  }
  Type & set__state(
    const bool & _arg)
  {
    this->state = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ur_msgs::msg::Digital_<ContainerAllocator> *;
  using ConstRawPtr =
    const ur_msgs::msg::Digital_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ur_msgs::msg::Digital_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ur_msgs::msg::Digital_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::Digital_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::Digital_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::Digital_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::Digital_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ur_msgs::msg::Digital_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ur_msgs::msg::Digital_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ur_msgs__msg__Digital
    std::shared_ptr<ur_msgs::msg::Digital_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ur_msgs__msg__Digital
    std::shared_ptr<ur_msgs::msg::Digital_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Digital_ & other) const
  {
    if (this->pin != other.pin) {
      return false;
    }
    if (this->state != other.state) {
      return false;
    }
    return true;
  }
  bool operator!=(const Digital_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Digital_

// alias to use template instance with default allocator
using Digital =
  ur_msgs::msg::Digital_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ur_msgs

#endif  // UR_MSGS__MSG__DETAIL__DIGITAL__STRUCT_HPP_
