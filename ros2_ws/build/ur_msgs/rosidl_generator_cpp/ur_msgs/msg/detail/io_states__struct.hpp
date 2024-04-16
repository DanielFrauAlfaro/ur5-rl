// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ur_msgs:msg/IOStates.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__IO_STATES__STRUCT_HPP_
#define UR_MSGS__MSG__DETAIL__IO_STATES__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'digital_in_states'
// Member 'digital_out_states'
// Member 'flag_states'
#include "ur_msgs/msg/detail/digital__struct.hpp"
// Member 'analog_in_states'
// Member 'analog_out_states'
#include "ur_msgs/msg/detail/analog__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__ur_msgs__msg__IOStates __attribute__((deprecated))
#else
# define DEPRECATED__ur_msgs__msg__IOStates __declspec(deprecated)
#endif

namespace ur_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct IOStates_
{
  using Type = IOStates_<ContainerAllocator>;

  explicit IOStates_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit IOStates_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _digital_in_states_type =
    std::vector<ur_msgs::msg::Digital_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ur_msgs::msg::Digital_<ContainerAllocator>>>;
  _digital_in_states_type digital_in_states;
  using _digital_out_states_type =
    std::vector<ur_msgs::msg::Digital_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ur_msgs::msg::Digital_<ContainerAllocator>>>;
  _digital_out_states_type digital_out_states;
  using _flag_states_type =
    std::vector<ur_msgs::msg::Digital_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ur_msgs::msg::Digital_<ContainerAllocator>>>;
  _flag_states_type flag_states;
  using _analog_in_states_type =
    std::vector<ur_msgs::msg::Analog_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ur_msgs::msg::Analog_<ContainerAllocator>>>;
  _analog_in_states_type analog_in_states;
  using _analog_out_states_type =
    std::vector<ur_msgs::msg::Analog_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ur_msgs::msg::Analog_<ContainerAllocator>>>;
  _analog_out_states_type analog_out_states;

  // setters for named parameter idiom
  Type & set__digital_in_states(
    const std::vector<ur_msgs::msg::Digital_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ur_msgs::msg::Digital_<ContainerAllocator>>> & _arg)
  {
    this->digital_in_states = _arg;
    return *this;
  }
  Type & set__digital_out_states(
    const std::vector<ur_msgs::msg::Digital_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ur_msgs::msg::Digital_<ContainerAllocator>>> & _arg)
  {
    this->digital_out_states = _arg;
    return *this;
  }
  Type & set__flag_states(
    const std::vector<ur_msgs::msg::Digital_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ur_msgs::msg::Digital_<ContainerAllocator>>> & _arg)
  {
    this->flag_states = _arg;
    return *this;
  }
  Type & set__analog_in_states(
    const std::vector<ur_msgs::msg::Analog_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ur_msgs::msg::Analog_<ContainerAllocator>>> & _arg)
  {
    this->analog_in_states = _arg;
    return *this;
  }
  Type & set__analog_out_states(
    const std::vector<ur_msgs::msg::Analog_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ur_msgs::msg::Analog_<ContainerAllocator>>> & _arg)
  {
    this->analog_out_states = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ur_msgs::msg::IOStates_<ContainerAllocator> *;
  using ConstRawPtr =
    const ur_msgs::msg::IOStates_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ur_msgs::msg::IOStates_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ur_msgs::msg::IOStates_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::IOStates_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::IOStates_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::IOStates_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::IOStates_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ur_msgs::msg::IOStates_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ur_msgs::msg::IOStates_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ur_msgs__msg__IOStates
    std::shared_ptr<ur_msgs::msg::IOStates_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ur_msgs__msg__IOStates
    std::shared_ptr<ur_msgs::msg::IOStates_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const IOStates_ & other) const
  {
    if (this->digital_in_states != other.digital_in_states) {
      return false;
    }
    if (this->digital_out_states != other.digital_out_states) {
      return false;
    }
    if (this->flag_states != other.flag_states) {
      return false;
    }
    if (this->analog_in_states != other.analog_in_states) {
      return false;
    }
    if (this->analog_out_states != other.analog_out_states) {
      return false;
    }
    return true;
  }
  bool operator!=(const IOStates_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct IOStates_

// alias to use template instance with default allocator
using IOStates =
  ur_msgs::msg::IOStates_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ur_msgs

#endif  // UR_MSGS__MSG__DETAIL__IO_STATES__STRUCT_HPP_
