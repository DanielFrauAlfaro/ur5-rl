// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ur_msgs:msg/RobotModeDataMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__ROBOT_MODE_DATA_MSG__STRUCT_HPP_
#define UR_MSGS__MSG__DETAIL__ROBOT_MODE_DATA_MSG__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ur_msgs__msg__RobotModeDataMsg __attribute__((deprecated))
#else
# define DEPRECATED__ur_msgs__msg__RobotModeDataMsg __declspec(deprecated)
#endif

namespace ur_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RobotModeDataMsg_
{
  using Type = RobotModeDataMsg_<ContainerAllocator>;

  explicit RobotModeDataMsg_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->is_robot_connected = false;
      this->is_real_robot_enabled = false;
      this->is_power_on_robot = false;
      this->is_emergency_stopped = false;
      this->is_protective_stopped = false;
      this->is_program_running = false;
      this->is_program_paused = false;
    }
  }

  explicit RobotModeDataMsg_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->timestamp = 0ull;
      this->is_robot_connected = false;
      this->is_real_robot_enabled = false;
      this->is_power_on_robot = false;
      this->is_emergency_stopped = false;
      this->is_protective_stopped = false;
      this->is_program_running = false;
      this->is_program_paused = false;
    }
  }

  // field types and members
  using _timestamp_type =
    uint64_t;
  _timestamp_type timestamp;
  using _is_robot_connected_type =
    bool;
  _is_robot_connected_type is_robot_connected;
  using _is_real_robot_enabled_type =
    bool;
  _is_real_robot_enabled_type is_real_robot_enabled;
  using _is_power_on_robot_type =
    bool;
  _is_power_on_robot_type is_power_on_robot;
  using _is_emergency_stopped_type =
    bool;
  _is_emergency_stopped_type is_emergency_stopped;
  using _is_protective_stopped_type =
    bool;
  _is_protective_stopped_type is_protective_stopped;
  using _is_program_running_type =
    bool;
  _is_program_running_type is_program_running;
  using _is_program_paused_type =
    bool;
  _is_program_paused_type is_program_paused;

  // setters for named parameter idiom
  Type & set__timestamp(
    const uint64_t & _arg)
  {
    this->timestamp = _arg;
    return *this;
  }
  Type & set__is_robot_connected(
    const bool & _arg)
  {
    this->is_robot_connected = _arg;
    return *this;
  }
  Type & set__is_real_robot_enabled(
    const bool & _arg)
  {
    this->is_real_robot_enabled = _arg;
    return *this;
  }
  Type & set__is_power_on_robot(
    const bool & _arg)
  {
    this->is_power_on_robot = _arg;
    return *this;
  }
  Type & set__is_emergency_stopped(
    const bool & _arg)
  {
    this->is_emergency_stopped = _arg;
    return *this;
  }
  Type & set__is_protective_stopped(
    const bool & _arg)
  {
    this->is_protective_stopped = _arg;
    return *this;
  }
  Type & set__is_program_running(
    const bool & _arg)
  {
    this->is_program_running = _arg;
    return *this;
  }
  Type & set__is_program_paused(
    const bool & _arg)
  {
    this->is_program_paused = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ur_msgs::msg::RobotModeDataMsg_<ContainerAllocator> *;
  using ConstRawPtr =
    const ur_msgs::msg::RobotModeDataMsg_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ur_msgs::msg::RobotModeDataMsg_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ur_msgs::msg::RobotModeDataMsg_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::RobotModeDataMsg_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::RobotModeDataMsg_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::RobotModeDataMsg_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::RobotModeDataMsg_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ur_msgs::msg::RobotModeDataMsg_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ur_msgs::msg::RobotModeDataMsg_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ur_msgs__msg__RobotModeDataMsg
    std::shared_ptr<ur_msgs::msg::RobotModeDataMsg_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ur_msgs__msg__RobotModeDataMsg
    std::shared_ptr<ur_msgs::msg::RobotModeDataMsg_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RobotModeDataMsg_ & other) const
  {
    if (this->timestamp != other.timestamp) {
      return false;
    }
    if (this->is_robot_connected != other.is_robot_connected) {
      return false;
    }
    if (this->is_real_robot_enabled != other.is_real_robot_enabled) {
      return false;
    }
    if (this->is_power_on_robot != other.is_power_on_robot) {
      return false;
    }
    if (this->is_emergency_stopped != other.is_emergency_stopped) {
      return false;
    }
    if (this->is_protective_stopped != other.is_protective_stopped) {
      return false;
    }
    if (this->is_program_running != other.is_program_running) {
      return false;
    }
    if (this->is_program_paused != other.is_program_paused) {
      return false;
    }
    return true;
  }
  bool operator!=(const RobotModeDataMsg_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RobotModeDataMsg_

// alias to use template instance with default allocator
using RobotModeDataMsg =
  ur_msgs::msg::RobotModeDataMsg_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ur_msgs

#endif  // UR_MSGS__MSG__DETAIL__ROBOT_MODE_DATA_MSG__STRUCT_HPP_
