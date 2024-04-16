// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ur_msgs:msg/RobotStateRTMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__ROBOT_STATE_RT_MSG__STRUCT_HPP_
#define UR_MSGS__MSG__DETAIL__ROBOT_STATE_RT_MSG__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ur_msgs__msg__RobotStateRTMsg __attribute__((deprecated))
#else
# define DEPRECATED__ur_msgs__msg__RobotStateRTMsg __declspec(deprecated)
#endif

namespace ur_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RobotStateRTMsg_
{
  using Type = RobotStateRTMsg_<ContainerAllocator>;

  explicit RobotStateRTMsg_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->time = 0.0;
      this->digital_input_bits = 0.0;
      this->controller_timer = 0.0;
      this->test_value = 0.0;
      this->robot_mode = 0.0;
    }
  }

  explicit RobotStateRTMsg_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->time = 0.0;
      this->digital_input_bits = 0.0;
      this->controller_timer = 0.0;
      this->test_value = 0.0;
      this->robot_mode = 0.0;
    }
  }

  // field types and members
  using _time_type =
    double;
  _time_type time;
  using _q_target_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _q_target_type q_target;
  using _qd_target_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _qd_target_type qd_target;
  using _qdd_target_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _qdd_target_type qdd_target;
  using _i_target_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _i_target_type i_target;
  using _m_target_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _m_target_type m_target;
  using _q_actual_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _q_actual_type q_actual;
  using _qd_actual_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _qd_actual_type qd_actual;
  using _i_actual_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _i_actual_type i_actual;
  using _tool_acc_values_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _tool_acc_values_type tool_acc_values;
  using _tcp_force_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _tcp_force_type tcp_force;
  using _tool_vector_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _tool_vector_type tool_vector;
  using _tcp_speed_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _tcp_speed_type tcp_speed;
  using _digital_input_bits_type =
    double;
  _digital_input_bits_type digital_input_bits;
  using _motor_temperatures_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _motor_temperatures_type motor_temperatures;
  using _controller_timer_type =
    double;
  _controller_timer_type controller_timer;
  using _test_value_type =
    double;
  _test_value_type test_value;
  using _robot_mode_type =
    double;
  _robot_mode_type robot_mode;
  using _joint_modes_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _joint_modes_type joint_modes;

  // setters for named parameter idiom
  Type & set__time(
    const double & _arg)
  {
    this->time = _arg;
    return *this;
  }
  Type & set__q_target(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->q_target = _arg;
    return *this;
  }
  Type & set__qd_target(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->qd_target = _arg;
    return *this;
  }
  Type & set__qdd_target(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->qdd_target = _arg;
    return *this;
  }
  Type & set__i_target(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->i_target = _arg;
    return *this;
  }
  Type & set__m_target(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->m_target = _arg;
    return *this;
  }
  Type & set__q_actual(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->q_actual = _arg;
    return *this;
  }
  Type & set__qd_actual(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->qd_actual = _arg;
    return *this;
  }
  Type & set__i_actual(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->i_actual = _arg;
    return *this;
  }
  Type & set__tool_acc_values(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->tool_acc_values = _arg;
    return *this;
  }
  Type & set__tcp_force(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->tcp_force = _arg;
    return *this;
  }
  Type & set__tool_vector(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->tool_vector = _arg;
    return *this;
  }
  Type & set__tcp_speed(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->tcp_speed = _arg;
    return *this;
  }
  Type & set__digital_input_bits(
    const double & _arg)
  {
    this->digital_input_bits = _arg;
    return *this;
  }
  Type & set__motor_temperatures(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->motor_temperatures = _arg;
    return *this;
  }
  Type & set__controller_timer(
    const double & _arg)
  {
    this->controller_timer = _arg;
    return *this;
  }
  Type & set__test_value(
    const double & _arg)
  {
    this->test_value = _arg;
    return *this;
  }
  Type & set__robot_mode(
    const double & _arg)
  {
    this->robot_mode = _arg;
    return *this;
  }
  Type & set__joint_modes(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->joint_modes = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ur_msgs::msg::RobotStateRTMsg_<ContainerAllocator> *;
  using ConstRawPtr =
    const ur_msgs::msg::RobotStateRTMsg_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ur_msgs::msg::RobotStateRTMsg_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ur_msgs::msg::RobotStateRTMsg_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::RobotStateRTMsg_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::RobotStateRTMsg_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::RobotStateRTMsg_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::RobotStateRTMsg_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ur_msgs::msg::RobotStateRTMsg_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ur_msgs::msg::RobotStateRTMsg_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ur_msgs__msg__RobotStateRTMsg
    std::shared_ptr<ur_msgs::msg::RobotStateRTMsg_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ur_msgs__msg__RobotStateRTMsg
    std::shared_ptr<ur_msgs::msg::RobotStateRTMsg_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RobotStateRTMsg_ & other) const
  {
    if (this->time != other.time) {
      return false;
    }
    if (this->q_target != other.q_target) {
      return false;
    }
    if (this->qd_target != other.qd_target) {
      return false;
    }
    if (this->qdd_target != other.qdd_target) {
      return false;
    }
    if (this->i_target != other.i_target) {
      return false;
    }
    if (this->m_target != other.m_target) {
      return false;
    }
    if (this->q_actual != other.q_actual) {
      return false;
    }
    if (this->qd_actual != other.qd_actual) {
      return false;
    }
    if (this->i_actual != other.i_actual) {
      return false;
    }
    if (this->tool_acc_values != other.tool_acc_values) {
      return false;
    }
    if (this->tcp_force != other.tcp_force) {
      return false;
    }
    if (this->tool_vector != other.tool_vector) {
      return false;
    }
    if (this->tcp_speed != other.tcp_speed) {
      return false;
    }
    if (this->digital_input_bits != other.digital_input_bits) {
      return false;
    }
    if (this->motor_temperatures != other.motor_temperatures) {
      return false;
    }
    if (this->controller_timer != other.controller_timer) {
      return false;
    }
    if (this->test_value != other.test_value) {
      return false;
    }
    if (this->robot_mode != other.robot_mode) {
      return false;
    }
    if (this->joint_modes != other.joint_modes) {
      return false;
    }
    return true;
  }
  bool operator!=(const RobotStateRTMsg_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RobotStateRTMsg_

// alias to use template instance with default allocator
using RobotStateRTMsg =
  ur_msgs::msg::RobotStateRTMsg_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ur_msgs

#endif  // UR_MSGS__MSG__DETAIL__ROBOT_STATE_RT_MSG__STRUCT_HPP_
