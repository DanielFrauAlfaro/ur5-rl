// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ur_msgs:msg/MasterboardDataMsg.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__MSG__DETAIL__MASTERBOARD_DATA_MSG__STRUCT_HPP_
#define UR_MSGS__MSG__DETAIL__MASTERBOARD_DATA_MSG__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ur_msgs__msg__MasterboardDataMsg __attribute__((deprecated))
#else
# define DEPRECATED__ur_msgs__msg__MasterboardDataMsg __declspec(deprecated)
#endif

namespace ur_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MasterboardDataMsg_
{
  using Type = MasterboardDataMsg_<ContainerAllocator>;

  explicit MasterboardDataMsg_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->digital_input_bits = 0ul;
      this->digital_output_bits = 0ul;
      this->analog_input_range0 = 0;
      this->analog_input_range1 = 0;
      this->analog_input0 = 0.0;
      this->analog_input1 = 0.0;
      this->analog_output_domain0 = 0;
      this->analog_output_domain1 = 0;
      this->analog_output0 = 0.0;
      this->analog_output1 = 0.0;
      this->masterboard_temperature = 0.0f;
      this->robot_voltage_48v = 0.0f;
      this->robot_current = 0.0f;
      this->master_io_current = 0.0f;
      this->master_safety_state = 0;
      this->master_onoff_state = 0;
    }
  }

  explicit MasterboardDataMsg_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->digital_input_bits = 0ul;
      this->digital_output_bits = 0ul;
      this->analog_input_range0 = 0;
      this->analog_input_range1 = 0;
      this->analog_input0 = 0.0;
      this->analog_input1 = 0.0;
      this->analog_output_domain0 = 0;
      this->analog_output_domain1 = 0;
      this->analog_output0 = 0.0;
      this->analog_output1 = 0.0;
      this->masterboard_temperature = 0.0f;
      this->robot_voltage_48v = 0.0f;
      this->robot_current = 0.0f;
      this->master_io_current = 0.0f;
      this->master_safety_state = 0;
      this->master_onoff_state = 0;
    }
  }

  // field types and members
  using _digital_input_bits_type =
    uint32_t;
  _digital_input_bits_type digital_input_bits;
  using _digital_output_bits_type =
    uint32_t;
  _digital_output_bits_type digital_output_bits;
  using _analog_input_range0_type =
    int8_t;
  _analog_input_range0_type analog_input_range0;
  using _analog_input_range1_type =
    int8_t;
  _analog_input_range1_type analog_input_range1;
  using _analog_input0_type =
    double;
  _analog_input0_type analog_input0;
  using _analog_input1_type =
    double;
  _analog_input1_type analog_input1;
  using _analog_output_domain0_type =
    int8_t;
  _analog_output_domain0_type analog_output_domain0;
  using _analog_output_domain1_type =
    int8_t;
  _analog_output_domain1_type analog_output_domain1;
  using _analog_output0_type =
    double;
  _analog_output0_type analog_output0;
  using _analog_output1_type =
    double;
  _analog_output1_type analog_output1;
  using _masterboard_temperature_type =
    float;
  _masterboard_temperature_type masterboard_temperature;
  using _robot_voltage_48v_type =
    float;
  _robot_voltage_48v_type robot_voltage_48v;
  using _robot_current_type =
    float;
  _robot_current_type robot_current;
  using _master_io_current_type =
    float;
  _master_io_current_type master_io_current;
  using _master_safety_state_type =
    uint8_t;
  _master_safety_state_type master_safety_state;
  using _master_onoff_state_type =
    uint8_t;
  _master_onoff_state_type master_onoff_state;

  // setters for named parameter idiom
  Type & set__digital_input_bits(
    const uint32_t & _arg)
  {
    this->digital_input_bits = _arg;
    return *this;
  }
  Type & set__digital_output_bits(
    const uint32_t & _arg)
  {
    this->digital_output_bits = _arg;
    return *this;
  }
  Type & set__analog_input_range0(
    const int8_t & _arg)
  {
    this->analog_input_range0 = _arg;
    return *this;
  }
  Type & set__analog_input_range1(
    const int8_t & _arg)
  {
    this->analog_input_range1 = _arg;
    return *this;
  }
  Type & set__analog_input0(
    const double & _arg)
  {
    this->analog_input0 = _arg;
    return *this;
  }
  Type & set__analog_input1(
    const double & _arg)
  {
    this->analog_input1 = _arg;
    return *this;
  }
  Type & set__analog_output_domain0(
    const int8_t & _arg)
  {
    this->analog_output_domain0 = _arg;
    return *this;
  }
  Type & set__analog_output_domain1(
    const int8_t & _arg)
  {
    this->analog_output_domain1 = _arg;
    return *this;
  }
  Type & set__analog_output0(
    const double & _arg)
  {
    this->analog_output0 = _arg;
    return *this;
  }
  Type & set__analog_output1(
    const double & _arg)
  {
    this->analog_output1 = _arg;
    return *this;
  }
  Type & set__masterboard_temperature(
    const float & _arg)
  {
    this->masterboard_temperature = _arg;
    return *this;
  }
  Type & set__robot_voltage_48v(
    const float & _arg)
  {
    this->robot_voltage_48v = _arg;
    return *this;
  }
  Type & set__robot_current(
    const float & _arg)
  {
    this->robot_current = _arg;
    return *this;
  }
  Type & set__master_io_current(
    const float & _arg)
  {
    this->master_io_current = _arg;
    return *this;
  }
  Type & set__master_safety_state(
    const uint8_t & _arg)
  {
    this->master_safety_state = _arg;
    return *this;
  }
  Type & set__master_onoff_state(
    const uint8_t & _arg)
  {
    this->master_onoff_state = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ur_msgs::msg::MasterboardDataMsg_<ContainerAllocator> *;
  using ConstRawPtr =
    const ur_msgs::msg::MasterboardDataMsg_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ur_msgs::msg::MasterboardDataMsg_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ur_msgs::msg::MasterboardDataMsg_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::MasterboardDataMsg_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::MasterboardDataMsg_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ur_msgs::msg::MasterboardDataMsg_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::msg::MasterboardDataMsg_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ur_msgs::msg::MasterboardDataMsg_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ur_msgs::msg::MasterboardDataMsg_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ur_msgs__msg__MasterboardDataMsg
    std::shared_ptr<ur_msgs::msg::MasterboardDataMsg_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ur_msgs__msg__MasterboardDataMsg
    std::shared_ptr<ur_msgs::msg::MasterboardDataMsg_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MasterboardDataMsg_ & other) const
  {
    if (this->digital_input_bits != other.digital_input_bits) {
      return false;
    }
    if (this->digital_output_bits != other.digital_output_bits) {
      return false;
    }
    if (this->analog_input_range0 != other.analog_input_range0) {
      return false;
    }
    if (this->analog_input_range1 != other.analog_input_range1) {
      return false;
    }
    if (this->analog_input0 != other.analog_input0) {
      return false;
    }
    if (this->analog_input1 != other.analog_input1) {
      return false;
    }
    if (this->analog_output_domain0 != other.analog_output_domain0) {
      return false;
    }
    if (this->analog_output_domain1 != other.analog_output_domain1) {
      return false;
    }
    if (this->analog_output0 != other.analog_output0) {
      return false;
    }
    if (this->analog_output1 != other.analog_output1) {
      return false;
    }
    if (this->masterboard_temperature != other.masterboard_temperature) {
      return false;
    }
    if (this->robot_voltage_48v != other.robot_voltage_48v) {
      return false;
    }
    if (this->robot_current != other.robot_current) {
      return false;
    }
    if (this->master_io_current != other.master_io_current) {
      return false;
    }
    if (this->master_safety_state != other.master_safety_state) {
      return false;
    }
    if (this->master_onoff_state != other.master_onoff_state) {
      return false;
    }
    return true;
  }
  bool operator!=(const MasterboardDataMsg_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MasterboardDataMsg_

// alias to use template instance with default allocator
using MasterboardDataMsg =
  ur_msgs::msg::MasterboardDataMsg_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ur_msgs

#endif  // UR_MSGS__MSG__DETAIL__MASTERBOARD_DATA_MSG__STRUCT_HPP_
