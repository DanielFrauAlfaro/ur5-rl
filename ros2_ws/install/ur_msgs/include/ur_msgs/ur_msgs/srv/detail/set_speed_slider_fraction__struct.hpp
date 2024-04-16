// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ur_msgs:srv/SetSpeedSliderFraction.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__SRV__DETAIL__SET_SPEED_SLIDER_FRACTION__STRUCT_HPP_
#define UR_MSGS__SRV__DETAIL__SET_SPEED_SLIDER_FRACTION__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ur_msgs__srv__SetSpeedSliderFraction_Request __attribute__((deprecated))
#else
# define DEPRECATED__ur_msgs__srv__SetSpeedSliderFraction_Request __declspec(deprecated)
#endif

namespace ur_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetSpeedSliderFraction_Request_
{
  using Type = SetSpeedSliderFraction_Request_<ContainerAllocator>;

  explicit SetSpeedSliderFraction_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->speed_slider_fraction = 0.0;
    }
  }

  explicit SetSpeedSliderFraction_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->speed_slider_fraction = 0.0;
    }
  }

  // field types and members
  using _speed_slider_fraction_type =
    double;
  _speed_slider_fraction_type speed_slider_fraction;

  // setters for named parameter idiom
  Type & set__speed_slider_fraction(
    const double & _arg)
  {
    this->speed_slider_fraction = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ur_msgs::srv::SetSpeedSliderFraction_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const ur_msgs::srv::SetSpeedSliderFraction_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ur_msgs::srv::SetSpeedSliderFraction_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ur_msgs::srv::SetSpeedSliderFraction_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ur_msgs::srv::SetSpeedSliderFraction_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::srv::SetSpeedSliderFraction_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ur_msgs::srv::SetSpeedSliderFraction_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::srv::SetSpeedSliderFraction_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ur_msgs::srv::SetSpeedSliderFraction_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ur_msgs::srv::SetSpeedSliderFraction_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ur_msgs__srv__SetSpeedSliderFraction_Request
    std::shared_ptr<ur_msgs::srv::SetSpeedSliderFraction_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ur_msgs__srv__SetSpeedSliderFraction_Request
    std::shared_ptr<ur_msgs::srv::SetSpeedSliderFraction_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetSpeedSliderFraction_Request_ & other) const
  {
    if (this->speed_slider_fraction != other.speed_slider_fraction) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetSpeedSliderFraction_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetSpeedSliderFraction_Request_

// alias to use template instance with default allocator
using SetSpeedSliderFraction_Request =
  ur_msgs::srv::SetSpeedSliderFraction_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace ur_msgs


#ifndef _WIN32
# define DEPRECATED__ur_msgs__srv__SetSpeedSliderFraction_Response __attribute__((deprecated))
#else
# define DEPRECATED__ur_msgs__srv__SetSpeedSliderFraction_Response __declspec(deprecated)
#endif

namespace ur_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetSpeedSliderFraction_Response_
{
  using Type = SetSpeedSliderFraction_Response_<ContainerAllocator>;

  explicit SetSpeedSliderFraction_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit SetSpeedSliderFraction_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ur_msgs::srv::SetSpeedSliderFraction_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const ur_msgs::srv::SetSpeedSliderFraction_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ur_msgs::srv::SetSpeedSliderFraction_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ur_msgs::srv::SetSpeedSliderFraction_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ur_msgs::srv::SetSpeedSliderFraction_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::srv::SetSpeedSliderFraction_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ur_msgs::srv::SetSpeedSliderFraction_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::srv::SetSpeedSliderFraction_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ur_msgs::srv::SetSpeedSliderFraction_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ur_msgs::srv::SetSpeedSliderFraction_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ur_msgs__srv__SetSpeedSliderFraction_Response
    std::shared_ptr<ur_msgs::srv::SetSpeedSliderFraction_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ur_msgs__srv__SetSpeedSliderFraction_Response
    std::shared_ptr<ur_msgs::srv::SetSpeedSliderFraction_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetSpeedSliderFraction_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetSpeedSliderFraction_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetSpeedSliderFraction_Response_

// alias to use template instance with default allocator
using SetSpeedSliderFraction_Response =
  ur_msgs::srv::SetSpeedSliderFraction_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace ur_msgs

namespace ur_msgs
{

namespace srv
{

struct SetSpeedSliderFraction
{
  using Request = ur_msgs::srv::SetSpeedSliderFraction_Request;
  using Response = ur_msgs::srv::SetSpeedSliderFraction_Response;
};

}  // namespace srv

}  // namespace ur_msgs

#endif  // UR_MSGS__SRV__DETAIL__SET_SPEED_SLIDER_FRACTION__STRUCT_HPP_
