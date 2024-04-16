// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ur_msgs:srv/SetPayload.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__SRV__DETAIL__SET_PAYLOAD__STRUCT_HPP_
#define UR_MSGS__SRV__DETAIL__SET_PAYLOAD__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'center_of_gravity'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__ur_msgs__srv__SetPayload_Request __attribute__((deprecated))
#else
# define DEPRECATED__ur_msgs__srv__SetPayload_Request __declspec(deprecated)
#endif

namespace ur_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetPayload_Request_
{
  using Type = SetPayload_Request_<ContainerAllocator>;

  explicit SetPayload_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : center_of_gravity(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mass = 0.0f;
    }
  }

  explicit SetPayload_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : center_of_gravity(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mass = 0.0f;
    }
  }

  // field types and members
  using _mass_type =
    float;
  _mass_type mass;
  using _center_of_gravity_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _center_of_gravity_type center_of_gravity;

  // setters for named parameter idiom
  Type & set__mass(
    const float & _arg)
  {
    this->mass = _arg;
    return *this;
  }
  Type & set__center_of_gravity(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->center_of_gravity = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ur_msgs::srv::SetPayload_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const ur_msgs::srv::SetPayload_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ur_msgs::srv::SetPayload_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ur_msgs::srv::SetPayload_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ur_msgs::srv::SetPayload_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::srv::SetPayload_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ur_msgs::srv::SetPayload_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::srv::SetPayload_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ur_msgs::srv::SetPayload_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ur_msgs::srv::SetPayload_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ur_msgs__srv__SetPayload_Request
    std::shared_ptr<ur_msgs::srv::SetPayload_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ur_msgs__srv__SetPayload_Request
    std::shared_ptr<ur_msgs::srv::SetPayload_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetPayload_Request_ & other) const
  {
    if (this->mass != other.mass) {
      return false;
    }
    if (this->center_of_gravity != other.center_of_gravity) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetPayload_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetPayload_Request_

// alias to use template instance with default allocator
using SetPayload_Request =
  ur_msgs::srv::SetPayload_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace ur_msgs


#ifndef _WIN32
# define DEPRECATED__ur_msgs__srv__SetPayload_Response __attribute__((deprecated))
#else
# define DEPRECATED__ur_msgs__srv__SetPayload_Response __declspec(deprecated)
#endif

namespace ur_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetPayload_Response_
{
  using Type = SetPayload_Response_<ContainerAllocator>;

  explicit SetPayload_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit SetPayload_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    ur_msgs::srv::SetPayload_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const ur_msgs::srv::SetPayload_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ur_msgs::srv::SetPayload_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ur_msgs::srv::SetPayload_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ur_msgs::srv::SetPayload_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::srv::SetPayload_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ur_msgs::srv::SetPayload_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ur_msgs::srv::SetPayload_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ur_msgs::srv::SetPayload_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ur_msgs::srv::SetPayload_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ur_msgs__srv__SetPayload_Response
    std::shared_ptr<ur_msgs::srv::SetPayload_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ur_msgs__srv__SetPayload_Response
    std::shared_ptr<ur_msgs::srv::SetPayload_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetPayload_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetPayload_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetPayload_Response_

// alias to use template instance with default allocator
using SetPayload_Response =
  ur_msgs::srv::SetPayload_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace ur_msgs

namespace ur_msgs
{

namespace srv
{

struct SetPayload
{
  using Request = ur_msgs::srv::SetPayload_Request;
  using Response = ur_msgs::srv::SetPayload_Response;
};

}  // namespace srv

}  // namespace ur_msgs

#endif  // UR_MSGS__SRV__DETAIL__SET_PAYLOAD__STRUCT_HPP_
