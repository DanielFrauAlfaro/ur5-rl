// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ur_msgs:srv/SetIO.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__SRV__DETAIL__SET_IO__BUILDER_HPP_
#define UR_MSGS__SRV__DETAIL__SET_IO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ur_msgs/srv/detail/set_io__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ur_msgs
{

namespace srv
{

namespace builder
{

class Init_SetIO_Request_state
{
public:
  explicit Init_SetIO_Request_state(::ur_msgs::srv::SetIO_Request & msg)
  : msg_(msg)
  {}
  ::ur_msgs::srv::SetIO_Request state(::ur_msgs::srv::SetIO_Request::_state_type arg)
  {
    msg_.state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ur_msgs::srv::SetIO_Request msg_;
};

class Init_SetIO_Request_pin
{
public:
  explicit Init_SetIO_Request_pin(::ur_msgs::srv::SetIO_Request & msg)
  : msg_(msg)
  {}
  Init_SetIO_Request_state pin(::ur_msgs::srv::SetIO_Request::_pin_type arg)
  {
    msg_.pin = std::move(arg);
    return Init_SetIO_Request_state(msg_);
  }

private:
  ::ur_msgs::srv::SetIO_Request msg_;
};

class Init_SetIO_Request_fun
{
public:
  Init_SetIO_Request_fun()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetIO_Request_pin fun(::ur_msgs::srv::SetIO_Request::_fun_type arg)
  {
    msg_.fun = std::move(arg);
    return Init_SetIO_Request_pin(msg_);
  }

private:
  ::ur_msgs::srv::SetIO_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::ur_msgs::srv::SetIO_Request>()
{
  return ur_msgs::srv::builder::Init_SetIO_Request_fun();
}

}  // namespace ur_msgs


namespace ur_msgs
{

namespace srv
{

namespace builder
{

class Init_SetIO_Response_success
{
public:
  Init_SetIO_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::ur_msgs::srv::SetIO_Response success(::ur_msgs::srv::SetIO_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ur_msgs::srv::SetIO_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::ur_msgs::srv::SetIO_Response>()
{
  return ur_msgs::srv::builder::Init_SetIO_Response_success();
}

}  // namespace ur_msgs

#endif  // UR_MSGS__SRV__DETAIL__SET_IO__BUILDER_HPP_
