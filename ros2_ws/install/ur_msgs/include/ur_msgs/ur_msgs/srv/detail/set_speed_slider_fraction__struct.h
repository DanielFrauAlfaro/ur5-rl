// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ur_msgs:srv/SetSpeedSliderFraction.idl
// generated code does not contain a copyright notice

#ifndef UR_MSGS__SRV__DETAIL__SET_SPEED_SLIDER_FRACTION__STRUCT_H_
#define UR_MSGS__SRV__DETAIL__SET_SPEED_SLIDER_FRACTION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/SetSpeedSliderFraction in the package ur_msgs.
typedef struct ur_msgs__srv__SetSpeedSliderFraction_Request
{
  double speed_slider_fraction;
} ur_msgs__srv__SetSpeedSliderFraction_Request;

// Struct for a sequence of ur_msgs__srv__SetSpeedSliderFraction_Request.
typedef struct ur_msgs__srv__SetSpeedSliderFraction_Request__Sequence
{
  ur_msgs__srv__SetSpeedSliderFraction_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ur_msgs__srv__SetSpeedSliderFraction_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetSpeedSliderFraction in the package ur_msgs.
typedef struct ur_msgs__srv__SetSpeedSliderFraction_Response
{
  bool success;
} ur_msgs__srv__SetSpeedSliderFraction_Response;

// Struct for a sequence of ur_msgs__srv__SetSpeedSliderFraction_Response.
typedef struct ur_msgs__srv__SetSpeedSliderFraction_Response__Sequence
{
  ur_msgs__srv__SetSpeedSliderFraction_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ur_msgs__srv__SetSpeedSliderFraction_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // UR_MSGS__SRV__DETAIL__SET_SPEED_SLIDER_FRACTION__STRUCT_H_
