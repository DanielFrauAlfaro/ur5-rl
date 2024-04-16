// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ur_msgs:msg/RobotStateRTMsg.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/robot_state_rt_msg__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `q_target`
// Member `qd_target`
// Member `qdd_target`
// Member `i_target`
// Member `m_target`
// Member `q_actual`
// Member `qd_actual`
// Member `i_actual`
// Member `tool_acc_values`
// Member `tcp_force`
// Member `tool_vector`
// Member `tcp_speed`
// Member `motor_temperatures`
// Member `joint_modes`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
ur_msgs__msg__RobotStateRTMsg__init(ur_msgs__msg__RobotStateRTMsg * msg)
{
  if (!msg) {
    return false;
  }
  // time
  // q_target
  if (!rosidl_runtime_c__double__Sequence__init(&msg->q_target, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // qd_target
  if (!rosidl_runtime_c__double__Sequence__init(&msg->qd_target, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // qdd_target
  if (!rosidl_runtime_c__double__Sequence__init(&msg->qdd_target, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // i_target
  if (!rosidl_runtime_c__double__Sequence__init(&msg->i_target, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // m_target
  if (!rosidl_runtime_c__double__Sequence__init(&msg->m_target, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // q_actual
  if (!rosidl_runtime_c__double__Sequence__init(&msg->q_actual, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // qd_actual
  if (!rosidl_runtime_c__double__Sequence__init(&msg->qd_actual, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // i_actual
  if (!rosidl_runtime_c__double__Sequence__init(&msg->i_actual, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // tool_acc_values
  if (!rosidl_runtime_c__double__Sequence__init(&msg->tool_acc_values, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // tcp_force
  if (!rosidl_runtime_c__double__Sequence__init(&msg->tcp_force, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // tool_vector
  if (!rosidl_runtime_c__double__Sequence__init(&msg->tool_vector, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // tcp_speed
  if (!rosidl_runtime_c__double__Sequence__init(&msg->tcp_speed, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // digital_input_bits
  // motor_temperatures
  if (!rosidl_runtime_c__double__Sequence__init(&msg->motor_temperatures, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  // controller_timer
  // test_value
  // robot_mode
  // joint_modes
  if (!rosidl_runtime_c__double__Sequence__init(&msg->joint_modes, 0)) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
    return false;
  }
  return true;
}

void
ur_msgs__msg__RobotStateRTMsg__fini(ur_msgs__msg__RobotStateRTMsg * msg)
{
  if (!msg) {
    return;
  }
  // time
  // q_target
  rosidl_runtime_c__double__Sequence__fini(&msg->q_target);
  // qd_target
  rosidl_runtime_c__double__Sequence__fini(&msg->qd_target);
  // qdd_target
  rosidl_runtime_c__double__Sequence__fini(&msg->qdd_target);
  // i_target
  rosidl_runtime_c__double__Sequence__fini(&msg->i_target);
  // m_target
  rosidl_runtime_c__double__Sequence__fini(&msg->m_target);
  // q_actual
  rosidl_runtime_c__double__Sequence__fini(&msg->q_actual);
  // qd_actual
  rosidl_runtime_c__double__Sequence__fini(&msg->qd_actual);
  // i_actual
  rosidl_runtime_c__double__Sequence__fini(&msg->i_actual);
  // tool_acc_values
  rosidl_runtime_c__double__Sequence__fini(&msg->tool_acc_values);
  // tcp_force
  rosidl_runtime_c__double__Sequence__fini(&msg->tcp_force);
  // tool_vector
  rosidl_runtime_c__double__Sequence__fini(&msg->tool_vector);
  // tcp_speed
  rosidl_runtime_c__double__Sequence__fini(&msg->tcp_speed);
  // digital_input_bits
  // motor_temperatures
  rosidl_runtime_c__double__Sequence__fini(&msg->motor_temperatures);
  // controller_timer
  // test_value
  // robot_mode
  // joint_modes
  rosidl_runtime_c__double__Sequence__fini(&msg->joint_modes);
}

bool
ur_msgs__msg__RobotStateRTMsg__are_equal(const ur_msgs__msg__RobotStateRTMsg * lhs, const ur_msgs__msg__RobotStateRTMsg * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // time
  if (lhs->time != rhs->time) {
    return false;
  }
  // q_target
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->q_target), &(rhs->q_target)))
  {
    return false;
  }
  // qd_target
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->qd_target), &(rhs->qd_target)))
  {
    return false;
  }
  // qdd_target
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->qdd_target), &(rhs->qdd_target)))
  {
    return false;
  }
  // i_target
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->i_target), &(rhs->i_target)))
  {
    return false;
  }
  // m_target
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->m_target), &(rhs->m_target)))
  {
    return false;
  }
  // q_actual
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->q_actual), &(rhs->q_actual)))
  {
    return false;
  }
  // qd_actual
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->qd_actual), &(rhs->qd_actual)))
  {
    return false;
  }
  // i_actual
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->i_actual), &(rhs->i_actual)))
  {
    return false;
  }
  // tool_acc_values
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->tool_acc_values), &(rhs->tool_acc_values)))
  {
    return false;
  }
  // tcp_force
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->tcp_force), &(rhs->tcp_force)))
  {
    return false;
  }
  // tool_vector
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->tool_vector), &(rhs->tool_vector)))
  {
    return false;
  }
  // tcp_speed
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->tcp_speed), &(rhs->tcp_speed)))
  {
    return false;
  }
  // digital_input_bits
  if (lhs->digital_input_bits != rhs->digital_input_bits) {
    return false;
  }
  // motor_temperatures
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->motor_temperatures), &(rhs->motor_temperatures)))
  {
    return false;
  }
  // controller_timer
  if (lhs->controller_timer != rhs->controller_timer) {
    return false;
  }
  // test_value
  if (lhs->test_value != rhs->test_value) {
    return false;
  }
  // robot_mode
  if (lhs->robot_mode != rhs->robot_mode) {
    return false;
  }
  // joint_modes
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->joint_modes), &(rhs->joint_modes)))
  {
    return false;
  }
  return true;
}

bool
ur_msgs__msg__RobotStateRTMsg__copy(
  const ur_msgs__msg__RobotStateRTMsg * input,
  ur_msgs__msg__RobotStateRTMsg * output)
{
  if (!input || !output) {
    return false;
  }
  // time
  output->time = input->time;
  // q_target
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->q_target), &(output->q_target)))
  {
    return false;
  }
  // qd_target
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->qd_target), &(output->qd_target)))
  {
    return false;
  }
  // qdd_target
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->qdd_target), &(output->qdd_target)))
  {
    return false;
  }
  // i_target
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->i_target), &(output->i_target)))
  {
    return false;
  }
  // m_target
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->m_target), &(output->m_target)))
  {
    return false;
  }
  // q_actual
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->q_actual), &(output->q_actual)))
  {
    return false;
  }
  // qd_actual
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->qd_actual), &(output->qd_actual)))
  {
    return false;
  }
  // i_actual
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->i_actual), &(output->i_actual)))
  {
    return false;
  }
  // tool_acc_values
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->tool_acc_values), &(output->tool_acc_values)))
  {
    return false;
  }
  // tcp_force
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->tcp_force), &(output->tcp_force)))
  {
    return false;
  }
  // tool_vector
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->tool_vector), &(output->tool_vector)))
  {
    return false;
  }
  // tcp_speed
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->tcp_speed), &(output->tcp_speed)))
  {
    return false;
  }
  // digital_input_bits
  output->digital_input_bits = input->digital_input_bits;
  // motor_temperatures
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->motor_temperatures), &(output->motor_temperatures)))
  {
    return false;
  }
  // controller_timer
  output->controller_timer = input->controller_timer;
  // test_value
  output->test_value = input->test_value;
  // robot_mode
  output->robot_mode = input->robot_mode;
  // joint_modes
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->joint_modes), &(output->joint_modes)))
  {
    return false;
  }
  return true;
}

ur_msgs__msg__RobotStateRTMsg *
ur_msgs__msg__RobotStateRTMsg__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__RobotStateRTMsg * msg = (ur_msgs__msg__RobotStateRTMsg *)allocator.allocate(sizeof(ur_msgs__msg__RobotStateRTMsg), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ur_msgs__msg__RobotStateRTMsg));
  bool success = ur_msgs__msg__RobotStateRTMsg__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ur_msgs__msg__RobotStateRTMsg__destroy(ur_msgs__msg__RobotStateRTMsg * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ur_msgs__msg__RobotStateRTMsg__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ur_msgs__msg__RobotStateRTMsg__Sequence__init(ur_msgs__msg__RobotStateRTMsg__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__RobotStateRTMsg * data = NULL;

  if (size) {
    data = (ur_msgs__msg__RobotStateRTMsg *)allocator.zero_allocate(size, sizeof(ur_msgs__msg__RobotStateRTMsg), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ur_msgs__msg__RobotStateRTMsg__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ur_msgs__msg__RobotStateRTMsg__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
ur_msgs__msg__RobotStateRTMsg__Sequence__fini(ur_msgs__msg__RobotStateRTMsg__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      ur_msgs__msg__RobotStateRTMsg__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

ur_msgs__msg__RobotStateRTMsg__Sequence *
ur_msgs__msg__RobotStateRTMsg__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__RobotStateRTMsg__Sequence * array = (ur_msgs__msg__RobotStateRTMsg__Sequence *)allocator.allocate(sizeof(ur_msgs__msg__RobotStateRTMsg__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ur_msgs__msg__RobotStateRTMsg__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ur_msgs__msg__RobotStateRTMsg__Sequence__destroy(ur_msgs__msg__RobotStateRTMsg__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ur_msgs__msg__RobotStateRTMsg__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ur_msgs__msg__RobotStateRTMsg__Sequence__are_equal(const ur_msgs__msg__RobotStateRTMsg__Sequence * lhs, const ur_msgs__msg__RobotStateRTMsg__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ur_msgs__msg__RobotStateRTMsg__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ur_msgs__msg__RobotStateRTMsg__Sequence__copy(
  const ur_msgs__msg__RobotStateRTMsg__Sequence * input,
  ur_msgs__msg__RobotStateRTMsg__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ur_msgs__msg__RobotStateRTMsg);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ur_msgs__msg__RobotStateRTMsg * data =
      (ur_msgs__msg__RobotStateRTMsg *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ur_msgs__msg__RobotStateRTMsg__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ur_msgs__msg__RobotStateRTMsg__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ur_msgs__msg__RobotStateRTMsg__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
