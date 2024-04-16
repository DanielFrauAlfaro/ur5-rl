// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ur_msgs:msg/RobotModeDataMsg.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/robot_mode_data_msg__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
ur_msgs__msg__RobotModeDataMsg__init(ur_msgs__msg__RobotModeDataMsg * msg)
{
  if (!msg) {
    return false;
  }
  // timestamp
  // is_robot_connected
  // is_real_robot_enabled
  // is_power_on_robot
  // is_emergency_stopped
  // is_protective_stopped
  // is_program_running
  // is_program_paused
  return true;
}

void
ur_msgs__msg__RobotModeDataMsg__fini(ur_msgs__msg__RobotModeDataMsg * msg)
{
  if (!msg) {
    return;
  }
  // timestamp
  // is_robot_connected
  // is_real_robot_enabled
  // is_power_on_robot
  // is_emergency_stopped
  // is_protective_stopped
  // is_program_running
  // is_program_paused
}

bool
ur_msgs__msg__RobotModeDataMsg__are_equal(const ur_msgs__msg__RobotModeDataMsg * lhs, const ur_msgs__msg__RobotModeDataMsg * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // timestamp
  if (lhs->timestamp != rhs->timestamp) {
    return false;
  }
  // is_robot_connected
  if (lhs->is_robot_connected != rhs->is_robot_connected) {
    return false;
  }
  // is_real_robot_enabled
  if (lhs->is_real_robot_enabled != rhs->is_real_robot_enabled) {
    return false;
  }
  // is_power_on_robot
  if (lhs->is_power_on_robot != rhs->is_power_on_robot) {
    return false;
  }
  // is_emergency_stopped
  if (lhs->is_emergency_stopped != rhs->is_emergency_stopped) {
    return false;
  }
  // is_protective_stopped
  if (lhs->is_protective_stopped != rhs->is_protective_stopped) {
    return false;
  }
  // is_program_running
  if (lhs->is_program_running != rhs->is_program_running) {
    return false;
  }
  // is_program_paused
  if (lhs->is_program_paused != rhs->is_program_paused) {
    return false;
  }
  return true;
}

bool
ur_msgs__msg__RobotModeDataMsg__copy(
  const ur_msgs__msg__RobotModeDataMsg * input,
  ur_msgs__msg__RobotModeDataMsg * output)
{
  if (!input || !output) {
    return false;
  }
  // timestamp
  output->timestamp = input->timestamp;
  // is_robot_connected
  output->is_robot_connected = input->is_robot_connected;
  // is_real_robot_enabled
  output->is_real_robot_enabled = input->is_real_robot_enabled;
  // is_power_on_robot
  output->is_power_on_robot = input->is_power_on_robot;
  // is_emergency_stopped
  output->is_emergency_stopped = input->is_emergency_stopped;
  // is_protective_stopped
  output->is_protective_stopped = input->is_protective_stopped;
  // is_program_running
  output->is_program_running = input->is_program_running;
  // is_program_paused
  output->is_program_paused = input->is_program_paused;
  return true;
}

ur_msgs__msg__RobotModeDataMsg *
ur_msgs__msg__RobotModeDataMsg__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__RobotModeDataMsg * msg = (ur_msgs__msg__RobotModeDataMsg *)allocator.allocate(sizeof(ur_msgs__msg__RobotModeDataMsg), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ur_msgs__msg__RobotModeDataMsg));
  bool success = ur_msgs__msg__RobotModeDataMsg__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ur_msgs__msg__RobotModeDataMsg__destroy(ur_msgs__msg__RobotModeDataMsg * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ur_msgs__msg__RobotModeDataMsg__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ur_msgs__msg__RobotModeDataMsg__Sequence__init(ur_msgs__msg__RobotModeDataMsg__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__RobotModeDataMsg * data = NULL;

  if (size) {
    data = (ur_msgs__msg__RobotModeDataMsg *)allocator.zero_allocate(size, sizeof(ur_msgs__msg__RobotModeDataMsg), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ur_msgs__msg__RobotModeDataMsg__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ur_msgs__msg__RobotModeDataMsg__fini(&data[i - 1]);
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
ur_msgs__msg__RobotModeDataMsg__Sequence__fini(ur_msgs__msg__RobotModeDataMsg__Sequence * array)
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
      ur_msgs__msg__RobotModeDataMsg__fini(&array->data[i]);
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

ur_msgs__msg__RobotModeDataMsg__Sequence *
ur_msgs__msg__RobotModeDataMsg__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__RobotModeDataMsg__Sequence * array = (ur_msgs__msg__RobotModeDataMsg__Sequence *)allocator.allocate(sizeof(ur_msgs__msg__RobotModeDataMsg__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ur_msgs__msg__RobotModeDataMsg__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ur_msgs__msg__RobotModeDataMsg__Sequence__destroy(ur_msgs__msg__RobotModeDataMsg__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ur_msgs__msg__RobotModeDataMsg__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ur_msgs__msg__RobotModeDataMsg__Sequence__are_equal(const ur_msgs__msg__RobotModeDataMsg__Sequence * lhs, const ur_msgs__msg__RobotModeDataMsg__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ur_msgs__msg__RobotModeDataMsg__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ur_msgs__msg__RobotModeDataMsg__Sequence__copy(
  const ur_msgs__msg__RobotModeDataMsg__Sequence * input,
  ur_msgs__msg__RobotModeDataMsg__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ur_msgs__msg__RobotModeDataMsg);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ur_msgs__msg__RobotModeDataMsg * data =
      (ur_msgs__msg__RobotModeDataMsg *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ur_msgs__msg__RobotModeDataMsg__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ur_msgs__msg__RobotModeDataMsg__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ur_msgs__msg__RobotModeDataMsg__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
