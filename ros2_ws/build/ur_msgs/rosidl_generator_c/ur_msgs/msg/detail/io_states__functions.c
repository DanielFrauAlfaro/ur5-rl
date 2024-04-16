// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ur_msgs:msg/IOStates.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/io_states__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `digital_in_states`
// Member `digital_out_states`
// Member `flag_states`
#include "ur_msgs/msg/detail/digital__functions.h"
// Member `analog_in_states`
// Member `analog_out_states`
#include "ur_msgs/msg/detail/analog__functions.h"

bool
ur_msgs__msg__IOStates__init(ur_msgs__msg__IOStates * msg)
{
  if (!msg) {
    return false;
  }
  // digital_in_states
  if (!ur_msgs__msg__Digital__Sequence__init(&msg->digital_in_states, 0)) {
    ur_msgs__msg__IOStates__fini(msg);
    return false;
  }
  // digital_out_states
  if (!ur_msgs__msg__Digital__Sequence__init(&msg->digital_out_states, 0)) {
    ur_msgs__msg__IOStates__fini(msg);
    return false;
  }
  // flag_states
  if (!ur_msgs__msg__Digital__Sequence__init(&msg->flag_states, 0)) {
    ur_msgs__msg__IOStates__fini(msg);
    return false;
  }
  // analog_in_states
  if (!ur_msgs__msg__Analog__Sequence__init(&msg->analog_in_states, 0)) {
    ur_msgs__msg__IOStates__fini(msg);
    return false;
  }
  // analog_out_states
  if (!ur_msgs__msg__Analog__Sequence__init(&msg->analog_out_states, 0)) {
    ur_msgs__msg__IOStates__fini(msg);
    return false;
  }
  return true;
}

void
ur_msgs__msg__IOStates__fini(ur_msgs__msg__IOStates * msg)
{
  if (!msg) {
    return;
  }
  // digital_in_states
  ur_msgs__msg__Digital__Sequence__fini(&msg->digital_in_states);
  // digital_out_states
  ur_msgs__msg__Digital__Sequence__fini(&msg->digital_out_states);
  // flag_states
  ur_msgs__msg__Digital__Sequence__fini(&msg->flag_states);
  // analog_in_states
  ur_msgs__msg__Analog__Sequence__fini(&msg->analog_in_states);
  // analog_out_states
  ur_msgs__msg__Analog__Sequence__fini(&msg->analog_out_states);
}

bool
ur_msgs__msg__IOStates__are_equal(const ur_msgs__msg__IOStates * lhs, const ur_msgs__msg__IOStates * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // digital_in_states
  if (!ur_msgs__msg__Digital__Sequence__are_equal(
      &(lhs->digital_in_states), &(rhs->digital_in_states)))
  {
    return false;
  }
  // digital_out_states
  if (!ur_msgs__msg__Digital__Sequence__are_equal(
      &(lhs->digital_out_states), &(rhs->digital_out_states)))
  {
    return false;
  }
  // flag_states
  if (!ur_msgs__msg__Digital__Sequence__are_equal(
      &(lhs->flag_states), &(rhs->flag_states)))
  {
    return false;
  }
  // analog_in_states
  if (!ur_msgs__msg__Analog__Sequence__are_equal(
      &(lhs->analog_in_states), &(rhs->analog_in_states)))
  {
    return false;
  }
  // analog_out_states
  if (!ur_msgs__msg__Analog__Sequence__are_equal(
      &(lhs->analog_out_states), &(rhs->analog_out_states)))
  {
    return false;
  }
  return true;
}

bool
ur_msgs__msg__IOStates__copy(
  const ur_msgs__msg__IOStates * input,
  ur_msgs__msg__IOStates * output)
{
  if (!input || !output) {
    return false;
  }
  // digital_in_states
  if (!ur_msgs__msg__Digital__Sequence__copy(
      &(input->digital_in_states), &(output->digital_in_states)))
  {
    return false;
  }
  // digital_out_states
  if (!ur_msgs__msg__Digital__Sequence__copy(
      &(input->digital_out_states), &(output->digital_out_states)))
  {
    return false;
  }
  // flag_states
  if (!ur_msgs__msg__Digital__Sequence__copy(
      &(input->flag_states), &(output->flag_states)))
  {
    return false;
  }
  // analog_in_states
  if (!ur_msgs__msg__Analog__Sequence__copy(
      &(input->analog_in_states), &(output->analog_in_states)))
  {
    return false;
  }
  // analog_out_states
  if (!ur_msgs__msg__Analog__Sequence__copy(
      &(input->analog_out_states), &(output->analog_out_states)))
  {
    return false;
  }
  return true;
}

ur_msgs__msg__IOStates *
ur_msgs__msg__IOStates__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__IOStates * msg = (ur_msgs__msg__IOStates *)allocator.allocate(sizeof(ur_msgs__msg__IOStates), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ur_msgs__msg__IOStates));
  bool success = ur_msgs__msg__IOStates__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ur_msgs__msg__IOStates__destroy(ur_msgs__msg__IOStates * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ur_msgs__msg__IOStates__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ur_msgs__msg__IOStates__Sequence__init(ur_msgs__msg__IOStates__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__IOStates * data = NULL;

  if (size) {
    data = (ur_msgs__msg__IOStates *)allocator.zero_allocate(size, sizeof(ur_msgs__msg__IOStates), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ur_msgs__msg__IOStates__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ur_msgs__msg__IOStates__fini(&data[i - 1]);
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
ur_msgs__msg__IOStates__Sequence__fini(ur_msgs__msg__IOStates__Sequence * array)
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
      ur_msgs__msg__IOStates__fini(&array->data[i]);
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

ur_msgs__msg__IOStates__Sequence *
ur_msgs__msg__IOStates__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__IOStates__Sequence * array = (ur_msgs__msg__IOStates__Sequence *)allocator.allocate(sizeof(ur_msgs__msg__IOStates__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ur_msgs__msg__IOStates__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ur_msgs__msg__IOStates__Sequence__destroy(ur_msgs__msg__IOStates__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ur_msgs__msg__IOStates__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ur_msgs__msg__IOStates__Sequence__are_equal(const ur_msgs__msg__IOStates__Sequence * lhs, const ur_msgs__msg__IOStates__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ur_msgs__msg__IOStates__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ur_msgs__msg__IOStates__Sequence__copy(
  const ur_msgs__msg__IOStates__Sequence * input,
  ur_msgs__msg__IOStates__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ur_msgs__msg__IOStates);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ur_msgs__msg__IOStates * data =
      (ur_msgs__msg__IOStates *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ur_msgs__msg__IOStates__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ur_msgs__msg__IOStates__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ur_msgs__msg__IOStates__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
