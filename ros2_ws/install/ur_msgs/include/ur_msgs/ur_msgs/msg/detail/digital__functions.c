// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ur_msgs:msg/Digital.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/digital__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
ur_msgs__msg__Digital__init(ur_msgs__msg__Digital * msg)
{
  if (!msg) {
    return false;
  }
  // pin
  // state
  return true;
}

void
ur_msgs__msg__Digital__fini(ur_msgs__msg__Digital * msg)
{
  if (!msg) {
    return;
  }
  // pin
  // state
}

bool
ur_msgs__msg__Digital__are_equal(const ur_msgs__msg__Digital * lhs, const ur_msgs__msg__Digital * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // pin
  if (lhs->pin != rhs->pin) {
    return false;
  }
  // state
  if (lhs->state != rhs->state) {
    return false;
  }
  return true;
}

bool
ur_msgs__msg__Digital__copy(
  const ur_msgs__msg__Digital * input,
  ur_msgs__msg__Digital * output)
{
  if (!input || !output) {
    return false;
  }
  // pin
  output->pin = input->pin;
  // state
  output->state = input->state;
  return true;
}

ur_msgs__msg__Digital *
ur_msgs__msg__Digital__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__Digital * msg = (ur_msgs__msg__Digital *)allocator.allocate(sizeof(ur_msgs__msg__Digital), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ur_msgs__msg__Digital));
  bool success = ur_msgs__msg__Digital__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ur_msgs__msg__Digital__destroy(ur_msgs__msg__Digital * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ur_msgs__msg__Digital__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ur_msgs__msg__Digital__Sequence__init(ur_msgs__msg__Digital__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__Digital * data = NULL;

  if (size) {
    data = (ur_msgs__msg__Digital *)allocator.zero_allocate(size, sizeof(ur_msgs__msg__Digital), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ur_msgs__msg__Digital__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ur_msgs__msg__Digital__fini(&data[i - 1]);
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
ur_msgs__msg__Digital__Sequence__fini(ur_msgs__msg__Digital__Sequence * array)
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
      ur_msgs__msg__Digital__fini(&array->data[i]);
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

ur_msgs__msg__Digital__Sequence *
ur_msgs__msg__Digital__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__Digital__Sequence * array = (ur_msgs__msg__Digital__Sequence *)allocator.allocate(sizeof(ur_msgs__msg__Digital__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ur_msgs__msg__Digital__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ur_msgs__msg__Digital__Sequence__destroy(ur_msgs__msg__Digital__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ur_msgs__msg__Digital__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ur_msgs__msg__Digital__Sequence__are_equal(const ur_msgs__msg__Digital__Sequence * lhs, const ur_msgs__msg__Digital__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ur_msgs__msg__Digital__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ur_msgs__msg__Digital__Sequence__copy(
  const ur_msgs__msg__Digital__Sequence * input,
  ur_msgs__msg__Digital__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ur_msgs__msg__Digital);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ur_msgs__msg__Digital * data =
      (ur_msgs__msg__Digital *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ur_msgs__msg__Digital__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ur_msgs__msg__Digital__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ur_msgs__msg__Digital__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
