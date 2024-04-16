// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ur_msgs:msg/Analog.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/analog__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
ur_msgs__msg__Analog__init(ur_msgs__msg__Analog * msg)
{
  if (!msg) {
    return false;
  }
  // pin
  // domain
  // state
  return true;
}

void
ur_msgs__msg__Analog__fini(ur_msgs__msg__Analog * msg)
{
  if (!msg) {
    return;
  }
  // pin
  // domain
  // state
}

bool
ur_msgs__msg__Analog__are_equal(const ur_msgs__msg__Analog * lhs, const ur_msgs__msg__Analog * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // pin
  if (lhs->pin != rhs->pin) {
    return false;
  }
  // domain
  if (lhs->domain != rhs->domain) {
    return false;
  }
  // state
  if (lhs->state != rhs->state) {
    return false;
  }
  return true;
}

bool
ur_msgs__msg__Analog__copy(
  const ur_msgs__msg__Analog * input,
  ur_msgs__msg__Analog * output)
{
  if (!input || !output) {
    return false;
  }
  // pin
  output->pin = input->pin;
  // domain
  output->domain = input->domain;
  // state
  output->state = input->state;
  return true;
}

ur_msgs__msg__Analog *
ur_msgs__msg__Analog__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__Analog * msg = (ur_msgs__msg__Analog *)allocator.allocate(sizeof(ur_msgs__msg__Analog), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ur_msgs__msg__Analog));
  bool success = ur_msgs__msg__Analog__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ur_msgs__msg__Analog__destroy(ur_msgs__msg__Analog * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ur_msgs__msg__Analog__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ur_msgs__msg__Analog__Sequence__init(ur_msgs__msg__Analog__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__Analog * data = NULL;

  if (size) {
    data = (ur_msgs__msg__Analog *)allocator.zero_allocate(size, sizeof(ur_msgs__msg__Analog), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ur_msgs__msg__Analog__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ur_msgs__msg__Analog__fini(&data[i - 1]);
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
ur_msgs__msg__Analog__Sequence__fini(ur_msgs__msg__Analog__Sequence * array)
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
      ur_msgs__msg__Analog__fini(&array->data[i]);
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

ur_msgs__msg__Analog__Sequence *
ur_msgs__msg__Analog__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__Analog__Sequence * array = (ur_msgs__msg__Analog__Sequence *)allocator.allocate(sizeof(ur_msgs__msg__Analog__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ur_msgs__msg__Analog__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ur_msgs__msg__Analog__Sequence__destroy(ur_msgs__msg__Analog__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ur_msgs__msg__Analog__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ur_msgs__msg__Analog__Sequence__are_equal(const ur_msgs__msg__Analog__Sequence * lhs, const ur_msgs__msg__Analog__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ur_msgs__msg__Analog__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ur_msgs__msg__Analog__Sequence__copy(
  const ur_msgs__msg__Analog__Sequence * input,
  ur_msgs__msg__Analog__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ur_msgs__msg__Analog);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ur_msgs__msg__Analog * data =
      (ur_msgs__msg__Analog *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ur_msgs__msg__Analog__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ur_msgs__msg__Analog__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ur_msgs__msg__Analog__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
