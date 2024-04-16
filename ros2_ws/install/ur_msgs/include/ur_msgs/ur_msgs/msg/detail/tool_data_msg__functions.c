// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ur_msgs:msg/ToolDataMsg.idl
// generated code does not contain a copyright notice
#include "ur_msgs/msg/detail/tool_data_msg__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
ur_msgs__msg__ToolDataMsg__init(ur_msgs__msg__ToolDataMsg * msg)
{
  if (!msg) {
    return false;
  }
  // analog_input_range2
  // analog_input_range3
  // analog_input2
  // analog_input3
  // tool_voltage_48v
  // tool_output_voltage
  // tool_current
  // tool_temperature
  // tool_mode
  return true;
}

void
ur_msgs__msg__ToolDataMsg__fini(ur_msgs__msg__ToolDataMsg * msg)
{
  if (!msg) {
    return;
  }
  // analog_input_range2
  // analog_input_range3
  // analog_input2
  // analog_input3
  // tool_voltage_48v
  // tool_output_voltage
  // tool_current
  // tool_temperature
  // tool_mode
}

bool
ur_msgs__msg__ToolDataMsg__are_equal(const ur_msgs__msg__ToolDataMsg * lhs, const ur_msgs__msg__ToolDataMsg * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // analog_input_range2
  if (lhs->analog_input_range2 != rhs->analog_input_range2) {
    return false;
  }
  // analog_input_range3
  if (lhs->analog_input_range3 != rhs->analog_input_range3) {
    return false;
  }
  // analog_input2
  if (lhs->analog_input2 != rhs->analog_input2) {
    return false;
  }
  // analog_input3
  if (lhs->analog_input3 != rhs->analog_input3) {
    return false;
  }
  // tool_voltage_48v
  if (lhs->tool_voltage_48v != rhs->tool_voltage_48v) {
    return false;
  }
  // tool_output_voltage
  if (lhs->tool_output_voltage != rhs->tool_output_voltage) {
    return false;
  }
  // tool_current
  if (lhs->tool_current != rhs->tool_current) {
    return false;
  }
  // tool_temperature
  if (lhs->tool_temperature != rhs->tool_temperature) {
    return false;
  }
  // tool_mode
  if (lhs->tool_mode != rhs->tool_mode) {
    return false;
  }
  return true;
}

bool
ur_msgs__msg__ToolDataMsg__copy(
  const ur_msgs__msg__ToolDataMsg * input,
  ur_msgs__msg__ToolDataMsg * output)
{
  if (!input || !output) {
    return false;
  }
  // analog_input_range2
  output->analog_input_range2 = input->analog_input_range2;
  // analog_input_range3
  output->analog_input_range3 = input->analog_input_range3;
  // analog_input2
  output->analog_input2 = input->analog_input2;
  // analog_input3
  output->analog_input3 = input->analog_input3;
  // tool_voltage_48v
  output->tool_voltage_48v = input->tool_voltage_48v;
  // tool_output_voltage
  output->tool_output_voltage = input->tool_output_voltage;
  // tool_current
  output->tool_current = input->tool_current;
  // tool_temperature
  output->tool_temperature = input->tool_temperature;
  // tool_mode
  output->tool_mode = input->tool_mode;
  return true;
}

ur_msgs__msg__ToolDataMsg *
ur_msgs__msg__ToolDataMsg__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__ToolDataMsg * msg = (ur_msgs__msg__ToolDataMsg *)allocator.allocate(sizeof(ur_msgs__msg__ToolDataMsg), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ur_msgs__msg__ToolDataMsg));
  bool success = ur_msgs__msg__ToolDataMsg__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ur_msgs__msg__ToolDataMsg__destroy(ur_msgs__msg__ToolDataMsg * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ur_msgs__msg__ToolDataMsg__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ur_msgs__msg__ToolDataMsg__Sequence__init(ur_msgs__msg__ToolDataMsg__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__ToolDataMsg * data = NULL;

  if (size) {
    data = (ur_msgs__msg__ToolDataMsg *)allocator.zero_allocate(size, sizeof(ur_msgs__msg__ToolDataMsg), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ur_msgs__msg__ToolDataMsg__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ur_msgs__msg__ToolDataMsg__fini(&data[i - 1]);
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
ur_msgs__msg__ToolDataMsg__Sequence__fini(ur_msgs__msg__ToolDataMsg__Sequence * array)
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
      ur_msgs__msg__ToolDataMsg__fini(&array->data[i]);
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

ur_msgs__msg__ToolDataMsg__Sequence *
ur_msgs__msg__ToolDataMsg__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ur_msgs__msg__ToolDataMsg__Sequence * array = (ur_msgs__msg__ToolDataMsg__Sequence *)allocator.allocate(sizeof(ur_msgs__msg__ToolDataMsg__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ur_msgs__msg__ToolDataMsg__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ur_msgs__msg__ToolDataMsg__Sequence__destroy(ur_msgs__msg__ToolDataMsg__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ur_msgs__msg__ToolDataMsg__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ur_msgs__msg__ToolDataMsg__Sequence__are_equal(const ur_msgs__msg__ToolDataMsg__Sequence * lhs, const ur_msgs__msg__ToolDataMsg__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ur_msgs__msg__ToolDataMsg__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ur_msgs__msg__ToolDataMsg__Sequence__copy(
  const ur_msgs__msg__ToolDataMsg__Sequence * input,
  ur_msgs__msg__ToolDataMsg__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ur_msgs__msg__ToolDataMsg);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ur_msgs__msg__ToolDataMsg * data =
      (ur_msgs__msg__ToolDataMsg *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ur_msgs__msg__ToolDataMsg__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ur_msgs__msg__ToolDataMsg__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ur_msgs__msg__ToolDataMsg__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
