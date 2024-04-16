// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from ur_msgs:msg/ToolDataMsg.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "ur_msgs/msg/detail/tool_data_msg__struct.h"
#include "ur_msgs/msg/detail/tool_data_msg__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool ur_msgs__msg__tool_data_msg__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[39];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("ur_msgs.msg._tool_data_msg.ToolDataMsg", full_classname_dest, 38) == 0);
  }
  ur_msgs__msg__ToolDataMsg * ros_message = _ros_message;
  {  // analog_input_range2
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_input_range2");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->analog_input_range2 = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // analog_input_range3
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_input_range3");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->analog_input_range3 = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // analog_input2
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_input2");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->analog_input2 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // analog_input3
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_input3");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->analog_input3 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // tool_voltage_48v
    PyObject * field = PyObject_GetAttrString(_pymsg, "tool_voltage_48v");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->tool_voltage_48v = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // tool_output_voltage
    PyObject * field = PyObject_GetAttrString(_pymsg, "tool_output_voltage");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->tool_output_voltage = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // tool_current
    PyObject * field = PyObject_GetAttrString(_pymsg, "tool_current");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->tool_current = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // tool_temperature
    PyObject * field = PyObject_GetAttrString(_pymsg, "tool_temperature");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->tool_temperature = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // tool_mode
    PyObject * field = PyObject_GetAttrString(_pymsg, "tool_mode");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->tool_mode = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * ur_msgs__msg__tool_data_msg__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of ToolDataMsg */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("ur_msgs.msg._tool_data_msg");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "ToolDataMsg");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  ur_msgs__msg__ToolDataMsg * ros_message = (ur_msgs__msg__ToolDataMsg *)raw_ros_message;
  {  // analog_input_range2
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->analog_input_range2);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_input_range2", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_input_range3
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->analog_input_range3);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_input_range3", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_input2
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->analog_input2);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_input2", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_input3
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->analog_input3);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_input3", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tool_voltage_48v
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->tool_voltage_48v);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tool_voltage_48v", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tool_output_voltage
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->tool_output_voltage);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tool_output_voltage", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tool_current
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->tool_current);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tool_current", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tool_temperature
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->tool_temperature);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tool_temperature", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tool_mode
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->tool_mode);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tool_mode", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
