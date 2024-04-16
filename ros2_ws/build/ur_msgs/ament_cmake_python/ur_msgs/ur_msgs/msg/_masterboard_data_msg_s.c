// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from ur_msgs:msg/MasterboardDataMsg.idl
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
#include "ur_msgs/msg/detail/masterboard_data_msg__struct.h"
#include "ur_msgs/msg/detail/masterboard_data_msg__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool ur_msgs__msg__masterboard_data_msg__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[53];
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
    assert(strncmp("ur_msgs.msg._masterboard_data_msg.MasterboardDataMsg", full_classname_dest, 52) == 0);
  }
  ur_msgs__msg__MasterboardDataMsg * ros_message = _ros_message;
  {  // digital_input_bits
    PyObject * field = PyObject_GetAttrString(_pymsg, "digital_input_bits");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->digital_input_bits = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // digital_output_bits
    PyObject * field = PyObject_GetAttrString(_pymsg, "digital_output_bits");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->digital_output_bits = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // analog_input_range0
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_input_range0");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->analog_input_range0 = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // analog_input_range1
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_input_range1");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->analog_input_range1 = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // analog_input0
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_input0");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->analog_input0 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // analog_input1
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_input1");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->analog_input1 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // analog_output_domain0
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_output_domain0");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->analog_output_domain0 = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // analog_output_domain1
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_output_domain1");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->analog_output_domain1 = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // analog_output0
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_output0");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->analog_output0 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // analog_output1
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_output1");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->analog_output1 = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // masterboard_temperature
    PyObject * field = PyObject_GetAttrString(_pymsg, "masterboard_temperature");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->masterboard_temperature = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // robot_voltage_48v
    PyObject * field = PyObject_GetAttrString(_pymsg, "robot_voltage_48v");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->robot_voltage_48v = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // robot_current
    PyObject * field = PyObject_GetAttrString(_pymsg, "robot_current");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->robot_current = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // master_io_current
    PyObject * field = PyObject_GetAttrString(_pymsg, "master_io_current");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->master_io_current = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // master_safety_state
    PyObject * field = PyObject_GetAttrString(_pymsg, "master_safety_state");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->master_safety_state = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // master_onoff_state
    PyObject * field = PyObject_GetAttrString(_pymsg, "master_onoff_state");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->master_onoff_state = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * ur_msgs__msg__masterboard_data_msg__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of MasterboardDataMsg */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("ur_msgs.msg._masterboard_data_msg");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "MasterboardDataMsg");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  ur_msgs__msg__MasterboardDataMsg * ros_message = (ur_msgs__msg__MasterboardDataMsg *)raw_ros_message;
  {  // digital_input_bits
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->digital_input_bits);
    {
      int rc = PyObject_SetAttrString(_pymessage, "digital_input_bits", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // digital_output_bits
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->digital_output_bits);
    {
      int rc = PyObject_SetAttrString(_pymessage, "digital_output_bits", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_input_range0
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->analog_input_range0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_input_range0", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_input_range1
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->analog_input_range1);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_input_range1", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_input0
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->analog_input0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_input0", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_input1
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->analog_input1);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_input1", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_output_domain0
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->analog_output_domain0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_output_domain0", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_output_domain1
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->analog_output_domain1);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_output_domain1", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_output0
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->analog_output0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_output0", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_output1
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->analog_output1);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_output1", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // masterboard_temperature
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->masterboard_temperature);
    {
      int rc = PyObject_SetAttrString(_pymessage, "masterboard_temperature", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // robot_voltage_48v
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->robot_voltage_48v);
    {
      int rc = PyObject_SetAttrString(_pymessage, "robot_voltage_48v", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // robot_current
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->robot_current);
    {
      int rc = PyObject_SetAttrString(_pymessage, "robot_current", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // master_io_current
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->master_io_current);
    {
      int rc = PyObject_SetAttrString(_pymessage, "master_io_current", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // master_safety_state
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->master_safety_state);
    {
      int rc = PyObject_SetAttrString(_pymessage, "master_safety_state", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // master_onoff_state
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->master_onoff_state);
    {
      int rc = PyObject_SetAttrString(_pymessage, "master_onoff_state", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
