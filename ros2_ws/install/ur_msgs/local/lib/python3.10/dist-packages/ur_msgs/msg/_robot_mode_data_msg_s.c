// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from ur_msgs:msg/RobotModeDataMsg.idl
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
#include "ur_msgs/msg/detail/robot_mode_data_msg__struct.h"
#include "ur_msgs/msg/detail/robot_mode_data_msg__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool ur_msgs__msg__robot_mode_data_msg__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[50];
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
    assert(strncmp("ur_msgs.msg._robot_mode_data_msg.RobotModeDataMsg", full_classname_dest, 49) == 0);
  }
  ur_msgs__msg__RobotModeDataMsg * ros_message = _ros_message;
  {  // timestamp
    PyObject * field = PyObject_GetAttrString(_pymsg, "timestamp");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->timestamp = PyLong_AsUnsignedLongLong(field);
    Py_DECREF(field);
  }
  {  // is_robot_connected
    PyObject * field = PyObject_GetAttrString(_pymsg, "is_robot_connected");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->is_robot_connected = (Py_True == field);
    Py_DECREF(field);
  }
  {  // is_real_robot_enabled
    PyObject * field = PyObject_GetAttrString(_pymsg, "is_real_robot_enabled");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->is_real_robot_enabled = (Py_True == field);
    Py_DECREF(field);
  }
  {  // is_power_on_robot
    PyObject * field = PyObject_GetAttrString(_pymsg, "is_power_on_robot");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->is_power_on_robot = (Py_True == field);
    Py_DECREF(field);
  }
  {  // is_emergency_stopped
    PyObject * field = PyObject_GetAttrString(_pymsg, "is_emergency_stopped");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->is_emergency_stopped = (Py_True == field);
    Py_DECREF(field);
  }
  {  // is_protective_stopped
    PyObject * field = PyObject_GetAttrString(_pymsg, "is_protective_stopped");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->is_protective_stopped = (Py_True == field);
    Py_DECREF(field);
  }
  {  // is_program_running
    PyObject * field = PyObject_GetAttrString(_pymsg, "is_program_running");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->is_program_running = (Py_True == field);
    Py_DECREF(field);
  }
  {  // is_program_paused
    PyObject * field = PyObject_GetAttrString(_pymsg, "is_program_paused");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->is_program_paused = (Py_True == field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * ur_msgs__msg__robot_mode_data_msg__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of RobotModeDataMsg */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("ur_msgs.msg._robot_mode_data_msg");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "RobotModeDataMsg");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  ur_msgs__msg__RobotModeDataMsg * ros_message = (ur_msgs__msg__RobotModeDataMsg *)raw_ros_message;
  {  // timestamp
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLongLong(ros_message->timestamp);
    {
      int rc = PyObject_SetAttrString(_pymessage, "timestamp", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // is_robot_connected
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->is_robot_connected ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "is_robot_connected", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // is_real_robot_enabled
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->is_real_robot_enabled ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "is_real_robot_enabled", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // is_power_on_robot
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->is_power_on_robot ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "is_power_on_robot", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // is_emergency_stopped
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->is_emergency_stopped ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "is_emergency_stopped", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // is_protective_stopped
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->is_protective_stopped ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "is_protective_stopped", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // is_program_running
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->is_program_running ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "is_program_running", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // is_program_paused
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->is_program_paused ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "is_program_paused", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
