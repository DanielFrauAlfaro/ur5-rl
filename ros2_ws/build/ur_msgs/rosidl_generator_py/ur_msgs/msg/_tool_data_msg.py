# generated from rosidl_generator_py/resource/_idl.py.em
# with input from ur_msgs:msg/ToolDataMsg.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ToolDataMsg(type):
    """Metaclass of message 'ToolDataMsg'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'ANALOG_INPUT_RANGE_CURRENT': 0,
        'ANALOG_INPUT_RANGE_VOLTAGE': 1,
        'TOOL_BOOTLOADER_MODE': 249,
        'TOOL_RUNNING_MODE': 253,
        'TOOL_IDLE_MODE': 255,
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('ur_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'ur_msgs.msg.ToolDataMsg')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__tool_data_msg
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__tool_data_msg
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__tool_data_msg
            cls._TYPE_SUPPORT = module.type_support_msg__msg__tool_data_msg
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__tool_data_msg

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'ANALOG_INPUT_RANGE_CURRENT': cls.__constants['ANALOG_INPUT_RANGE_CURRENT'],
            'ANALOG_INPUT_RANGE_VOLTAGE': cls.__constants['ANALOG_INPUT_RANGE_VOLTAGE'],
            'TOOL_BOOTLOADER_MODE': cls.__constants['TOOL_BOOTLOADER_MODE'],
            'TOOL_RUNNING_MODE': cls.__constants['TOOL_RUNNING_MODE'],
            'TOOL_IDLE_MODE': cls.__constants['TOOL_IDLE_MODE'],
        }

    @property
    def ANALOG_INPUT_RANGE_CURRENT(self):
        """Message constant 'ANALOG_INPUT_RANGE_CURRENT'."""
        return Metaclass_ToolDataMsg.__constants['ANALOG_INPUT_RANGE_CURRENT']

    @property
    def ANALOG_INPUT_RANGE_VOLTAGE(self):
        """Message constant 'ANALOG_INPUT_RANGE_VOLTAGE'."""
        return Metaclass_ToolDataMsg.__constants['ANALOG_INPUT_RANGE_VOLTAGE']

    @property
    def TOOL_BOOTLOADER_MODE(self):
        """Message constant 'TOOL_BOOTLOADER_MODE'."""
        return Metaclass_ToolDataMsg.__constants['TOOL_BOOTLOADER_MODE']

    @property
    def TOOL_RUNNING_MODE(self):
        """Message constant 'TOOL_RUNNING_MODE'."""
        return Metaclass_ToolDataMsg.__constants['TOOL_RUNNING_MODE']

    @property
    def TOOL_IDLE_MODE(self):
        """Message constant 'TOOL_IDLE_MODE'."""
        return Metaclass_ToolDataMsg.__constants['TOOL_IDLE_MODE']


class ToolDataMsg(metaclass=Metaclass_ToolDataMsg):
    """
    Message class 'ToolDataMsg'.

    Constants:
      ANALOG_INPUT_RANGE_CURRENT
      ANALOG_INPUT_RANGE_VOLTAGE
      TOOL_BOOTLOADER_MODE
      TOOL_RUNNING_MODE
      TOOL_IDLE_MODE
    """

    __slots__ = [
        '_analog_input_range2',
        '_analog_input_range3',
        '_analog_input2',
        '_analog_input3',
        '_tool_voltage_48v',
        '_tool_output_voltage',
        '_tool_current',
        '_tool_temperature',
        '_tool_mode',
    ]

    _fields_and_field_types = {
        'analog_input_range2': 'int8',
        'analog_input_range3': 'int8',
        'analog_input2': 'double',
        'analog_input3': 'double',
        'tool_voltage_48v': 'float',
        'tool_output_voltage': 'uint8',
        'tool_current': 'float',
        'tool_temperature': 'float',
        'tool_mode': 'uint8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.analog_input_range2 = kwargs.get('analog_input_range2', int())
        self.analog_input_range3 = kwargs.get('analog_input_range3', int())
        self.analog_input2 = kwargs.get('analog_input2', float())
        self.analog_input3 = kwargs.get('analog_input3', float())
        self.tool_voltage_48v = kwargs.get('tool_voltage_48v', float())
        self.tool_output_voltage = kwargs.get('tool_output_voltage', int())
        self.tool_current = kwargs.get('tool_current', float())
        self.tool_temperature = kwargs.get('tool_temperature', float())
        self.tool_mode = kwargs.get('tool_mode', int())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.analog_input_range2 != other.analog_input_range2:
            return False
        if self.analog_input_range3 != other.analog_input_range3:
            return False
        if self.analog_input2 != other.analog_input2:
            return False
        if self.analog_input3 != other.analog_input3:
            return False
        if self.tool_voltage_48v != other.tool_voltage_48v:
            return False
        if self.tool_output_voltage != other.tool_output_voltage:
            return False
        if self.tool_current != other.tool_current:
            return False
        if self.tool_temperature != other.tool_temperature:
            return False
        if self.tool_mode != other.tool_mode:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def analog_input_range2(self):
        """Message field 'analog_input_range2'."""
        return self._analog_input_range2

    @analog_input_range2.setter
    def analog_input_range2(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'analog_input_range2' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'analog_input_range2' field must be an integer in [-128, 127]"
        self._analog_input_range2 = value

    @builtins.property
    def analog_input_range3(self):
        """Message field 'analog_input_range3'."""
        return self._analog_input_range3

    @analog_input_range3.setter
    def analog_input_range3(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'analog_input_range3' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'analog_input_range3' field must be an integer in [-128, 127]"
        self._analog_input_range3 = value

    @builtins.property
    def analog_input2(self):
        """Message field 'analog_input2'."""
        return self._analog_input2

    @analog_input2.setter
    def analog_input2(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'analog_input2' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'analog_input2' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._analog_input2 = value

    @builtins.property
    def analog_input3(self):
        """Message field 'analog_input3'."""
        return self._analog_input3

    @analog_input3.setter
    def analog_input3(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'analog_input3' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'analog_input3' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._analog_input3 = value

    @builtins.property
    def tool_voltage_48v(self):
        """Message field 'tool_voltage_48v'."""
        return self._tool_voltage_48v

    @tool_voltage_48v.setter
    def tool_voltage_48v(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'tool_voltage_48v' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'tool_voltage_48v' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._tool_voltage_48v = value

    @builtins.property
    def tool_output_voltage(self):
        """Message field 'tool_output_voltage'."""
        return self._tool_output_voltage

    @tool_output_voltage.setter
    def tool_output_voltage(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'tool_output_voltage' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'tool_output_voltage' field must be an unsigned integer in [0, 255]"
        self._tool_output_voltage = value

    @builtins.property
    def tool_current(self):
        """Message field 'tool_current'."""
        return self._tool_current

    @tool_current.setter
    def tool_current(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'tool_current' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'tool_current' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._tool_current = value

    @builtins.property
    def tool_temperature(self):
        """Message field 'tool_temperature'."""
        return self._tool_temperature

    @tool_temperature.setter
    def tool_temperature(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'tool_temperature' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'tool_temperature' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._tool_temperature = value

    @builtins.property
    def tool_mode(self):
        """Message field 'tool_mode'."""
        return self._tool_mode

    @tool_mode.setter
    def tool_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'tool_mode' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'tool_mode' field must be an unsigned integer in [0, 255]"
        self._tool_mode = value
