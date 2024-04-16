# generated from rosidl_generator_py/resource/_idl.py.em
# with input from ur_msgs:msg/MasterboardDataMsg.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_MasterboardDataMsg(type):
    """Metaclass of message 'MasterboardDataMsg'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
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
                'ur_msgs.msg.MasterboardDataMsg')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__masterboard_data_msg
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__masterboard_data_msg
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__masterboard_data_msg
            cls._TYPE_SUPPORT = module.type_support_msg__msg__masterboard_data_msg
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__masterboard_data_msg

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class MasterboardDataMsg(metaclass=Metaclass_MasterboardDataMsg):
    """Message class 'MasterboardDataMsg'."""

    __slots__ = [
        '_digital_input_bits',
        '_digital_output_bits',
        '_analog_input_range0',
        '_analog_input_range1',
        '_analog_input0',
        '_analog_input1',
        '_analog_output_domain0',
        '_analog_output_domain1',
        '_analog_output0',
        '_analog_output1',
        '_masterboard_temperature',
        '_robot_voltage_48v',
        '_robot_current',
        '_master_io_current',
        '_master_safety_state',
        '_master_onoff_state',
    ]

    _fields_and_field_types = {
        'digital_input_bits': 'uint32',
        'digital_output_bits': 'uint32',
        'analog_input_range0': 'int8',
        'analog_input_range1': 'int8',
        'analog_input0': 'double',
        'analog_input1': 'double',
        'analog_output_domain0': 'int8',
        'analog_output_domain1': 'int8',
        'analog_output0': 'double',
        'analog_output1': 'double',
        'masterboard_temperature': 'float',
        'robot_voltage_48v': 'float',
        'robot_current': 'float',
        'master_io_current': 'float',
        'master_safety_state': 'uint8',
        'master_onoff_state': 'uint8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.digital_input_bits = kwargs.get('digital_input_bits', int())
        self.digital_output_bits = kwargs.get('digital_output_bits', int())
        self.analog_input_range0 = kwargs.get('analog_input_range0', int())
        self.analog_input_range1 = kwargs.get('analog_input_range1', int())
        self.analog_input0 = kwargs.get('analog_input0', float())
        self.analog_input1 = kwargs.get('analog_input1', float())
        self.analog_output_domain0 = kwargs.get('analog_output_domain0', int())
        self.analog_output_domain1 = kwargs.get('analog_output_domain1', int())
        self.analog_output0 = kwargs.get('analog_output0', float())
        self.analog_output1 = kwargs.get('analog_output1', float())
        self.masterboard_temperature = kwargs.get('masterboard_temperature', float())
        self.robot_voltage_48v = kwargs.get('robot_voltage_48v', float())
        self.robot_current = kwargs.get('robot_current', float())
        self.master_io_current = kwargs.get('master_io_current', float())
        self.master_safety_state = kwargs.get('master_safety_state', int())
        self.master_onoff_state = kwargs.get('master_onoff_state', int())

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
        if self.digital_input_bits != other.digital_input_bits:
            return False
        if self.digital_output_bits != other.digital_output_bits:
            return False
        if self.analog_input_range0 != other.analog_input_range0:
            return False
        if self.analog_input_range1 != other.analog_input_range1:
            return False
        if self.analog_input0 != other.analog_input0:
            return False
        if self.analog_input1 != other.analog_input1:
            return False
        if self.analog_output_domain0 != other.analog_output_domain0:
            return False
        if self.analog_output_domain1 != other.analog_output_domain1:
            return False
        if self.analog_output0 != other.analog_output0:
            return False
        if self.analog_output1 != other.analog_output1:
            return False
        if self.masterboard_temperature != other.masterboard_temperature:
            return False
        if self.robot_voltage_48v != other.robot_voltage_48v:
            return False
        if self.robot_current != other.robot_current:
            return False
        if self.master_io_current != other.master_io_current:
            return False
        if self.master_safety_state != other.master_safety_state:
            return False
        if self.master_onoff_state != other.master_onoff_state:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def digital_input_bits(self):
        """Message field 'digital_input_bits'."""
        return self._digital_input_bits

    @digital_input_bits.setter
    def digital_input_bits(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'digital_input_bits' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'digital_input_bits' field must be an unsigned integer in [0, 4294967295]"
        self._digital_input_bits = value

    @builtins.property
    def digital_output_bits(self):
        """Message field 'digital_output_bits'."""
        return self._digital_output_bits

    @digital_output_bits.setter
    def digital_output_bits(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'digital_output_bits' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'digital_output_bits' field must be an unsigned integer in [0, 4294967295]"
        self._digital_output_bits = value

    @builtins.property
    def analog_input_range0(self):
        """Message field 'analog_input_range0'."""
        return self._analog_input_range0

    @analog_input_range0.setter
    def analog_input_range0(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'analog_input_range0' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'analog_input_range0' field must be an integer in [-128, 127]"
        self._analog_input_range0 = value

    @builtins.property
    def analog_input_range1(self):
        """Message field 'analog_input_range1'."""
        return self._analog_input_range1

    @analog_input_range1.setter
    def analog_input_range1(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'analog_input_range1' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'analog_input_range1' field must be an integer in [-128, 127]"
        self._analog_input_range1 = value

    @builtins.property
    def analog_input0(self):
        """Message field 'analog_input0'."""
        return self._analog_input0

    @analog_input0.setter
    def analog_input0(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'analog_input0' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'analog_input0' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._analog_input0 = value

    @builtins.property
    def analog_input1(self):
        """Message field 'analog_input1'."""
        return self._analog_input1

    @analog_input1.setter
    def analog_input1(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'analog_input1' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'analog_input1' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._analog_input1 = value

    @builtins.property
    def analog_output_domain0(self):
        """Message field 'analog_output_domain0'."""
        return self._analog_output_domain0

    @analog_output_domain0.setter
    def analog_output_domain0(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'analog_output_domain0' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'analog_output_domain0' field must be an integer in [-128, 127]"
        self._analog_output_domain0 = value

    @builtins.property
    def analog_output_domain1(self):
        """Message field 'analog_output_domain1'."""
        return self._analog_output_domain1

    @analog_output_domain1.setter
    def analog_output_domain1(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'analog_output_domain1' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'analog_output_domain1' field must be an integer in [-128, 127]"
        self._analog_output_domain1 = value

    @builtins.property
    def analog_output0(self):
        """Message field 'analog_output0'."""
        return self._analog_output0

    @analog_output0.setter
    def analog_output0(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'analog_output0' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'analog_output0' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._analog_output0 = value

    @builtins.property
    def analog_output1(self):
        """Message field 'analog_output1'."""
        return self._analog_output1

    @analog_output1.setter
    def analog_output1(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'analog_output1' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'analog_output1' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._analog_output1 = value

    @builtins.property
    def masterboard_temperature(self):
        """Message field 'masterboard_temperature'."""
        return self._masterboard_temperature

    @masterboard_temperature.setter
    def masterboard_temperature(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'masterboard_temperature' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'masterboard_temperature' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._masterboard_temperature = value

    @builtins.property
    def robot_voltage_48v(self):
        """Message field 'robot_voltage_48v'."""
        return self._robot_voltage_48v

    @robot_voltage_48v.setter
    def robot_voltage_48v(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'robot_voltage_48v' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'robot_voltage_48v' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._robot_voltage_48v = value

    @builtins.property
    def robot_current(self):
        """Message field 'robot_current'."""
        return self._robot_current

    @robot_current.setter
    def robot_current(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'robot_current' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'robot_current' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._robot_current = value

    @builtins.property
    def master_io_current(self):
        """Message field 'master_io_current'."""
        return self._master_io_current

    @master_io_current.setter
    def master_io_current(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'master_io_current' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'master_io_current' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._master_io_current = value

    @builtins.property
    def master_safety_state(self):
        """Message field 'master_safety_state'."""
        return self._master_safety_state

    @master_safety_state.setter
    def master_safety_state(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'master_safety_state' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'master_safety_state' field must be an unsigned integer in [0, 255]"
        self._master_safety_state = value

    @builtins.property
    def master_onoff_state(self):
        """Message field 'master_onoff_state'."""
        return self._master_onoff_state

    @master_onoff_state.setter
    def master_onoff_state(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'master_onoff_state' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'master_onoff_state' field must be an unsigned integer in [0, 255]"
        self._master_onoff_state = value
