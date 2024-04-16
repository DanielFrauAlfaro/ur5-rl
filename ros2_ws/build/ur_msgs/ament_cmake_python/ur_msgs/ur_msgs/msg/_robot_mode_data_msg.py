# generated from rosidl_generator_py/resource/_idl.py.em
# with input from ur_msgs:msg/RobotModeDataMsg.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RobotModeDataMsg(type):
    """Metaclass of message 'RobotModeDataMsg'."""

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
                'ur_msgs.msg.RobotModeDataMsg')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__robot_mode_data_msg
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__robot_mode_data_msg
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__robot_mode_data_msg
            cls._TYPE_SUPPORT = module.type_support_msg__msg__robot_mode_data_msg
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__robot_mode_data_msg

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class RobotModeDataMsg(metaclass=Metaclass_RobotModeDataMsg):
    """Message class 'RobotModeDataMsg'."""

    __slots__ = [
        '_timestamp',
        '_is_robot_connected',
        '_is_real_robot_enabled',
        '_is_power_on_robot',
        '_is_emergency_stopped',
        '_is_protective_stopped',
        '_is_program_running',
        '_is_program_paused',
    ]

    _fields_and_field_types = {
        'timestamp': 'uint64',
        'is_robot_connected': 'boolean',
        'is_real_robot_enabled': 'boolean',
        'is_power_on_robot': 'boolean',
        'is_emergency_stopped': 'boolean',
        'is_protective_stopped': 'boolean',
        'is_program_running': 'boolean',
        'is_program_paused': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint64'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.timestamp = kwargs.get('timestamp', int())
        self.is_robot_connected = kwargs.get('is_robot_connected', bool())
        self.is_real_robot_enabled = kwargs.get('is_real_robot_enabled', bool())
        self.is_power_on_robot = kwargs.get('is_power_on_robot', bool())
        self.is_emergency_stopped = kwargs.get('is_emergency_stopped', bool())
        self.is_protective_stopped = kwargs.get('is_protective_stopped', bool())
        self.is_program_running = kwargs.get('is_program_running', bool())
        self.is_program_paused = kwargs.get('is_program_paused', bool())

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
        if self.timestamp != other.timestamp:
            return False
        if self.is_robot_connected != other.is_robot_connected:
            return False
        if self.is_real_robot_enabled != other.is_real_robot_enabled:
            return False
        if self.is_power_on_robot != other.is_power_on_robot:
            return False
        if self.is_emergency_stopped != other.is_emergency_stopped:
            return False
        if self.is_protective_stopped != other.is_protective_stopped:
            return False
        if self.is_program_running != other.is_program_running:
            return False
        if self.is_program_paused != other.is_program_paused:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def timestamp(self):
        """Message field 'timestamp'."""
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'timestamp' field must be of type 'int'"
            assert value >= 0 and value < 18446744073709551616, \
                "The 'timestamp' field must be an unsigned integer in [0, 18446744073709551615]"
        self._timestamp = value

    @builtins.property
    def is_robot_connected(self):
        """Message field 'is_robot_connected'."""
        return self._is_robot_connected

    @is_robot_connected.setter
    def is_robot_connected(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_robot_connected' field must be of type 'bool'"
        self._is_robot_connected = value

    @builtins.property
    def is_real_robot_enabled(self):
        """Message field 'is_real_robot_enabled'."""
        return self._is_real_robot_enabled

    @is_real_robot_enabled.setter
    def is_real_robot_enabled(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_real_robot_enabled' field must be of type 'bool'"
        self._is_real_robot_enabled = value

    @builtins.property
    def is_power_on_robot(self):
        """Message field 'is_power_on_robot'."""
        return self._is_power_on_robot

    @is_power_on_robot.setter
    def is_power_on_robot(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_power_on_robot' field must be of type 'bool'"
        self._is_power_on_robot = value

    @builtins.property
    def is_emergency_stopped(self):
        """Message field 'is_emergency_stopped'."""
        return self._is_emergency_stopped

    @is_emergency_stopped.setter
    def is_emergency_stopped(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_emergency_stopped' field must be of type 'bool'"
        self._is_emergency_stopped = value

    @builtins.property
    def is_protective_stopped(self):
        """Message field 'is_protective_stopped'."""
        return self._is_protective_stopped

    @is_protective_stopped.setter
    def is_protective_stopped(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_protective_stopped' field must be of type 'bool'"
        self._is_protective_stopped = value

    @builtins.property
    def is_program_running(self):
        """Message field 'is_program_running'."""
        return self._is_program_running

    @is_program_running.setter
    def is_program_running(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_program_running' field must be of type 'bool'"
        self._is_program_running = value

    @builtins.property
    def is_program_paused(self):
        """Message field 'is_program_paused'."""
        return self._is_program_paused

    @is_program_paused.setter
    def is_program_paused(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_program_paused' field must be of type 'bool'"
        self._is_program_paused = value
