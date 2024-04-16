# generated from rosidl_generator_py/resource/_idl.py.em
# with input from ur_msgs:msg/Analog.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Analog(type):
    """Metaclass of message 'Analog'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'CURRENT': 0,
        'VOLTAGE': 1,
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
                'ur_msgs.msg.Analog')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__analog
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__analog
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__analog
            cls._TYPE_SUPPORT = module.type_support_msg__msg__analog
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__analog

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'CURRENT': cls.__constants['CURRENT'],
            'VOLTAGE': cls.__constants['VOLTAGE'],
        }

    @property
    def CURRENT(self):
        """Message constant 'CURRENT'."""
        return Metaclass_Analog.__constants['CURRENT']

    @property
    def VOLTAGE(self):
        """Message constant 'VOLTAGE'."""
        return Metaclass_Analog.__constants['VOLTAGE']


class Analog(metaclass=Metaclass_Analog):
    """
    Message class 'Analog'.

    Constants:
      CURRENT
      VOLTAGE
    """

    __slots__ = [
        '_pin',
        '_domain',
        '_state',
    ]

    _fields_and_field_types = {
        'pin': 'uint8',
        'domain': 'uint8',
        'state': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.pin = kwargs.get('pin', int())
        self.domain = kwargs.get('domain', int())
        self.state = kwargs.get('state', float())

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
        if self.pin != other.pin:
            return False
        if self.domain != other.domain:
            return False
        if self.state != other.state:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def pin(self):
        """Message field 'pin'."""
        return self._pin

    @pin.setter
    def pin(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'pin' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'pin' field must be an unsigned integer in [0, 255]"
        self._pin = value

    @builtins.property
    def domain(self):
        """Message field 'domain'."""
        return self._domain

    @domain.setter
    def domain(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'domain' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'domain' field must be an unsigned integer in [0, 255]"
        self._domain = value

    @builtins.property
    def state(self):
        """Message field 'state'."""
        return self._state

    @state.setter
    def state(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'state' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'state' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._state = value
