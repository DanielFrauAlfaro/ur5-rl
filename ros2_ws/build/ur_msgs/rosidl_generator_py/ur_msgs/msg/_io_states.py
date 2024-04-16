# generated from rosidl_generator_py/resource/_idl.py.em
# with input from ur_msgs:msg/IOStates.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_IOStates(type):
    """Metaclass of message 'IOStates'."""

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
                'ur_msgs.msg.IOStates')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__io_states
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__io_states
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__io_states
            cls._TYPE_SUPPORT = module.type_support_msg__msg__io_states
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__io_states

            from ur_msgs.msg import Analog
            if Analog.__class__._TYPE_SUPPORT is None:
                Analog.__class__.__import_type_support__()

            from ur_msgs.msg import Digital
            if Digital.__class__._TYPE_SUPPORT is None:
                Digital.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class IOStates(metaclass=Metaclass_IOStates):
    """Message class 'IOStates'."""

    __slots__ = [
        '_digital_in_states',
        '_digital_out_states',
        '_flag_states',
        '_analog_in_states',
        '_analog_out_states',
    ]

    _fields_and_field_types = {
        'digital_in_states': 'sequence<ur_msgs/Digital>',
        'digital_out_states': 'sequence<ur_msgs/Digital>',
        'flag_states': 'sequence<ur_msgs/Digital>',
        'analog_in_states': 'sequence<ur_msgs/Analog>',
        'analog_out_states': 'sequence<ur_msgs/Analog>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['ur_msgs', 'msg'], 'Digital')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['ur_msgs', 'msg'], 'Digital')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['ur_msgs', 'msg'], 'Digital')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['ur_msgs', 'msg'], 'Analog')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['ur_msgs', 'msg'], 'Analog')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.digital_in_states = kwargs.get('digital_in_states', [])
        self.digital_out_states = kwargs.get('digital_out_states', [])
        self.flag_states = kwargs.get('flag_states', [])
        self.analog_in_states = kwargs.get('analog_in_states', [])
        self.analog_out_states = kwargs.get('analog_out_states', [])

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
        if self.digital_in_states != other.digital_in_states:
            return False
        if self.digital_out_states != other.digital_out_states:
            return False
        if self.flag_states != other.flag_states:
            return False
        if self.analog_in_states != other.analog_in_states:
            return False
        if self.analog_out_states != other.analog_out_states:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def digital_in_states(self):
        """Message field 'digital_in_states'."""
        return self._digital_in_states

    @digital_in_states.setter
    def digital_in_states(self, value):
        if __debug__:
            from ur_msgs.msg import Digital
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, Digital) for v in value) and
                 True), \
                "The 'digital_in_states' field must be a set or sequence and each value of type 'Digital'"
        self._digital_in_states = value

    @builtins.property
    def digital_out_states(self):
        """Message field 'digital_out_states'."""
        return self._digital_out_states

    @digital_out_states.setter
    def digital_out_states(self, value):
        if __debug__:
            from ur_msgs.msg import Digital
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, Digital) for v in value) and
                 True), \
                "The 'digital_out_states' field must be a set or sequence and each value of type 'Digital'"
        self._digital_out_states = value

    @builtins.property
    def flag_states(self):
        """Message field 'flag_states'."""
        return self._flag_states

    @flag_states.setter
    def flag_states(self, value):
        if __debug__:
            from ur_msgs.msg import Digital
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, Digital) for v in value) and
                 True), \
                "The 'flag_states' field must be a set or sequence and each value of type 'Digital'"
        self._flag_states = value

    @builtins.property
    def analog_in_states(self):
        """Message field 'analog_in_states'."""
        return self._analog_in_states

    @analog_in_states.setter
    def analog_in_states(self, value):
        if __debug__:
            from ur_msgs.msg import Analog
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, Analog) for v in value) and
                 True), \
                "The 'analog_in_states' field must be a set or sequence and each value of type 'Analog'"
        self._analog_in_states = value

    @builtins.property
    def analog_out_states(self):
        """Message field 'analog_out_states'."""
        return self._analog_out_states

    @analog_out_states.setter
    def analog_out_states(self, value):
        if __debug__:
            from ur_msgs.msg import Analog
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, Analog) for v in value) and
                 True), \
                "The 'analog_out_states' field must be a set or sequence and each value of type 'Analog'"
        self._analog_out_states = value
