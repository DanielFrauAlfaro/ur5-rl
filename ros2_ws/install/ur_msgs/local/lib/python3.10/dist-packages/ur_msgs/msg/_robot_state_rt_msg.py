# generated from rosidl_generator_py/resource/_idl.py.em
# with input from ur_msgs:msg/RobotStateRTMsg.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'q_target'
# Member 'qd_target'
# Member 'qdd_target'
# Member 'i_target'
# Member 'm_target'
# Member 'q_actual'
# Member 'qd_actual'
# Member 'i_actual'
# Member 'tool_acc_values'
# Member 'tcp_force'
# Member 'tool_vector'
# Member 'tcp_speed'
# Member 'motor_temperatures'
# Member 'joint_modes'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RobotStateRTMsg(type):
    """Metaclass of message 'RobotStateRTMsg'."""

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
                'ur_msgs.msg.RobotStateRTMsg')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__robot_state_rt_msg
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__robot_state_rt_msg
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__robot_state_rt_msg
            cls._TYPE_SUPPORT = module.type_support_msg__msg__robot_state_rt_msg
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__robot_state_rt_msg

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class RobotStateRTMsg(metaclass=Metaclass_RobotStateRTMsg):
    """Message class 'RobotStateRTMsg'."""

    __slots__ = [
        '_time',
        '_q_target',
        '_qd_target',
        '_qdd_target',
        '_i_target',
        '_m_target',
        '_q_actual',
        '_qd_actual',
        '_i_actual',
        '_tool_acc_values',
        '_tcp_force',
        '_tool_vector',
        '_tcp_speed',
        '_digital_input_bits',
        '_motor_temperatures',
        '_controller_timer',
        '_test_value',
        '_robot_mode',
        '_joint_modes',
    ]

    _fields_and_field_types = {
        'time': 'double',
        'q_target': 'sequence<double>',
        'qd_target': 'sequence<double>',
        'qdd_target': 'sequence<double>',
        'i_target': 'sequence<double>',
        'm_target': 'sequence<double>',
        'q_actual': 'sequence<double>',
        'qd_actual': 'sequence<double>',
        'i_actual': 'sequence<double>',
        'tool_acc_values': 'sequence<double>',
        'tcp_force': 'sequence<double>',
        'tool_vector': 'sequence<double>',
        'tcp_speed': 'sequence<double>',
        'digital_input_bits': 'double',
        'motor_temperatures': 'sequence<double>',
        'controller_timer': 'double',
        'test_value': 'double',
        'robot_mode': 'double',
        'joint_modes': 'sequence<double>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.time = kwargs.get('time', float())
        self.q_target = array.array('d', kwargs.get('q_target', []))
        self.qd_target = array.array('d', kwargs.get('qd_target', []))
        self.qdd_target = array.array('d', kwargs.get('qdd_target', []))
        self.i_target = array.array('d', kwargs.get('i_target', []))
        self.m_target = array.array('d', kwargs.get('m_target', []))
        self.q_actual = array.array('d', kwargs.get('q_actual', []))
        self.qd_actual = array.array('d', kwargs.get('qd_actual', []))
        self.i_actual = array.array('d', kwargs.get('i_actual', []))
        self.tool_acc_values = array.array('d', kwargs.get('tool_acc_values', []))
        self.tcp_force = array.array('d', kwargs.get('tcp_force', []))
        self.tool_vector = array.array('d', kwargs.get('tool_vector', []))
        self.tcp_speed = array.array('d', kwargs.get('tcp_speed', []))
        self.digital_input_bits = kwargs.get('digital_input_bits', float())
        self.motor_temperatures = array.array('d', kwargs.get('motor_temperatures', []))
        self.controller_timer = kwargs.get('controller_timer', float())
        self.test_value = kwargs.get('test_value', float())
        self.robot_mode = kwargs.get('robot_mode', float())
        self.joint_modes = array.array('d', kwargs.get('joint_modes', []))

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
        if self.time != other.time:
            return False
        if self.q_target != other.q_target:
            return False
        if self.qd_target != other.qd_target:
            return False
        if self.qdd_target != other.qdd_target:
            return False
        if self.i_target != other.i_target:
            return False
        if self.m_target != other.m_target:
            return False
        if self.q_actual != other.q_actual:
            return False
        if self.qd_actual != other.qd_actual:
            return False
        if self.i_actual != other.i_actual:
            return False
        if self.tool_acc_values != other.tool_acc_values:
            return False
        if self.tcp_force != other.tcp_force:
            return False
        if self.tool_vector != other.tool_vector:
            return False
        if self.tcp_speed != other.tcp_speed:
            return False
        if self.digital_input_bits != other.digital_input_bits:
            return False
        if self.motor_temperatures != other.motor_temperatures:
            return False
        if self.controller_timer != other.controller_timer:
            return False
        if self.test_value != other.test_value:
            return False
        if self.robot_mode != other.robot_mode:
            return False
        if self.joint_modes != other.joint_modes:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def time(self):
        """Message field 'time'."""
        return self._time

    @time.setter
    def time(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'time' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'time' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._time = value

    @builtins.property
    def q_target(self):
        """Message field 'q_target'."""
        return self._q_target

    @q_target.setter
    def q_target(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'q_target' array.array() must have the type code of 'd'"
            self._q_target = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'q_target' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._q_target = array.array('d', value)

    @builtins.property
    def qd_target(self):
        """Message field 'qd_target'."""
        return self._qd_target

    @qd_target.setter
    def qd_target(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'qd_target' array.array() must have the type code of 'd'"
            self._qd_target = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'qd_target' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._qd_target = array.array('d', value)

    @builtins.property
    def qdd_target(self):
        """Message field 'qdd_target'."""
        return self._qdd_target

    @qdd_target.setter
    def qdd_target(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'qdd_target' array.array() must have the type code of 'd'"
            self._qdd_target = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'qdd_target' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._qdd_target = array.array('d', value)

    @builtins.property
    def i_target(self):
        """Message field 'i_target'."""
        return self._i_target

    @i_target.setter
    def i_target(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'i_target' array.array() must have the type code of 'd'"
            self._i_target = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'i_target' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._i_target = array.array('d', value)

    @builtins.property
    def m_target(self):
        """Message field 'm_target'."""
        return self._m_target

    @m_target.setter
    def m_target(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'm_target' array.array() must have the type code of 'd'"
            self._m_target = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'm_target' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._m_target = array.array('d', value)

    @builtins.property
    def q_actual(self):
        """Message field 'q_actual'."""
        return self._q_actual

    @q_actual.setter
    def q_actual(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'q_actual' array.array() must have the type code of 'd'"
            self._q_actual = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'q_actual' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._q_actual = array.array('d', value)

    @builtins.property
    def qd_actual(self):
        """Message field 'qd_actual'."""
        return self._qd_actual

    @qd_actual.setter
    def qd_actual(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'qd_actual' array.array() must have the type code of 'd'"
            self._qd_actual = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'qd_actual' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._qd_actual = array.array('d', value)

    @builtins.property
    def i_actual(self):
        """Message field 'i_actual'."""
        return self._i_actual

    @i_actual.setter
    def i_actual(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'i_actual' array.array() must have the type code of 'd'"
            self._i_actual = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'i_actual' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._i_actual = array.array('d', value)

    @builtins.property
    def tool_acc_values(self):
        """Message field 'tool_acc_values'."""
        return self._tool_acc_values

    @tool_acc_values.setter
    def tool_acc_values(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'tool_acc_values' array.array() must have the type code of 'd'"
            self._tool_acc_values = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'tool_acc_values' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._tool_acc_values = array.array('d', value)

    @builtins.property
    def tcp_force(self):
        """Message field 'tcp_force'."""
        return self._tcp_force

    @tcp_force.setter
    def tcp_force(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'tcp_force' array.array() must have the type code of 'd'"
            self._tcp_force = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'tcp_force' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._tcp_force = array.array('d', value)

    @builtins.property
    def tool_vector(self):
        """Message field 'tool_vector'."""
        return self._tool_vector

    @tool_vector.setter
    def tool_vector(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'tool_vector' array.array() must have the type code of 'd'"
            self._tool_vector = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'tool_vector' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._tool_vector = array.array('d', value)

    @builtins.property
    def tcp_speed(self):
        """Message field 'tcp_speed'."""
        return self._tcp_speed

    @tcp_speed.setter
    def tcp_speed(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'tcp_speed' array.array() must have the type code of 'd'"
            self._tcp_speed = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'tcp_speed' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._tcp_speed = array.array('d', value)

    @builtins.property
    def digital_input_bits(self):
        """Message field 'digital_input_bits'."""
        return self._digital_input_bits

    @digital_input_bits.setter
    def digital_input_bits(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'digital_input_bits' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'digital_input_bits' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._digital_input_bits = value

    @builtins.property
    def motor_temperatures(self):
        """Message field 'motor_temperatures'."""
        return self._motor_temperatures

    @motor_temperatures.setter
    def motor_temperatures(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'motor_temperatures' array.array() must have the type code of 'd'"
            self._motor_temperatures = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'motor_temperatures' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._motor_temperatures = array.array('d', value)

    @builtins.property
    def controller_timer(self):
        """Message field 'controller_timer'."""
        return self._controller_timer

    @controller_timer.setter
    def controller_timer(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'controller_timer' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'controller_timer' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._controller_timer = value

    @builtins.property
    def test_value(self):
        """Message field 'test_value'."""
        return self._test_value

    @test_value.setter
    def test_value(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'test_value' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'test_value' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._test_value = value

    @builtins.property
    def robot_mode(self):
        """Message field 'robot_mode'."""
        return self._robot_mode

    @robot_mode.setter
    def robot_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'robot_mode' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'robot_mode' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._robot_mode = value

    @builtins.property
    def joint_modes(self):
        """Message field 'joint_modes'."""
        return self._joint_modes

    @joint_modes.setter
    def joint_modes(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'joint_modes' array.array() must have the type code of 'd'"
            self._joint_modes = value
            return
        if __debug__:
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
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'joint_modes' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joint_modes = array.array('d', value)
