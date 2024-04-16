set(_AMENT_PACKAGE_NAME "ur_msgs")
set(ur_msgs_VERSION "2.0.0")
set(ur_msgs_MAINTAINER "G.A. vd. Hoorn <g.a.vanderhoorn@tudelft.nl>, Miguel Prada Sarasola <miguel.prada@tecnalia.com>, Nadia Hammoudeh Garcia <nhg@ipa.fhg.de>")
set(ur_msgs_BUILD_DEPENDS "builtin_interfaces" "geometry_msgs")
set(ur_msgs_BUILDTOOL_DEPENDS "ament_cmake" "rosidl_default_generators")
set(ur_msgs_BUILD_EXPORT_DEPENDS "builtin_interfaces" "geometry_msgs")
set(ur_msgs_BUILDTOOL_EXPORT_DEPENDS )
set(ur_msgs_EXEC_DEPENDS "rosidl_default_runtime" "builtin_interfaces" "geometry_msgs")
set(ur_msgs_TEST_DEPENDS "ament_lint_auto" "ament_lint_common")
set(ur_msgs_GROUP_DEPENDS )
set(ur_msgs_MEMBER_OF_GROUPS "rosidl_interface_packages")
set(ur_msgs_DEPRECATED "")
set(ur_msgs_EXPORT_TAGS)
list(APPEND ur_msgs_EXPORT_TAGS "<rosindex><tags>
        <tag>messages</tag>
        <tag>ros-industrial</tag>
        <tag>services</tag>
        <tag>universal_robot</tag>
      </tags></rosindex>")
list(APPEND ur_msgs_EXPORT_TAGS "<ros_industrial><support_level value=\"community\"/></ros_industrial>")
list(APPEND ur_msgs_EXPORT_TAGS "<build_type>ament_cmake</build_type>")
