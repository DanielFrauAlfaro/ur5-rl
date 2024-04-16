#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/daniel/Desktop/ur5-rl/ros_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/daniel/Desktop/ur5-rl/ros_ws/install/lib/python3/dist-packages:/daniel/Desktop/ur5-rl/ros_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/daniel/Desktop/ur5-rl/ros_ws/build" \
    "/usr/bin/python3" \
    "/daniel/Desktop/ur5-rl/ros_ws/src/canbus_interface/setup.py" \
    egg_info --egg-base /daniel/Desktop/ur5-rl/ros_ws/build/canbus_interface \
    build --build-base "/daniel/Desktop/ur5-rl/ros_ws/build/canbus_interface" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/daniel/Desktop/ur5-rl/ros_ws/install" --install-scripts="/daniel/Desktop/ur5-rl/ros_ws/install/bin"
