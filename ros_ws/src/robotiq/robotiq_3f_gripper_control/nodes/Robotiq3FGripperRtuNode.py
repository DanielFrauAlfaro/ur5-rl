#!/usr/bin/env python3

"""@package docstring
ROS node for controling a Robotiq S-Model gripper using the Modbus RTU protocol.

The script takes as an argument the device address of the gripper (e.g. /dev/ttyUSB0). Afterwards it acts like the
Robotiq3FGripperTcpNode.
"""

import rospy
import robotiq_3f_gripper_control.baseRobotiq3FGripper
import robotiq_modbus_rtu.comModbusRtu
import sys
from robotiq_3f_gripper_articulated_msgs.msg import Robotiq3FGripperRobotInput
from robotiq_3f_gripper_articulated_msgs.msg import Robotiq3FGripperRobotOutput


def mainLoop(address):
    # Gripper is a 3F gripper with a TCP connection
    gripper = (
        robotiq_3f_gripper_control.baseRobotiq3FGripper.robotiqbaseRobotiq3FGripper()
    )
    gripper.client = robotiq_modbus_rtu.comModbusRtu.communication(retry=True)

    # We connect to the address received as an argument
    gripper.client.connectToDevice(address)

    rospy.init_node("robotiq3FGripper")

    # The Gripper status is published on the topic named 'Robotiq3FGripperRobotInput'
    pub = rospy.Publisher(
        "Robotiq3FGripperRobotInput", Robotiq3FGripperRobotInput, queue_size=1
    )

    # The Gripper command is received from the topic named 'Robotiq3FGripperRobotOutput'
    rospy.Subscriber(
        "Robotiq3FGripperRobotOutput",
        Robotiq3FGripperRobotOutput,
        gripper.refreshCommand,
    )

    # We loop
    while not rospy.is_shutdown():
        # Get and publish the Gripper status
        status = gripper.getStatus()
        pub.publish(status)

        # Wait a little
        rospy.sleep(0.05)

        # Send the most recent command
        gripper.sendCommand()

        # Wait a little
        rospy.sleep(0.05)


if __name__ == "__main__":
    try:
        # TODO: Add verification that the argument is an IP address
        mainLoop(sys.argv[1])
    except rospy.ROSInterruptException:
        pass
