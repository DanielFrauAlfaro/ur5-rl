#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

bridge = CvBridge()
cv_image = None

def cb(data):
    global bridge, cv_image
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
   


if __name__ == "__main__":
    rospy.init_node("camera2video")

    rospy.Subscriber("/camera1/color/image_raw", Image, cb)

    out = cv2.VideoWriter('filename.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 180, (640, 480)) 

    while not rospy.is_shutdown():
        if cv_image is not None:
            # cv_image = cv2.resize(cv_image, (640, 480), interpolation = cv2.INTER_AREA)
            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            cv2.imshow("Image window", image)
            out.write(image)
        cv2.waitKey(1)