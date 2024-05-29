#! /usr/bin/python3

import rospy
from stable_baselines3 import SAC
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import time



frame_h = 160
frame_w = 160

bridge = CvBridge()
images = {}
x = {}  

images["frontal"] = [np.ones((frame_w, frame_h), dtype=np.int16), 
                     np.ones((frame_w, frame_h), dtype=np.int16)]

images["side"] = [np.ones((frame_w, frame_h), dtype=np.int16), 
                  np.ones((frame_w, frame_h), dtype=np.int16)]

images["robot"] = [np.ones((frame_w, frame_h), dtype=np.int16), 
                   np.ones((frame_w, frame_h), dtype=np.int16)]





def ee_cb(data):
    x["ee_position"] = data.data

def image_cb(data):
    global bridge, images, x, frame_w, frame_h

    if "color" in data.header.frame_id:
        if "camera" in data.header.frame_id:
            images["robot"][0] = bridge.imgmsg_to_cv2(data)

            # Convert color image to grayscale
            images["robot"][0] = cv2.cvtColor(images["robot"][0], cv2.COLOR_BGR2GRAY)

            # Resize grayscale image to 160x160
            images["robot"][0] = cv2.resize(images["robot"][0], (frame_h, frame_w))
            images["robot"][0] = (images["robot"][0] - np.min(images["robot"][0])) / float(max(1.0, (np.max(images["robot"][0]) - np.min(images["robot"][0]))))


        elif "frontal" in data.header.frame_id:
            images["frontal"][0] = bridge.imgmsg_to_cv2(data)

            # Convert color image to grayscale
            images["frontal"][0] = cv2.cvtColor(images["frontal"][0], cv2.COLOR_BGR2GRAY)

            # Resize grayscale image to 160x160
            images["frontal"][0] = cv2.resize(images["frontal"][0], (frame_h, frame_w))
            images["frontal"][0] = (images["frontal"][0] - np.min(images["frontal"][0])) / float(max(1.0, (np.max(images["frontal"][0]) - np.min(images["frontal"][0])))) 

        else:
            images["side"][0] = bridge.imgmsg_to_cv2(data)

            # Convert color image to grayscale
            images["side"][0] = cv2.cvtColor(images["side"][0], cv2.COLOR_BGR2GRAY)

            # Resize grayscale image to 160x160
            images["side"][0] = cv2.resize(images["side"][0], (frame_h, frame_w))
            images["side"][0] = (images["side"][0] - np.min(images["side"][0])) / float(max(1.0, (np.max(images["side"][0]) - np.min(images["side"][0]))))


    elif "depth" in data.header.frame_id:
        if "camera" in data.header.frame_id:
            images["robot"][1] = bridge.imgmsg_to_cv2(data)
            
            
            images["robot"][1] = cv2.resize(images["robot"][1], (frame_h, frame_w))
            images["robot"][1] = (images["robot"][1] - np.min(images["robot"][1])) / float(max(1.0, (np.max(images["robot"][1]) - np.min(images["robot"][1]))))
            images["robot"][1] = (images["robot"][1] * 255).astype(np.uint8)
            
            images["robot"][1] = (images["robot"][1] - np.min(images["robot"][1])) / float(max(1.0, (np.max(images["robot"][1]) - np.min(images["robot"][1]))))  


        elif "frontal" in data.header.frame_id:
            images["frontal"][1] = bridge.imgmsg_to_cv2(data)
            
            # Resize grayscale image to 160x160
            images["frontal"][1] = cv2.resize(images["frontal"][1], (frame_h, frame_w))
            images["frontal"][1] = (images["frontal"][1] - np.min(images["frontal"][1])) / float(max(1.0,(np.max(images["frontal"][1]) - np.min(images["frontal"][1]))))
            images["frontal"][1] = (images["frontal"][1] * 255).astype(np.uint8)
            
            images["frontal"][1] = (images["frontal"][1] - np.min(images["frontal"][1])) / float(max(1.0, (np.max(images["frontal"][1]) - np.min(images["frontal"][1])))) 
    
        else:
            images["side"][1] = bridge.imgmsg_to_cv2(data)

            # Resize grayscale image to 160x160
            images["side"][1] = cv2.resize(images["side"][1], (frame_h, frame_w))
            images["side"][1] = (images["side"][1] - np.min(images["side"][1])) / float(max(1.0,(np.max(images["side"][1]) - np.min(images["side"][1]))))
            images["side"][1] = (images["side"][1] * 255).astype(np.uint8)
            
            images["side"][1] = (images["side"][1] - np.min(images["side"][1])) / float(max(1.0, (np.max(images["side"][1]) - np.min(images["side"][1]))))
        


def rl_controller(pubs, model_path):
    global x

    # Model loading
    model = SAC.load(model_path)      
    model.policy.eval()

    rate = rospy.Rate(10)

    command_msg = Float32MultiArray()

    while not rospy.is_shutdown():
        try:
            
            # print(np.unique(images["side"][1]))
            merged = cv2.merge([images["frontal"][0], images["frontal"][1],
                            images["side"][0], images["side"][1],
                            images["robot"][0], images["robot"][1]])
            merged = np.transpose(merged, (2,0,1))
            x["image"] = merged
        except:
            pass
        
        # print(np.unique(x["image"][0][0]))
        # print(np.unique(x["image"][0][1]))
        # print(np.unique(x["image"][1][0]))
        # print(np.unique(x["image"][1][1]))
        # print(np.unique(x["image"][2][0]))
        # print(np.unique(x["image"][2][1]))
        # print("---n\n")
        action, __ = model.predict(x, deterministic = True)
        command_msg.data = action.tolist()
        pubs["commands"].publish(command_msg)
        

        rate.sleep()






if __name__ == "__main__":
    try:
        rospy.init_node("rl_controller")

        model_path = rospy.get_param("model_path")
        k_to_controller_topic = rospy.get_param("k_to_controller_topic")
        controller_to_k_topic = rospy.get_param("controller_to_k_topic")
        
        # -- Publishers --
        pubs = {}
        pubs["commands"] = rospy.Publisher(controller_to_k_topic, Float32MultiArray, queue_size=10)
        
        # -- Subscribers
        rospy.Subscriber(k_to_controller_topic, Float32MultiArray, ee_cb)

        rospy.Subscriber("/camera_robot/color/image_raw", Image, image_cb)
        rospy.Subscriber("/camera_frontal/color/image_raw", Image, image_cb)
        rospy.Subscriber("/camera_side/color/image_raw", Image, image_cb)
        rospy.Subscriber("/camera_robot/depth/image_raw", Image, image_cb)
        rospy.Subscriber("/camera_frontal/depth/image_raw", Image, image_cb)
        rospy.Subscriber("/camera_side/depth/image_raw", Image, image_cb)

        time.sleep(5)

        rl_controller(pubs, model_path)


    except rospy.ROSInterruptException:
        pass

