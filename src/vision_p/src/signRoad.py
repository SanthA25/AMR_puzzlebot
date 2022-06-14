#!/usr/bin/env python3
# import the necessary packages
# import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from matplotlib import pyplot as plt
from random import randint

class SignDetect(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(10000)

        # Publishers
        self.pub = rospy.Publisher('SignDetect', String, queue_size=10)

        # Load model
        self.model = load_model("/home/puzzlebot/reto_ws/src/vision_p/src/myCNN_model") #/home/puzzlebot/reto_ws/src/vision_p/src
        self.threshold = 0.75         # PROBABLITY THRESHOLD

        # Subscribers
        rospy.Subscriber("/video_source/raw",Image,self.callback)

    def callback(self, data):
        self.image = self.br.imgmsg_to_cv2(data, "bgr8")

    # Image preprocessing functions ###############3
    def grayscale(self, image):
        self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        return self.image

    def equalize(image):
        self.image = cv2.equalizeHist(self.image)
        return self.image

    def preprocessing(self, image):
        self.image = grayscale(self.image)
        self.image = equalize(self.image)
        self.image = self.image/255
        return self.image

    def getCalssName(classNo):
        if   classNo == 14: return 'Stop'
        elif classNo == 32: return 'End of all speed and passing limits'
        elif classNo == 33: return 'Turn right ahead'
        elif classNo == 35: return 'Ahead only'
        else: pass

    def DetectSignal(self):
        if self.image is not None:
            # read Image ###############3
            img = self.image

            # Convert to graycsale ###############3
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Blur the image for better edge detection ###############3
            img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

            # Cany Edge Detection method ###############3
            th1 = 100  # lower threshold 50
            th2 = 200  # upper threshold 100
            edges = cv2.Canny(img_blur, th1, th2)

            # Find contours ###############3
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_blur, contours, -1, (0,255,0), 3)
            # print(type(contours))

            # find the biggest countour (c) by the area ###############3
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)

            # crop image at given contour ###############3
            cropped_contour= img[y-10:y+h+10, x-10:x+w+10]
            # print(contours.index(c))

            # PROCESS IMAGE
            img2 = np.asarray(cropped_contour)
            img2 = cv2.resize(img2, (32, 32))
            img2 = preprocessing(img2)
            cv2.imshow("Processed Image", img2)
            img2 = img2.reshape(1, 32, 32, 1)

            # PREDICT IMAGE
            predictions = self.model.predict(img2)
            classIndex = self.model.predict_classes(img2)
            probabilityValue =np.amax(predictions)
            if probabilityValue > self.threshold:
                print(classIndex,getCalssName(classIndex))
                self.pub.publish(getCalssName(classIndex))


if __name__ == '__main__':
    print("Starting...")
    rospy.init_node("Signal", anonymous=True)
    my_node =  SignDetect()
    my_node.DetectSignal()
