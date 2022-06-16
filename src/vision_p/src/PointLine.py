#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from matplotlib import pyplot as plt
from random import randint
#from tensorflow import keras
#import scipy
# from scipy import signal
# import scipy.signal
import cv2
import os
import numpy as np

class LineFollower(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(10000)

        # Publishers
        # self.pub = rospy.Publisher('PuzzleVideo', Image,queue_size=10)
        self.pub = rospy.Publisher('LineFollower', Int32, queue_size=10)

        # Subscribers
        rospy.Subscriber("/video_source/raw",Image,self.callback)

    def callback(self, data):
        # rospy.loginfo('Image received')
        self.image = self.br.imgmsg_to_cv2(data, "bgr8")

        #cv2.imwrite('Original.png',new_image)

    def start(self):
        # rospy.loginfo("Timing images")

        while not rospy.is_shutdown():
            # rospy.loginfo('Publishing image')
            cont = 0

            if self.image is not None:
                # cv2.imwrite('testIMG.png',self.image)

                # Rezise image and crop (region of interest ROI) #####################
                scale_percent = 30 # percent of original size
                width = int(self.image.shape[1] * scale_percent / 100)
                height = int(self.image.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized = cv2.resize(self.image, dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite('Original.png',resized)
                ratio = height/5
                crop = resized[int(height-ratio):height, 0:width]

                img_blur = cv2.GaussianBlur(crop, (1,1), 0)
                grey = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
                proImg = grey.mean(axis=0)
                meanImg = np.mean(proImg)
                ThreshImg = meanImg - 40

                # define kernel
                kernel = np.ones((5, 5), np.uint8)
                #Modify the black values depending on the enviroment ###############################
                #95 on the lab
                g1, bw = cv2.threshold(grey,ThreshImg,255,cv2.THRESH_BINARY)
                # Erosion
                erosion = cv2.erode(bw, kernel, iterations = 1)

                #cv2.imshow('ero',erosion)
                # Dilation
                dilation = cv2.dilate(erosion, kernel, iterations=1)
                cv2.imwrite('dil.png',dilation)
                pro = dilation.mean(axis=0)
                LineBool = np.mean(pro)

                print("Pro",LineBool, ThreshImg)


                proHat = pro
                #proHat = savgol_filter(pro, 51, 3)
                #print(pro)
                xP = np.argsort(proHat)[:25]
                x = xP.mean(axis=0)

                #print(pro)
                desiredX = (int(resized.shape[1])/2)
                #x = np.argsort(proHat)[:3]
                #print(x)
                myY = int(height-10)
                point = cv2.circle(resized, (int(x),myY),5, (0,0,255))
                point2 = cv2.circle(resized, (int(desiredX),myY),5, (0,255,255))
                error = desiredX-x

                # if error>150 and LineBool<=250:
                #     error = 150
                # elif error<-150 and LineBool<=250:
                #     error = -150

                if LineBool>250 :
                    error = 500
                    print("Error",error)

                elif LineBool<205 :
                    error = 0
                    print("Out",error)
                # Publish point

                self.pub.publish(error)
                print(error)


                cv2.imwrite('myPath.png',point2)

                # cv2.imwrite('myPath.png',point)
                # cv2.imwrite('edges.png',self.image)

            # self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("PuzzleBotCamera", anonymous=True)
    my_node = LineFollower()
    my_node.start()
