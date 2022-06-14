#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from matplotlib import pyplot as plt
from random import randint
# import scipy
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

                ratio = height/5
                crop = resized[int(height-ratio):height, 0:width]

                blur = cv2.GaussianBlur(crop,(1,1),0)
                # cv2.imwrite('blur.png',blur)
                grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                #cv2.imwrite('gris.png',grey)


                # Dilation #####################
                kernel = np.ones((5, 5), np.uint8)
                dilation = cv2.dilate(grey, kernel, iterations=2)
                # cv2.imshow('dil',dilation)

                # Erosion #####################
                erosion = cv2.erode(dilation, kernel, iterations = 1)
                # cv2.imshow('ero',erosion)
                #cv2.imwrite('erosin.png',erosion)

                # Mask Threshold #####################
                # g1, bw = cv2.threshold(erosion,110,255,cv2.THRESH_BINARY)
                # cv2.imshow('mask',bw)
                # pro2 = bw.sum(axis=0)

                # Sum vertically #####################
                pro = erosion.sum(axis=0) #mean(axis=0)
                y = np.array([])
                count = 0
                for i in pro:
                    y = np.append(y,count)
                    count += 1

                # Gradient ##########################
                # print(pro)
                grad = np.gradient(pro)
                # plt.subplot(2, 2, 1)
                # plt.plot(y, grad)
                grad2 = np.gradient(grad)
                # plt.subplot(2, 2, 2)
                # plt.plot(y, grad2)

                # Threshold gradient #####################
                thPos = np.where(grad > 200, 1, 0)
                thNeg = np.where(grad < -200, 1, 0)

                # Multiply 2nd Derivative * Threshold #####################
                mult = np.multiply(grad2,thPos) # RIGHT edge line
                mult[mult<0] = 0.
                mult2 = np.multiply(grad2,thNeg) # LEFT edge line
                mult2[mult2<0] = 0.
                # plt.subplot(2, 2, 3)
                # plt.plot(y, mult, 'r')
                # plt.plot(y, mult2, 'g')

                # Shift graph 1 pixel #####################
                shft = np.roll(mult, 1)
                shft2 = np.roll(mult2, 1)
                cmp1 = shft - mult
                cmp2 = shft2 - mult2
                # plt.subplot(2, 2, 4)
                # plt.plot(y, mult,'o')
                # plt.plot(y, mult2, 'o')
                # plt.show()
                # plt.savefig('dataIMGs.png')

                # Find lowest value #####################
                x = np.argmin(pro)
                # print('min val index' , x)

                # Find Left & Right edges ###############
                left = np.argmax(mult)
                right = np.argmax(mult2)
                center = (left+right)/2
                # print('line center' , center)
                myY = int(height-10)



                desiredX = (int(resized.shape[1])/2)
                point = cv2.circle(resized, (int(x),myY),5, (0,0,255))
                point2 = cv2.circle(resized, (int(desiredX),myY),5, (0,255,255))
                error = desiredX-x
                # print('error', error)

                cv2.imwrite('outputPath.png',point2)
                cv2.imwrite('crop.png',crop)

                # Publish point
                self.pub.publish(error)
                print(error)

                # cv2.imwrite('myPath.png',point)
                # cv2.imwrite('edges.png',self.image)

            # self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("PuzzleBotCamera", anonymous=True)
    my_node = LineFollower()
    my_node.start()
