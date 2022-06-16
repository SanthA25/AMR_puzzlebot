#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

class Nodo(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(10)

        # Publishers
        # self.pub = rospy.Publisher('PuzzleVideo', Image,queue_size=10)
        self.pub = rospy.Publisher('Semaphore', String, queue_size=10)

        # Subscribers
        rospy.Subscriber("/video_source/raw",Image,self.callback)

    def callback(self, data):
        # rospy.loginfo('Image received')
        self.image = self.br.imgmsg_to_cv2(data, "bgr8")

    def start(self):
        # rospy.loginfo("Timing images")

        while not rospy.is_shutdown():
            # rospy.loginfo('Publishing image')
            cont = 0

            if self.image is not None:
                # self.pub.publish(self.br.cv2_to_imgmsg(self.image,"bgr8"))
                image_res = cv2.resize(self.image, (0,0), fx=0.5, fy=0.5)

                # blur to revome some noise
                blur = cv2.GaussianBlur(image_res,(5,5),0)
                # Image color to HSV
                hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
                #------------------Blob detector Creation------------------
                params = cv2.SimpleBlobDetector_Params()
                params.blobColor = 0
                params.filterByColor = False
                params.filterByArea = True
                params.minArea = 150
                params.maxArea = 600
                params.minThreshold = 0;
                params.maxThreshold = 255;
                ver = (cv2.__version__).split('.')
                if int(ver[0]) < 3:
                    detector = cv2.SimpleBlobDetector(params)
                else:
                    detector = cv2.SimpleBlobDetector_create(params)

                #----------------------Color segmentation------------------
                #Green mask
                min_g = np.array([40, 50, 50])
                max_g = np.array([80, 255, 255])
                msk_0 = cv2.inRange(hsv, min_g, max_g)

                min_g = np.array([50, 100, 100])
                max_g = np.array([70, 255, 255])
                msk_1 = cv2.inRange(hsv, min_g, max_g)

                # join my masks for Green
                msk_green = msk_0 + msk_1

                # Morphological operations ##########################################
                kernel = np.ones((4, 4), np.uint8)
                #erosion
                erosionG = cv2.erode(msk_0, kernel, iterations = 1)
                # cv2.imwrite('erosion.png',erosion)


                keypoints_G = detector.detect(erosionG)
                im_with_keypoints_G = cv2.drawKeypoints(erosionG, keypoints_G, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                # # if there are any white pixels on mask, sum will be > 0
                hasGreen = np.sum(erosionG)
                # if hasGreen > 0:
                #     colorStr = "green"
                #     self.pub.publish(colorStr)
                #     print('Green detected!')

                # Red mask
                lower_red = np.array([160,90,90])
                upper_red = np.array([180,255,255])
                mask0 = cv2.inRange(hsv, lower_red, upper_red)
                #                      h  s   v
                lower_red = np.array([0,110,110])
                upper_red = np.array([15,255,255])
                mask1 = cv2.inRange(hsv, lower_red, upper_red)

                lower_red = np.array([15,100,100])
                upper_red = np.array([30,255,255])
                mask2 = cv2.inRange(hsv, lower_red, upper_red)
                # join my masks for Red
                msk_red = mask0+mask1+mask2

                erosionR = cv2.erode(mask0, kernel, iterations = 1)
                # cv2.imwrite('erosion.png',erosionR)

                #Blue mask
                #                      h  s   v
                lower_blue = np.array([85,80,80])
                upper_blue = np.array([120,255,255])
                mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
                # join my masks for blue
                msk_blue = mask1

                erosionB = cv2.erode(msk_blue, kernel, iterations = 1)
                #dilateR = cv2.dilate(msk_blue, kernel, iterations = 1)
                # cv2.imwrite('erosion.png',erosionR)
                #cv2.imwrite('Blur1.png',dilateR)

                # if there are any white pixels on mask, sum will be > 0
                hasBlue = np.sum(erosionB)

                keypoints_R = detector.detect(erosionR)
                im_with_keypoints_R = cv2.drawKeypoints(erosionR, keypoints_R, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                rng = np.concatenate((im_with_keypoints_R,im_with_keypoints_G), axis = 1)
                cv2.imwrite('Blobs.png',rng)

                # if there are any white pixels on mask, sum will be > 0
                hasRed = np.sum(erosionR)
                # if hasRed > 0:
                #     colorStr = "red"
                #     self.pub.publish(colorStr)
                #     print('Red detected!')

                if hasGreen<10000 and hasRed<10000:
                    colorStr = "None"
                    self.pub.publish(colorStr)

                elif hasRed > hasGreen:
                    colorStr = "red"
                    self.pub.publish(colorStr)



                elif hasGreen >= hasRed:
                    colorStr = "green"
                    self.pub.publish(colorStr)


                if hasBlue > 15000 and hasGreen > hasRed:
                    colorStr = "blue"
                    self.pub.publish(colorStr)

                elif hasRed > hasGreen:
                    colorStr = "red"
                    self.pub.publish(colorStr)

                else:
                    colorStr = "None"
                    self.pub.publish(colorStr)

                res_red = cv2.bitwise_and(image_res,image_res, mask=erosionR)
                res_green = cv2.bitwise_and(image_res,image_res, mask=erosionG)
                color_dec = np.concatenate((res_red,res_green), axis = 1)
                # cv2.imshow('Red & Green Detection', color_dec)

                cv2.imwrite('Semaphores.png',color_dec)

                print(colorStr)
                print("HasRed =", hasRed)
                print("hasGreen =", hasGreen)
                print("hasBlue =",hasBlue)

                cont = cont+1

            self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("PuzzleBotCamera", anonymous=True)
    my_node = Nodo()
    my_node.start()











    #----------THRESHOLDING--------------------
    #Images form color to grayscale
    # gray_red = cv2.cvtColor(res_red,cv2.COLOR_BGR2GRAY)
    # gray_green = cv2.cvtColor(res_green,cv2.COLOR_BGR2GRAY)
    #
    # th_red, bin_red = cv2.threshold(gray_red, 0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # th_green, bin_green = cv2.threshold(gray_green, 0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # color_dec_bin = np.concatenate((bin_red,bin_green), axis = 1)
    # cv2.imshow('Thresholding', color_dec_bin)

    # thh_red = cv2.inRange(hsv, (0, 88, 179), (33, 255, 255))
    # thh_green = cv2.inRange(hsv, (49, 39, 130), (98, 255, 255))

    #Threshholding ----------------------------------------
    #Color dectection in binary image
    '''
    thr, binr = cv2.threshold(thh_red, 179, 255, cv2.THRESH_BINARY)
    thg, bing = cv2.threshold(thh_green, 130, 255, cv2.THRESH_BINARY)

    noiseless_image_bg = cv2.fastNlMeansDenoising(bing, None,30, 7,27)
    noiseless_image_br = cv2.fastNlMeansDenoising(binr, None,30, 7,27)
    myMask = cv2.bitwise_or(noiseless_image_bg, noiseless_image_br)
    target = cv2.bitwise_and(image_res, image_res, mask=myMask)
    '''
    #-----------------------------------------------------

    #cv2.imshow('Original', self.image)
    # cv2.imwrite('imagenJet'+str(cont)+'.png',color_dec_bin)

    # cont = cont+1
