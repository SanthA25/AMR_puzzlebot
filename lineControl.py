#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
from std_msgs.msg import Int32
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import numpy as np
import math
import sys

class DiffRobot:
    def __init__(self):
        # initial conditions
        self.x = 192
        self.y = 1
        self.theta = 0

        self.Wl = 0
        self.Wr = 0

        self.r = 0.05
        self.l = 0.19

        self.eD = 0
        self.eT = 0

        self.roboVel = 0
        self.roboW = 0

        self.errLine = 0

        #Setup ROS subscribers and publishers
        rospy.Subscriber('/wr',Float32,self.callback_wr)
        rospy.Subscriber('/wl',Float32,self.callback_wl)
        rospy.Subscriber('/LineFollower',Int32,self.callback_line)

        self.w_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)

        #setup node
        # rospy.init_node("Controller")
        self.rate = rospy.Rate(10)
        rospy.on_shutdown(self.stop)

    def callback_wr(self,data):
        self.Wr = data.data

    def callback_wl(self,data):
        self.Wl = data.data

    def callback_line(self,data):
        self.errLine = data.data

    def getLoc(self, dt):#, Wr, Wl, dt):
        self.roboVel = self.r * ((self.Wr + self.Wl) /2)
        self.roboW = self.r * ((self.Wr - self.Wl) / self.l)

        self.theta +=  self.roboW * dt#math.radians(self.roboW * dt)
        if (self.theta > np.pi):
            self.theta -= 2*np.pi
        if (self.theta < -np.pi):
            self.theta += 2*np.pi

        self.x += (self.roboVel * np.cos(self.theta) * dt)#/20
        self.y += (self.roboVel * np.sin(self.theta) * dt)#/20

    def getEd(self, xt, yt):#, xr, yr, xt, yt):
        self.eD = math.sqrt((pow(xt-self.x,2)) + (pow(yt-self.y,2)))

    def getEtheta(self, xt, yt):#, xt, yt, thetaR):
        d_y = yt - self.y
        thetaT = np.arctan2(d_y,xt-self.x)
        #thetaT = np.arctan2(yt,xt)
        self.eT = thetaT - self.theta

        if (self.eT > np.pi):
            self.eT -= 2*np.pi
        if (self.eT < -np.pi):
            self.eT += 2*np.pi

    def controllerP(self):
        # Create message for publishing

        current_time = rospy.get_time()
        last_time = rospy.get_time()

        msg = Twist()
        msg.linear.x = 0
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = 0

        kV = 0.25
        kW = 0.2

        ##########################33
        # pathPoint = 1
        # xT = self.xP#self.path[pathPoint][0]
        # yT = 1#self.path[pathPoint][1]
        ################################3

        prvLVel = 0
        prvWVel = 0

        while not rospy.is_shutdown():
            # Compute time since last loop
            current_time = rospy.get_time()
            dt = current_time - last_time
            last_time = current_time

            # get Loc, eD, eT
            # self.getLoc(dt)
            # self.getEd(xT, yT)
            # self.getEtheta(xT, yT)
            # rospy.sleep(0.5)

            if (self.errLine > 15):
                msg.linear.x = 0
                msg.angular.z  = 0.06

            if (self.errLine < -15):
                msg.linear.x = 0
                msg.angular.z  = -0.06

            if (self.errLine < 15 and self.errLine > -15):
                msg.linear.x = 0.05
                msg.angular.z  = 0

            # if (self.errLine < -100):
            #     msg.linear.x = 0
            #     msg.angular.z  = 0.06
            # if (self.errLine > 100):
            #     msg.linear.x = 0
            #     msg.angular.z  = -0.06


            # if (self.eD > 0.15 and (-0.15 < self.eT and self.eT < 0.15)):
            #     msg.linear.x = kV * self.eD
            #     msg.angular.z  = 0.0
            #     if (msg.linear.x >= 0.5):
            #         msg.linear.x = 0.5
            #     if (msg.linear.x <= -0.5):
            #         msg.linear.x = -0.5
            #
            # if (self.color == 'red'):
            #     msg.linear.x = 0
            #     msg.angular.z = 0
            #     print("Red Light")

            # if (-0.15 <= self.eT and self.eT <= 0.15 and self.eD <= 0.15):
                #
                #
                # if pathPoint<len(self.path):
                #
                #     xT = self.path[pathPoint][0]
                #     yT = self.path[pathPoint][1]
                #     pathPoint = pathPoint + 1
                #     print('next point',xT,yT)
                # else:
                #     msg.linear.x = 0
                #     msg.angular.z = 0
                #     print("Target Reached")
                #     rospy.signal_shutdown("Route completed")
                #     self.stop()


            # Publish message and sleep
            self.w_pub.publish(msg)
            self.rate.sleep()

    def stop(self):
        msg = Twist()
        msg.linear.x = 0
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = 0
        self.w_pub.publish(msg)



if __name__ == '__main__':
    rospy.init_node("Controller")

    # print("Enter target (x,y): ")
    # xT = input()
    # yT = input()

    myRobot = DiffRobot()

    try:
        myRobot.controllerP()
    except rospy.ROSInterruptException:
        None













#def getLoc(Wl, Wr, dt):
#     r = 0.05    # radius of wheels
#     l = 0.19    # distance between wheels
#
#     global x
#     global y
#     global theta
#
#     xTemp = 0
#     yTemp = 0
#     thetaTemp = 0
#
#
#     vRobot = r * ((Wr+Wl)/2)
#     wRobot = r * ((Wr-Wl) /l)
#     # print(math.radians(theta), wRobot)
#
#     theta += thetaTemp + wRobot * dt
#     x += xTemp + vRobot * math.cos(math.radians(theta)) * dt
#     y += yTemp + vRobot * math.sin(math.radians(theta)) * dt
#
#     xTemp = x
#     yTemp = y
#     thetaTemp = theta
#
#     # limit -180 < theta < 180
#     # theta = theta % 360
#     # theta = (theta+360)%360
#     # if (theta > 180):
#     #     theta -= 360
#
#     limit_x = round(x,5)
#     limit_y = round(y,5)
#     # pi_theta = math.radians(theta)
#     limit_theta = round(theta,5)
#
#     xstr = "x: " + str(limit_x)
#     ystr = "y: " + str(limit_y)
#     tstr = "theta: " + str(limit_theta)
#     print(x, y)
    # print(ystr)
    # print(tstr)
