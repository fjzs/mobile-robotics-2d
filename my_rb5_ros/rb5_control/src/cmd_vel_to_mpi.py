#!/usr/bin/env python

import rospy
import threading
import traceback
import math

import numpy as np
from geometry_msgs.msg import Twist
from mpi_control import MegaPiController
import time
import matplotlib.pyplot as plt


class CMDVelToMpi:
    def __init__(self):
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=False)
        self.verbose = False

        # Physical params
        self.WHEEL_RADIUS = rospy.get_param("~wheel_radius", 0.033)
        self.LENGTH_X = rospy.get_param("~length_x", 0.06)
        self.LENGTH_Y = rospy.get_param("~length_y", 0.08)

        self.x = []
        self.y = []

        # Stop the robot
        self.mpi_ctrl.setFourMotors(0, 0, 0, 0)

        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback, queue_size=1)

    def cmd_vel_callback(self, cmd_vel_msgs):
        if(rospy.is_shutdown()):
            self.mpi_ctrl.carStop()

        x = cmd_vel_msgs.linear.x
        y = cmd_vel_msgs.linear.y
        omega = cmd_vel_msgs.angular.z
        print("\nVx, Vy, Omz: {}, {}, {}".format(x, y, omega))
        #self.x.append(x)
        #self.y.append(y)

        # take x, y, theta and plug it to the motor equations r=1.3cm lx=6, ly=8
        speed1 = (1/self.WHEEL_RADIUS) * (x-y-(self.LENGTH_X + self.LENGTH_Y)*omega)
        speed2 = (1/self.WHEEL_RADIUS) * (x+y+(self.LENGTH_X + self.LENGTH_Y)*omega)
        speed3 = (1/self.WHEEL_RADIUS) * (x+y-(self.LENGTH_X + self.LENGTH_Y)*omega)
        speed4 = (1/self.WHEEL_RADIUS) * (x-y+(self.LENGTH_X + self.LENGTH_Y)*omega)

        print("Initial Omega speeds: {}, {}, {}, {}".format(
            round(speed1,1), 
            round(speed2,1),
            round(speed3,1),
            round(speed4,1)))
        
        # omega = 60 deg/s, 3 seconds, Slope = 2.5, Min = 35

        SLOPE = 2.5
        MINIMUM = 35
        speed1 = speed1*SLOPE + np.sign(speed1) * MINIMUM
        speed2 = speed2*SLOPE + np.sign(speed2) * MINIMUM
        speed3 = speed3*SLOPE + np.sign(speed3) * MINIMUM
        speed4 = speed4*SLOPE + np.sign(speed4) * MINIMUM

        print("Power wheel: {}, {}, {}, {}".format(
            round(speed1,1), 
            round(speed2,1),
            round(speed3,1),
            round(speed4,1)))

        if self.verbose:
            rospy.loginfo("The speed is %f %f %f %f: FR,BL,BR,FL", speed2, speed3, speed4, speed1)

        print("Final Speeds: {}, {}, {}, {}".format(speed1, speed2, speed3, speed4))

        if (x == 0.0 and y == 0.0 and omega == 0.0):
            self.mpi_ctrl.setFourMotors(0, 0, 0, 0)
        
        else:
            self.mpi_ctrl.setFourMotors(speed2, -speed3, speed4, -speed1)
       

if __name__ == "__main__":
    rospy.init_node('CMD_To_Motor')
    mpi_ctrl_node = CMDVelToMpi()
    try:
        rospy.spin()
    except BaseException:
        #print("the x and y are: ", mpi_ctrl_node.x, mpi_ctrl_node.y)
        #plt.plot(mpi_ctrl_node.x, mpi_ctrl_node.y, 'o')
        #plt.show()
        pass
