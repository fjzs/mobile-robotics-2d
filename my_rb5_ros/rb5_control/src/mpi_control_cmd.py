#!/usr/bin/env python

import rospy
import math
import time
import threading
import traceback
import numpy as np
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt


class MegaPiCMDNode:
    def __init__(self):
        # for future if multiple than one robot
        self.VEHICLE_ID = rospy.get_param("~vehicle_id", "rb5")
        self.MAX_LINEAR_SPEED = rospy.get_param("~MAX_LINEAR_SPEED", 100)
        self.MAX_ANGULAR_SPEED = rospy.get_param("~MAX_ANGULAR_SPEED", 70)
        self.Kp = rospy.get_param("~Kp", 1.0) #0.3
        self.Ki = rospy.get_param("~Ki", 0.0) #0.5
        self.Kd = rospy.get_param("~Kd", 0.0) #0.2

        self.dt = rospy.get_param("~dt", 0.2)
        self.sleet_time = rospy.get_param("~sleep_time", 0.2)

        self.previous_error = np.array((0,0,0))
        self.integral = np.array((0,0,0))
        self.current_state = np.array((0,0,0))

        self.listener = tf.TransformListener()
        self.start = time.time()
        self.debug = False

        time.sleep(5)

        # assignment 1 route
        #self.route = [(0,0,0), (-1,0,0), (-1,1,1.57), (-2,1,0), (-2,2,-1.57), (-1,1,-0.78), (0,0,0)]
        
        # assignment 2 route
        #self.route = [(0,0,0), (1,0,0), (1,2,np.pi), (0,0,0)]

        print(len(self.route))
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.x = []
        self.y = []
        for item in range (len(self.route)):
            self.previous_error = np.array((0,0,0))
            self.integral = np.array((0,0,0))
            while (np.linalg.norm(np.array(self.route[item]) - self.current_state) > 0.4):
                self.cmd_publisher(self.dt, np.array(self.route[item]), self.current_state, item)

            rospy.loginfo("Reached Waypoint %s", str(self.route[item]))
            
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

        # for debugging
        #print("the x and y are: ", x, y)
        # plt.plot(self.x, self.y, 'o')
        # plt.show()

    def cmd_publisher(self, dt, setpoint, y, item):
        body = 'body_' + str(item - 1)
        sensor_y = np.array((0.0,0.0,0.0))
        error = np.array((0.0,0.0,0.0))
        rospy.loginfo("The body is----------->%s", str(body))
        rospy.loginfo("the current state is: %s", str(self.current_state))
        
        try:
            (trans,rot) = self.listener.lookupTransform('world', body, rospy.Time(0))
            (roll, pitch, yaw) = euler_from_quaternion(rot)
            sensor_y[0] = trans[0]
            sensor_y[1] = trans[1]
            sensor_y[2] = yaw
            
            # error = setpoint - y
            error = setpoint - sensor_y
            self.integral = self.integral + error * self.dt
            derivative = np.array(error - self.previous_error) / self.dt
            u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
            self.previous_error = error
            theta = self.current_state[2]
            self.current_state = sensor_y
            # self.current_state = u * dt + y

            R =  np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
            u[:2] = np.matmul(R, u[:2])

            # compute vx, vy, w
            twist = Twist()
            twist.linear.x = u[0]
            twist.linear.y = u[1]
            twist.linear.z = 0.0

            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = u[2]
            self.cmd_vel_pub.publish(twist)
            
            if time.time() - self.start > 0.5:
                self.x.append(sensor_y[0])
                self.y.append(sensor_y[1])
                self.start = time.time()

            time.sleep(self.sleet_time)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("No transformation yet rotating to detect apriltag ****************")
            rospy.logerr("The goal point is, %s", str(setpoint))
            error = setpoint - y
            #error = setpoint - sensor_y
            self.integral = self.integral + error * self.dt
            derivative = np.array(error - self.previous_error) / self.dt
            u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
            self.previous_error = error
            theta = self.current_state[2]
            #self.current_state = sensor_y
            self.current_state = u * dt + y

            R =  np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
            u[:2] = np.matmul(R, u[:2])

            # compute vx, vy, w
            twist = Twist()
            twist.linear.x = u[0]
            twist.linear.y = u[1]
            twist.linear.z = 0.0

            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = u[2]
            self.cmd_vel_pub.publish(twist)
            
            if time.time() - self.start > 0.5:
                self.x.append(sensor_y[0])
                self.y.append(sensor_y[1])
                self.start = time.time()

            time.sleep(self.sleet_time)
            
            # twist = Twist()
            # twist.linear.x = 0.0
            # twist.linear.y = 0.0
            # twist.linear.z = 0.0

            # twist.angular.x = 0.0
            # twist.angular.y = 0.0
            # twist.angular.z = 0.001
            # self.cmd_vel_pub.publish(twist)
            # time.sleep(self.sleet_time)

            # if time.time() - self.start > 0.2:
            #     self.x.append(twist.linear.x)
            #     self.y.append(twist.linear.y)
            #     self.start = time.time()
            # return

        if self.debug:
            rospy.loginfo("integral %s", str(self.integral))
            rospy.loginfo("derivative %s", str(derivative))
            rospy.loginfo("error %s", str(error))
            rospy.loginfo("The u state %s", str(u))
            rospy.loginfo("The current state %s", self.current_state)
        

if __name__ == "__main__":
    rospy.init_node('megapi_commander')
    mpi_ctrl_node = MegaPiCMDNode()
    rospy.spin()