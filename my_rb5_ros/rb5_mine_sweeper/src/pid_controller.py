#!/usr/bin/env python
import sys
import rospy
from geometry_msgs.msg import Twist
import numpy as np

"""
The class of the pid controller.
"""
class PIDcontroller:
    def __init__(self, Kp, Ki, Kd, maximumNormValue, minimumNormValue, time_step):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0,0.0,0.0])
        self.timestep = time_step
        self.maximumNormValue = maximumNormValue
        self.minimumNormValue = minimumNormValue

    def setTarget(self, state):
        """
        set the target pose (np.ndarray)
        """
        assert isinstance(state, np.ndarray)
        assert len(state) == 3

        self.I = np.array([0.0,0.0,0.0]) 
        self.lastError = np.array([0.0,0.0,0.0])
        self.target = np.array(state)

    def getError(self, currentState, targetState):
        """
        return the different between two states (vector of size 3)
        """
        result = targetState - currentState
        result[2] = (result[2] + np.pi) % (2 * np.pi) - np.pi
        return result 

    def setMaximumUpdate(self, mv):
        """
        set maximum velocity for stability.
        """
        self.maximumValue = mv

    def update(self, currentState):
        """
        calculate the update value on the state based on the error between current state and target state with PID.
        """
        e = self.getError(currentState, self.target)
        P = self.Kp * e
        self.I = self.I + self.Ki * e * self.timestep 
        I = self.I
        D = self.Kd * (e - self.lastError)
        result = P + I + D
        self.lastError = e

        # scale down the twist if its norm is more than the maximum value. 
        original_velocities = result
        original_norm = np.linalg.norm(result)
        if original_norm > self.maximumNormValue:
            result = (result / original_norm) * self.maximumNormValue
            self.I = np.array([0.0, 0.0, 0.0])

        elif original_norm < self.minimumNormValue:
            result = (result / original_norm) * self.minimumNormValue
            self.I = np.array([0.0, 0.0, 0.0])
        
        final_velocities = result
        final_norm = np.linalg.norm(final_velocities)

        return original_velocities, original_norm, final_velocities, final_norm

def genTwistMsg(desired_twist):
    """
    Convert the twist to twist msg.
    """
    twist_msg = Twist()
    twist_msg.linear.x = desired_twist[0] 
    twist_msg.linear.y = desired_twist[1] 
    twist_msg.linear.z = 0
    twist_msg.angular.x = 0
    twist_msg.angular.y = 0
    twist_msg.angular.z = desired_twist[2]
    return twist_msg

def coord(twist, current_state):
    J = np.array([[np.cos(current_state[2]), np.sin(current_state[2]), 0.0],
                  [-np.sin(current_state[2]), np.cos(current_state[2]), 0.0],
                  [0.0,0.0,1.0]])
    return np.dot(J, twist)
    


if __name__ == "__main__":
    import time
    rospy.init_node("pid")
    pub_twist = rospy.Publisher("/cmd_vel", Twist, queue_size=1)