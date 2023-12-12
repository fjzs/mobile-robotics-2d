#!/usr/bin/env python

import numpy as np
from slam import wraptopi
#import tf
# from tf.transformations import euler_from_quaternion, quaternion_from_euler


def clip_angles(vector):
    """Clips the angles of a vector organized as 
    (x1, y1, theta1, x2, y2, theta2, ..., xn, yn, thetan)

    Args:
        vector (np.ndarray): 

    Returns:
        None
    """
    assert isinstance(vector, np.ndarray), "it is not a numpy array"
    assert len(vector.shape) == 1, "it is not a vector"
    m = len(vector)
    assert m >= 3
    assert m % 3 == 0, "its size is not multiple of 3"
    for i in range(2, m, 3):
        vector[i] = wraptopi(vector[i])


class KalmanFilter(object):
    def __init__(self):
        pass

    def kalman_filter(self, tag_id_to_state_index, state_prev, Sigma_prev, cmd_vel, measurements, dt, R_Motion_value, Q_Sensor_value):

        # tag_id_to_state_index is a dictionary mapping the id of each april tag to its position on state vector
        # example: if I detect landmarks in this order: 4, 3, 0, then the dict will have:
        # {4: 1, 3: 2, 0: 3}, because the state will be:
        # (xr, yr, tr, x4, y4, t4, x3, y3, t3, x0, y0, t0)
        assert isinstance(tag_id_to_state_index, dict)
        for k,v in tag_id_to_state_index.items():
            assert v >= 1

        # state_prev is a state_size = M = (3 + 3N), where N is the total number of landmarks discovered so far
        assert isinstance(state_prev, np.ndarray)
        assert len(state_prev.shape) == 1
        M = len(state_prev)     # state length
        N = (M - 3)//3          # number of landmarks discovered 
        assert M >= 3
        assert N >= 0
        assert N == len(tag_id_to_state_index), "N={}, len tags={}".format(N, len(tag_id_to_state_index))
        self.M = M
        self.N = N
        print("   Kalman Filter with M={}, N={}".format(self.M, self.N))

        # Sigma_prev is a (M, M) matrix
        assert isinstance(Sigma_prev, np.ndarray)
        assert Sigma_prev.shape == (M, M), "it is {}".format(Sigma_prev.shape)
        
        # cmd_vel is a (3,) vector (vx, vy, om_z)
        assert isinstance(cmd_vel, np.ndarray)
        assert len(cmd_vel) == 3
        assert len(cmd_vel.shape) == 1
        
        # measurements is a dictionary with the observations
        # tag_id -> [x, y, theta] in robot frame
        assert isinstance(measurements, dict)
        
        # In this variable we store the missing landmarks in the state vector indexing, which is
        # 0 -> robot
        # 1 -> 1st landmark discovered (id = 4)
        # 2 -> 2nd landmark discovered (id = 7)
        # So if the id=7 tag is not detected, then we would store 2 as missing
        missing_object_indices_in_state = set(range(1,N+1))
        
        # Now we can construct z vector, which has shape 3*N, N=number of landmarks discovered
        z = np.zeros(3*N)
        for tag_id, x_y_om in measurements.items():
            assert len(x_y_om) == 3
            index_state = tag_id_to_state_index[tag_id]
            missing_object_indices_in_state.remove(index_state)
            index_z = index_state - 1
            z[3*index_z: 3*index_z + 3] = np.asarray(x_y_om)
        self.missing_object_indices_in_state = missing_object_indices_in_state
        print("   missing_object_indices_in_state: {}".format(missing_object_indices_in_state))

        # dt
        assert isinstance(dt, float)
        assert dt > 0

        # R_Motion_value is the unit noise of the motion, R is a (M,M) matrix
        assert isinstance(R_Motion_value, float)
        self.R = np.eye(M) * R_Motion_value 

        # Q_Sensor_value is the unit noise of the sensors, Q is a (3N, 3N) matrix
        assert isinstance(Q_Sensor_value, float)
        self.Q = np.eye(3*N) * Q_Sensor_value 

        self.state_prev = state_prev
        self.Sigma_prev = Sigma_prev
        self.cmd_vel = cmd_vel
        self.z = z
        self.dt = dt

        # Motion model
        # state_pred = A * state_prev + B * cmd_vel
        self.A = np.identity(M)
        self.B = np.zeros((M, 3))
        self.B[0:3,:] = np.identity(3)*dt

        # Sensor model
        self.compute_C_matrix()
        self.handle_missing_landmarks_C()        

        original_state = self.state_prev

        # Prediction
        self.state_prev, self.Sigma_prev = self.predict(self.A, 
                                                            self.state_prev, 
                                                            self.B, 
                                                            self.cmd_vel, 
                                                            self.Sigma_prev, 
                                                            self.R)
        
        if N > 0: # We only do correction if there are landmarks discovered
            # Correction
            self.state_prev, self.Sigma_prev = self.update(self.state_prev, 
                                                    self.Sigma_prev, 
                                                    self.Q, 
                                                    self.C, 
                                                    self.z)
        
        # Check that missing landmarks should change their estimate
        difference_state = original_state - self.state_prev
        for i in missing_object_indices_in_state:
            np.testing.assert_array_equal(difference_state[3*i: 3*i + 3], np.zeros(3))

        return self.state_prev, self.Sigma_prev


    def handle_missing_landmarks_C(self):
        """
        Given a list of missing_object_indices_in_state >= 1
        for every index i (state) not seen:
        - zero-out rows of (i-1) in C
        - zero-out cols of (i) in C
        """
        for i in self.missing_object_indices_in_state:
            self.C[3*(i-1): 3*i, :] = 0 # zero in rows (z space)
            self.C[:, 3*i: 3+(i+1)] = 0 # zero in cols (s space)


    def handle_missing_landmarks_K(self, K):
        """
        Given a list of missing_object_indices_in_state >= 1
        for every index i (state) not seen:
        zero out all cols of (i-1) (this is in z space)
        """
        for i in self.missing_object_indices_in_state:
            K[:, 3*(i-1):3*i] = 0 # zero all the cols (z space)


    def compute_C_matrix(self):
        """Computes the Sensor Matrix (C) that relates an observation z to the state x with 
        z = Cx. C has shape (3N, M)
        """
        # Sensor model
        # z = C1 @ C2 * state
        # Where C1 is the rob_T_world rotation and C2 is the translation to robot frame
        theta = self.state_prev[2]
        robot_R_world = np.asarray([ # 2x2 rotation matrix
            [np.cos(-theta),-np.sin(-theta)],
            [np.sin(-theta), np.cos(-theta)]
        ])
        
        # This is the rotation component
        C1 = np.zeros((3*self.N, 3*self.N))
        for i in range(self.N):
            C1[3*i: 3*i+2, 3*i: 3*i+2] = robot_R_world
            C1[3*i + 2, 3*i + 2] = 1
        
        # This is the translation component
        C2 = np.zeros((3*self.N, self.M))
        for i in range(self.N):
            C2[3*i:3*i+3, 0:3] = np.eye(3)*-1
        C2[:, 3:] = np.eye(3*self.N)
        
        self.C = np.dot(C1, C2)


    def predict(self, A , previous_state, B, command, Previous_Cov, Motion_Noise):
        current_state = np.matmul(A, previous_state) + np.matmul(B, command)
        # print("\nWrapping to pi:")
        # print("  before: {}".format(np.round(current_state, 3)))
        clip_angles(current_state)
        # print("  after: {}".format(np.round(current_state, 3)))
        current_cov = np.matmul(np.matmul(A, Previous_Cov), A.T) + Motion_Noise
        return current_state, current_cov


    def handle_filter_matrix(self):
        """Creates a F matrix to be multiplied like:
        state_corrected = state_predicted + F * K * (z - Cx).

        So F is filtering the rows to be updated (in state space)
        F is (M,M) where M is the size of the state vector

        Returns:
            F (np.ndarray) of shape (M,M)
        """
        F = np.eye(self.M)
        # for i (state space) missing,  
        for i in self.missing_object_indices_in_state:
            F[3*i: 3*(i+1), 3*i: 3*(i+1)] = np.zeros((3,3))
        return F


    def update(self, predicted_state, Predicted_Covariance, Sensor_Noise, Sensor_Matrix, measurements):
        # Compute Kalman gain
        part1 = np.linalg.inv(np.dot(Sensor_Matrix, np.dot(Predicted_Covariance, Sensor_Matrix.T)) + Sensor_Noise)
        gain = np.dot(Predicted_Covariance, np.dot(Sensor_Matrix.T, part1))
        self.handle_missing_landmarks_K(gain)
        #print("K shape: {}".format(gain.shape))

        # Correction state:
        # z_predicted = C*mu_pred
        # innovation = z_observed - z_predicted
        # mu_corrected = mu_pred + K*innovation
        z_predicted = np.dot(Sensor_Matrix, predicted_state)
        # print("Clipping z_predicted")
        # print("  before: {}".format(np.round(z_predicted, 3)))
        clip_angles(z_predicted)
        # print("  after: {}".format(np.round(z_predicted, 3)))
        
        innovation = measurements - z_predicted
        #print("innovation shape: {}".format(innovation.shape))
        # print("Clipping innovation")
        # print("  before: {}".format(np.round(innovation, 3)))
        clip_angles(innovation)
        # print("  after: {}".format(np.round(innovation, 3)))

        K_times_innovation = np.dot(gain, innovation)
        #print("K_times_innovation shape: {}".format(K_times_innovation.shape))
        # print("Clipping K_innovation")
        # print("  before: {}".format(np.round(K_innovation, 3)))
        clip_angles(K_times_innovation)
        # print("  after: {}".format(np.round(K_innovation, 3)))

        F = self.handle_filter_matrix()
        #print("F shape: {}".format(F.shape))
        
        corrected_state = predicted_state + np.dot(F, K_times_innovation)
        # print("Clipping corrected_state")
        # print("  before: {}".format(np.round(corrected_state, 3)))
        clip_angles(corrected_state)
        # print("  after: {}".format(np.round(corrected_state, 3)))
        
        # Correction covariance
        Kt_Ct = np.dot(gain, Sensor_Matrix)
        I = np.eye(Kt_Ct.shape[0])
        corrected_covar = np.dot((I - Kt_Ct), Predicted_Covariance)        
        
        return corrected_state, corrected_covar
