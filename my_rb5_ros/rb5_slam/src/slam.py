#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from april_detection.msg import AprilTagDetectionArray
import tf
from pid_controller import PIDcontroller, coord, genTwistMsg
import kalman_filter
import json

### CONFIGURATION
THRESHOLD_ARRIVED = 0.05
R_KALMAN_NOISE_MOTION = 0.1 # noise related to the motion model (this value then is put as a matrix)
Q_KALMAN_NOISE_MEASUR = 0.1 # noise related to the measurement model (this value then is put as a matrix)
COVARIANCE_INITIAL_NUMBER = 10.0
MAX_PID_VAL = 0.6
MIN_PID_VAL = 0.4
PID_K_P = 0.2
PID_K_I = 0.05
PID_K_D = 0.02

MOVE_ROBOT = True
MAX_GLOBAL_ITERATIONS = 1000
SECONDS_BETWEEN_PUBLISH = 0.1 # this is delta t
INITIAL_STATE = np.asarray([0.25, 0, 0], dtype=float)

WAYPOINTS_SQUARE_ROTATION_FIRST = np.asarray(
    [
        [1, 0, 0],
        [1, 0, np.pi/2],        
        [1, 1, np.pi/2],
        [1, 1, np.pi],
        [0, 1, np.pi],
        [0, 1,-np.pi/2],
        [0, 0, -np.pi/2]
    ])

WAYPOINTS_SQUARE = np.asarray(
    [
        [1, 0, 0],
        [1, 1, np.pi/2],
        [0, 1, np.pi],
        [0, 0, -np.pi/2]
    ])

WAYPOINTS_OCTA = np.asarray(
    [
        [0.75, 0.25, np.pi/2],
        [1, 1, np.pi/2],
        [1, 0.75, np.pi/2],
        [0.75, 1, np.pi],
        [0.25, 1, np.pi],
        [0, 0.75, -np.pi/2],
        [0.25, 0, 0]
    ])

WAYPOINTS = WAYPOINTS_OCTA
MAX_APRIL_TAG_ID = 7
LAPS = 1
DECIMALS_TO_PRINT = 2
CAM_T_ROBOT = np.asarray([
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [1,  0,  0,  0],
    [0,  0,  0,  1]
])
# we are assuming the robot camera is at the robot frame, but with the coord frame
# as z upward, x front and y to the left
ROBOT_T_CAM = np.linalg.inv(CAM_T_ROBOT) 

GROUND_TRUTH_TAG = {}
GROUND_TRUTH_TAG[0] = np.asarray([1.5, 0, 0])
GROUND_TRUTH_TAG[1] = np.asarray([1.5, 1, 0])
GROUND_TRUTH_TAG[2] = np.asarray([1, 1.5, np.pi/2])
GROUND_TRUTH_TAG[3] = np.asarray([0, 1.5, np.pi/2])
GROUND_TRUTH_TAG[4] = np.asarray([-0.5, 1, np.pi])
GROUND_TRUTH_TAG[5] = np.asarray([-0.5, 0, np.pi])
GROUND_TRUTH_TAG[6] = np.asarray([0, -0.5, -np.pi/2])
GROUND_TRUTH_TAG[7] = np.asarray([1, -0.5, -np.pi/2])


### GLOBAL VARIABLES
april_detections = [] # List of (id, robot_T_april, robot_omega_z)

def wraptopi(angle_rad):
    """
    Wraps angle to (-pi,pi] range

    Args:
        angle_rad (float): angle [rad]

    Returns:
        angle_rad (float): cliped to (-pi,pi]
    """
    assert isinstance(angle_rad, float)
    if angle_rad > np.pi:
        angle_rad = angle_rad - (np.floor(angle_rad / (2 * np.pi)) + 1) * 2 * np.pi
    elif angle_rad < -np.pi:
        angle_rad = angle_rad + (np.floor(angle_rad / (-2 * np.pi)) + 1) * 2 * np.pi
    return angle_rad


def get_world_T_robot(robot_state):
    assert isinstance(robot_state, np.ndarray)
    assert len(robot_state) == 3

    xr, yr, theta_r = robot_state
    world_T_robot = np.zeros((3,3)) # this is 3x3
    world_R_robot = np.asarray([
        [np.cos(theta_r),-np.sin(theta_r)],
        [np.sin(theta_r), np.cos(theta_r)]
    ])
    world_T_robot[0:2, 0:2] = world_R_robot
    world_T_robot[0, 2] = xr
    world_T_robot[1, 2] = yr
    world_T_robot[2, 2] = 1
    return world_T_robot


def april_detection_handler(detections):
    """
    Processes the detection of apriltags to store the robot_T_april transformation
    in the global variable april_detections

    Args:
    - detections (AprilTagDetectionArray). It has:
        - header (Header)
        - detections (AprilTagDetection[])
    Returns:
    - None
    """
    global april_detections
    if len(detections.detections) > 0:
        april_detections = [get_robot_T_april_from_AprilTagDetection(d) for d in detections.detections if d.id <= MAX_APRIL_TAG_ID]
        

def get_robot_T_april_from_AprilTagDetection(detection):
    """
    Process a detection (AprilTagDetection) to get the transformation cam_T_april

    Args:
    - detection (AprilTagDetection)
        - header (Header)
        - id (int)        
        - pose (Pose): quaternion
            - position (Point): (x,y,z) 
            - orientation (Quaternion): (x,y,z,w)
    
    Returns:
    - (id, robot_T_april, robot_omega_z)

    Source: https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Pose.html
    """
    id = detection.id
    assert id >= 0
    assert id <= MAX_APRIL_TAG_ID

    pose = detection.pose
    tfros = tf.TransformerROS()
    # The transformation that takes a point in april ref frame and returns it in camera frame
    cam_T_april = tfros.fromTranslationRotation((pose.position.x, 
                                                 pose.position.y, 
                                                 pose.position.z), 
                                                (pose.orientation.x,
                                                 pose.orientation.y,
                                                 pose.orientation.z,
                                                 pose.orientation.w,
                                                )) # this is 4x4 np.ndarray --> [R|t]
    angle_x, angle_y, angle_z = tf.transformations.euler_from_matrix(cam_T_april)
    # print("\n\ncam_T_april:\n{}".format(np.round(cam_T_april, DECIMALS_TO_PRINT)))
    # print("\t cam_T_april angles x,y,z: {}, {}, {}".format(
    #         round(angle_x,2),
    #         round(angle_y,2),
    #         round(angle_z,2),
    #         ))
    robot_omega_z = wraptopi(-angle_y) # this is in rad
    robot_T_april = np.dot(ROBOT_T_CAM, cam_T_april)
    return (id, robot_T_april, robot_omega_z)


def handle_new_landmarks(local_april_tags, 
                        tag_id_to_state_index, 
                        state_current,
                        covar_current, 
                        measurements):
    """If there is a new landmark detected, we update the state and covariance matrix

    Args:
    - local_april_tags(list): (id, robot_T_april, robot_omega_z)
    - tag_id_to_state_index (dict): maps april tag id to landmark position in state vector
    example: if I detect landmarks in this order: 4, 3, 0, then the dict will have:
    {4: 1, 3: 2, 0: 3}, because the state will be:
    (xr, yr, tr, x4, y4, t4, x3, y3, t3, x0, y0, t0)
    - state_current (np.ndarray)
    - covar_current (np.ndarray)
    - measurements (dict): tag_id -> [x, y, theta] in robot frame

    Returns:
    - None
    """
    assert isinstance(local_april_tags, list)
    assert isinstance(tag_id_to_state_index, dict)
    
    # this is what is stored in the latest april tag detections
    world_T_robot = get_world_T_robot(state_current[0:3])
    robot_theta_worldFrame = state_current[2]

    for (id, _, _) in local_april_tags: 
        if id not in tag_id_to_state_index: # if this is a new april tag
            detected_3 = (id == 3)
            index_state = 1 + len(tag_id_to_state_index) # index to the state position
            tag_id_to_state_index[id] = index_state
            m = len(state_current)
            x_robotFrame, y_robotFrame, april_theta_robotFrame = measurements[id] # robot frame

            # Convert to world frame
            position_world = np.dot(world_T_robot, np.asarray([x_robotFrame, y_robotFrame, 1])) # this is a 3x1 vector
            position_world = position_world / position_world[2] # homogenous coordinate
            x_april_worldFrame = position_world[0]
            y_april_worldFrame = position_world[1]
            april_theta_worldFrame = wraptopi(robot_theta_worldFrame + april_theta_robotFrame) # check

            # Update state
            state_new = np.zeros(m + 3)
            state_new[0:m] = state_current
            state_new[m+0] = x_april_worldFrame
            state_new[m+1] = y_april_worldFrame
            state_new[m+2] = april_theta_worldFrame
            state_current = state_new

            # Update covar_current
            covar_new = np.zeros((m+3, m+3))
            covar_new[0:m, 0:m] = covar_current
            covar_new[m:, m:] = np.eye(3, dtype=float)*COVARIANCE_INITIAL_NUMBER
            covar_current = covar_new

    return tag_id_to_state_index, state_current, covar_current


def get_measurements(local_april_tags):
    """
    local_april_tags list of (id, robot_T_april, robot_omega_z)
    
    Returns:
    - dictionary: tag_id -> [x, y, theta] in robot frame
    """
    assert isinstance(local_april_tags, list)    
    measurements = {}
    for (id, robot_T_april, robot_omega_z) in local_april_tags:
        
        # Handle the detection of a new landmark
        x = robot_T_april[0, 3]
        y = robot_T_april[1, 3]        
        measurements[id] = [x, y, robot_omega_z]
    return measurements


def print_state(state, tag_id_to_state_index):
    # tag_id_to_state_index: 0->1, 1->2
    
    m = len(state) // 3 # number of objects
    print("   State:")

    # state_index_to_tag_id
    # 1 -> 0, 2 -> 1
    state_index_to_tag_id = {}
    for k,v in tag_id_to_state_index.items():
        state_index_to_tag_id[v]=k

    for i in range(m):
        if i==0:
            print("     rob: {}, {}, {}".format(
                round(state[0], DECIMALS_TO_PRINT), 
                round(state[1], DECIMALS_TO_PRINT),
                round(state[2], DECIMALS_TO_PRINT)))
        else:
            tag_id = state_index_to_tag_id[i]
            error = GROUND_TRUTH_TAG[tag_id] - state[3*i: 3*i+3]
            error[2] = wraptopi(error[2])
            print("     tag{}: {}, {}, {}\t Error: {}, {}, {}".format(
                tag_id,
                round(state[3*i + 0], DECIMALS_TO_PRINT), 
                round(state[3*i + 1], DECIMALS_TO_PRINT),
                round(state[3*i + 2], DECIMALS_TO_PRINT),
                round(error[0], DECIMALS_TO_PRINT),
                round(error[1], DECIMALS_TO_PRINT),
                round(error[2], DECIMALS_TO_PRINT)
                ))

def get_log_data(lap_iteration, target_waypoint, global_iteration, state, covariance):
    data = {}
    data['metadata'] = [lap_iteration, global_iteration]
    data['waypoint'] = target_waypoint.tolist()
    data['state'] = state.tolist()
    data['covariance'] = np.diagonal(covariance).tolist()
    return data

def save_logfile(logfile):
    with open("hw3_log.json", "w") as json_file:
        json.dump(logfile, json_file)
    print("\n\nlog saved!")


def slam(waypoints, laps, publisher_twist):
    """Builds a map from given waypoints and laps

    Args:
        waypoints (np.ndarray): matrix of [x, y, theta]
        laps (int): # of laps to go through the waypoints

    Return:
        map (TODO) which is a state vector that have robot x,y,theta, and land marks
    """
    assert isinstance(waypoints, np.ndarray)
    assert isinstance(laps, int)
    assert laps >= 1

    # This is the global variable containinig the april tag detections
    global april_detections
    
    # id april tag -> index in the state vector
    # example: if I detect landmarks in this order: 4, 3, 0, then the dict will have:
    # {4: 1, 3: 2, 0: 3}, because the state will be:
    # (xr, yr, tr, x4, y4, t4, x3, y3, t3, x0, y0, t0)
    tag_id_to_state_index = dict() 
    state_current = INITIAL_STATE
    covar_current = np.eye(len(state_current), dtype=float)*COVARIANCE_INITIAL_NUMBER
    kf = kalman_filter.KalmanFilter()
    global_iteration = 0
    pid = PIDcontroller(PID_K_P, PID_K_I, PID_K_D, MAX_PID_VAL, MIN_PID_VAL, SECONDS_BETWEEN_PUBLISH)
    velocities_frameWorld = np.asarray([0,0,0], dtype=float)
    
    # List of dictionaries
    # at each iteration will save state_t and diag_cov
    log_file = []

    print("\n\n\n\n\n\n")
    state_per_iteration = {}
    for i in range(laps):
        for waypoint_index, wp in enumerate(waypoints):
            
            pid.setTarget(wp)            
            while (np.linalg.norm(pid.getError(state_current[0:3], wp)) > THRESHOLD_ARRIVED) and global_iteration < MAX_GLOBAL_ITERATIONS:
                
                # Update log
                log_file.append(get_log_data(i+1, wp, global_iteration, state_current, covar_current))                
                
                # To Command-C to kill this
                if rospy.is_shutdown():
                    exit()

                # Show current situation
                distance_to_waypoint = np.linalg.norm(pid.getError(state_current[0:3], wp))
                print("\nLap {}, Wp: {}, Wp it: {}, Dist: {}".format(
                    i + 1, 
                    waypoint_index + 1,
                    global_iteration, 
                    np.round(state_current[0:3], DECIMALS_TO_PRINT), 
                    round(distance_to_waypoint, DECIMALS_TO_PRINT)))
                
                # Get the velocities from the PID controller
                if MOVE_ROBOT:
                    velocities_frameWorld = pid.update(state_current[0:3])
                    #velocities_frameWorld = np.asarray([0, 0, np.pi/4])
                    #velocities_frameWorld = np.asarray([0.2, 0, 0])
                    print("   pid vel: {}".format(np.round(velocities_frameWorld, DECIMALS_TO_PRINT)))
                velocities_frameRobot = coord(velocities_frameWorld, state_current[0:3])

                # Make a local copy of the april tags detected
                local_april_tags = april_detections
                print("   # Detections: {}".format(len(local_april_tags)))
                measurements = get_measurements(local_april_tags) # this is in robot frame

                # Handle the case for new landmarks discovered
                tag_id_to_state_index, state_current, covar_current = handle_new_landmarks(
                    local_april_tags, 
                    tag_id_to_state_index, 
                    state_current,
                    covar_current, 
                    measurements)
                
                print("   Tag_id_to_state_index: {}".format(tag_id_to_state_index))
                print_state(state_current, tag_id_to_state_index)
                print("   Measurements:")
                if len(measurements) == 0:
                    print("     no measurements!")
                else:
                    for tag_id, vector in measurements.items():
                        print("     id #{}: {}".format(tag_id, np.round(vector, DECIMALS_TO_PRINT)))

                # Apply Kalman Filter
                state_current, covar_current = kf.kalman_filter(
                    tag_id_to_state_index,
                    state_current,
                    covar_current, 
                    velocities_frameWorld,
                    measurements,
                    SECONDS_BETWEEN_PUBLISH,
                    R_KALMAN_NOISE_MOTION,
                    Q_KALMAN_NOISE_MEASUR)
                print("   Kalman Filter...")
                print_state(state_current, tag_id_to_state_index)                

                # Erase the actual observations to avoid them to be used in the same loop in the next iteration
                april_detections = []
                local_april_tags = []

                print("   Velocities_world from PID: {}".format(np.round(velocities_frameWorld, DECIMALS_TO_PRINT)))
                publisher_twist.publish(genTwistMsg(velocities_frameRobot))

                # Sleep for delta t
                rospy.sleep(SECONDS_BETWEEN_PUBLISH)
                global_iteration += 1
                
                print("-----------------------------------------")

            print("Arrived to waypoint!!!")
            print("--------------------------------------------\n\n")

        print("ITERATION {} FINISHED".format(i+1))
        print("--------------------------------------------")
        print("--------------------------------------------\n\n")
        state_per_iteration[i] = state_current

    # Save the log file
    save_logfile(log_file) 

    # Stop the robot (publish velocities = 0)
    publisher_twist.publish(genTwistMsg(np.array([0.0,0.0,0.0])))

    # Summary of state per iteration
    print("\n\n\n")
    for i in state_per_iteration:
        print("ITERATION #{}".format(i+1))
        print_state(state_per_iteration[i], tag_id_to_state_index)


if __name__ == "__main__":
    rospy.init_node('slam_node')
    suscriber_apriltag = rospy.Subscriber("/apriltag_detection_array",
                                          AprilTagDetectionArray, 
                                          april_detection_handler, 
                                          queue_size=1)
    publisher_twist = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    iteration = rospy.get_param("~iteration", 1)
    waypoints = rospy.get_param("~waypoints", 0.0) 

    slam(WAYPOINTS, LAPS, publisher_twist)
    rospy.spin()