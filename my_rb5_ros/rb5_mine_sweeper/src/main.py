#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

from april_detection.msg import AprilTagDetectionArray
from geometry_msgs.msg import Twist
import json
import numpy as np
from planner import MineSweeperPath
from pid_controller import PIDcontroller, coord, genTwistMsg
from representation import WorldModel
import rospy
import tf
import matplotlib.pyplot as plt

# CONFIGURATION
MOVE_ROBOT = False
ACTIVATE_APRIL_TAGS = True
SHOW_APRIL_TAG_LOCALIZATION = True
SHOW_ROBOT_UPDATE_STATE = True
FLAG_DRIVE_TO_WAYPOINTS = False
FLAG_USE_ROS = True
FLAG_MEASURE_LOCALIZATION_ERROR = True

MAX_NODES_BETWEEN_WAYPOINTS = 3
SECONDS_BETWEEN_PUBLISH = 0.1 # this is delta t
THRESHOLD_ARRIVED = 0.15
MAX_APRIL_TAG_ID = 11
DECIMALS_TO_PRINT = 2
MAX_ITERATIONS = 50000 # to reach the final waypoint
INITIAL_ROBOT_STATE = np.asarray([0.1, 1.9, 0])
current_state_global = np.copy(INITIAL_ROBOT_STATE)

# PID CONFIG: this controls how fast the robot goes
MAX_PID_VAL = 0.5 # 0.6
MIN_PID_VAL = 0.4 # 0.4
PID_K_P = 0.2
PID_K_I = 0.05
PID_K_D = 0.02

# REPRESENTATION
CELL_LENGTH_M = 0.2
MODEL_WIDTH_M = 2
MODEL_HEIGHT_M = 2
MODEL_SAFETY_NODES = 0
MODEL_OBSTACLE_CORNERS = None

# FIXED TRANSFORMATIONS
CAM_T_ROBOT = np.asarray([
        [0, -1,  0,  0.03],
        [0,  0, -1,  0.18],
        [1,  0,  0, -0.05],
        [0,  0,  0,  1.00]
    ])
ROBOT_T_CAM = np.linalg.inv(CAM_T_ROBOT)
ROTATION_SIDE = {}
# A is parallel to X, looking to +Y
# B is parallel to Y, looking to -X
# C is parallel to X, looking to -Y
# D is parallel to Y, looking to +X
ROTATION_SIDE["A"] = np.asarray([
    [-1, 0, 0 ],
    [0 , 0, -1],
    [0, -1, 0 ]]
)
ROTATION_SIDE["B"] = np.asarray([
    [0 , 0, 1 ],
    [-1, 0, 0],
    [0, -1, 0 ]]
)
ROTATION_SIDE["C"] = np.asarray([
    [1 , 0, 0 ],
    [0 , 0, 1],
    [0, -1, 0 ]]
)
ROTATION_SIDE["D"] = np.asarray([
    [0, 0, -1 ],
    [1 , 0, 0],
    [0, -1, 0 ]]
)

april_tag_translation = {} # in metres (x,y,z)
april_tag_translation[0] = np.asarray([-0.50, 1.90, 0.145])
april_tag_translation[1] = np.asarray([-0.50, 1.00, 0.145])
april_tag_translation[2] = np.asarray([-0.50, 0.10, 0.143])

april_tag_translation[3] = np.asarray([ 0.10, -0.50, 0.145])
april_tag_translation[4] = np.asarray([ 1.00, -0.50, 0.145])
april_tag_translation[5] = np.asarray([ 1.90, -0.50, 0.145])

april_tag_translation[6] = np.asarray([ 2.50, 0.10, 0.145])
april_tag_translation[7] = np.asarray([ 2.50, 1.00, 0.120])
april_tag_translation[8] = np.asarray([ 2.50, 1.90, 0.120])

april_tag_translation[9] =  np.asarray([1.90, 2.50, 0.120])
april_tag_translation[10] = np.asarray([1.00, 2.50, 0.150])
april_tag_translation[11] = np.asarray([0.10, 2.50, 0.125])


WORLD_T_APRIL = {}
for tag_id in april_tag_translation:
    T_4x4 = np.eye(4)    
    letter_tag = ""

    if tag_id in [3,4,5]:
        letter_tag = "A"
    if tag_id in [6,7,8]:
        letter_tag = "B"
    elif tag_id in [9,10,11]:
        letter_tag = "C"
    elif tag_id in [0,1,2]:
        letter_tag = "D"

    T_4x4[0:3, 0:3] = ROTATION_SIDE[letter_tag]
    T_4x4[0:3,   3] = april_tag_translation[tag_id]
    WORLD_T_APRIL[tag_id] = T_4x4
    print("\nWORLD_T_APRIL id {} is:\n".format(tag_id))
    print(WORLD_T_APRIL[tag_id])

### GLOBAL VARIABLES
april_detections = [] # List of (id, robot_T_april, robot_omega_z)
log_localization_error = []
LOC_ERROR_GT_X = 0
LOC_ERROR_GT_Y = 0
LOC_ERROR_ITERATIONS = 250

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


def apriltag_detection_callback(detections):
    """
    Processes the detection of apriltags to update the current_state_global. An AprilTagDetection
    object has:
    - id (int)
    - size (float)
    - pose (PoseStamped) 

    Args:
    - detections (AprilTagDetectionArray). It has:
        - header (Header)
        - detections (AprilTagDetection[])
    Returns:
    - None
    """
    global current_state_global
    global LOC_ERROR_ITERATIONS

    n = len(detections.detections)
    if ACTIVATE_APRIL_TAGS and n > 0:
        states = [estimate_state_from_april_detection(d) for d in detections.detections if d.id <= MAX_APRIL_TAG_ID]
        
        # # This code is for taking the average
        # avg_x = sum(s[0] for s in states) / n
        # avg_y = sum(s[1] for s in states) / n        
        # # https://en.wikipedia.org/wiki/Circular_mean
        # sum_sin = sum(np.sin(s[2]) for s in states)
        # sum_cos = sum(np.cos(s[2]) for s in states)
        # avg_theta = np.arctan2(sum_sin, sum_cos)
        # current_state_global = np.asarray([avg_x, avg_y, avg_theta])        
        
        # This code is for taking the median, should handle better outliers
        current_state_global[0] = np.median([state[0] for state in states])
        current_state_global[1] = np.median([state[1] for state in states])
        current_state_global[2] = np.median([state[2] for state in states])
        
        if FLAG_MEASURE_LOCALIZATION_ERROR:
            log_localization_error.append([
                current_state_global[0], 
                current_state_global[1],
                LOC_ERROR_GT_X,
                LOC_ERROR_GT_Y 
                ])
            LOC_ERROR_ITERATIONS -= 1
            print("loc iterations left: {}".format(LOC_ERROR_ITERATIONS))
            if LOC_ERROR_ITERATIONS == 0:
                save_logfile(log_localization_error, "loc_error.json")
                exit()                    


        print("   \nRobot state update:")
        print("      world rob state: {}, {}, {} [rad] or {} [deg]".format(
            round(current_state_global[0], DECIMALS_TO_PRINT), # x
            round(current_state_global[1], DECIMALS_TO_PRINT), # y
            round(current_state_global[2], DECIMALS_TO_PRINT), # theta rad
            round(current_state_global[2]*180/np.pi), # theta deg
        ))
        

def estimate_state_from_april_detection(detection):
    """
    Process a detection (AprilTagDetection) to update the current state of the robot.
    world_T_robot = world_T_april * (april_T_cam * cam_T_robot).
    Recall that april_T_cam = (cam_T_april)^-1

    Args:
    - detection (AprilTagDetection)
        - header (Header)
        - id (int)        
        - pose (Pose): quaternion
            - position (Point): (x,y,z) 
            - orientation (Quaternion): (x,y,z,w)
    
    Returns:
    - state_world (np.ndarray): [x, y, theta] (in world frame)

    Source: https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Pose.html
    """
    
    id = detection.id
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
    #print("\n")
    # CHECK
    #print("cam_T_april:\n{}".format(cam_T_april))
    #cam_T_april_angles = tf.transformations.euler_from_matrix(cam_T_april, axes="rxyz")
    #print("cam_T_april_angles = {}".format(cam_T_april_angles))
    
    # CHECK
    april_T_cam = np.linalg.inv(cam_T_april) # 4x4
    #print("\napril_T_cam:\n{}".format(april_T_cam))
    
    # CHECK APRIL TAG
    #print("\nworld_T_april id={}:\n{}".format(id, WORLD_T_APRIL[id]))

    # CHECK CAM_T_ROBOT
    #print("\nCAM_T_ROBOT:\n{}".format(CAM_T_ROBOT))
    
    april_T_robot = np.dot(april_T_cam, CAM_T_ROBOT)
    #print("\napril_T_robot:\n{}".format(april_T_robot)) # check

    world_T_robot = np.dot(WORLD_T_APRIL[id], april_T_robot) # 4x4
    #print("\nworld_T_robot:\n{}".format(world_T_robot))
    angle_x, angle_y, angle_z = tf.transformations.euler_from_matrix(world_T_robot)
    translation = world_T_robot[0:3, 3]/world_T_robot[3,3] # (3,)
    #print("angles (x,y,z) = {}, {}, {}".format(angle_x, angle_y, angle_z))
    if SHOW_APRIL_TAG_LOCALIZATION:
        print("      tag_id: {}".format(id))
        #print("      theta_z: {} degrees".format(round(angle_z*180/np.pi, DECIMALS_TO_PRINT)))
        print("         world rob om_z: {} rad or {} deg".format(round(angle_z, DECIMALS_TO_PRINT), round(angle_z*180/np.pi)))
        print("         world rob x, y: {}".format(np.round(translation[0:2], DECIMALS_TO_PRINT)))

    #print("---------------------------------------------")

    # This is the estimation of the state
    return np.asarray([translation[0], translation[1], angle_z])


def get_log_iteration(iteration, waypoint_index, waypoint, robot_state, error, error_norm):
    data = {}
    data['iteration'] = iteration
    data['waypoint_index'] = waypoint_index
    data['waypoint'] = waypoint.tolist()
    data['robot_state'] = robot_state.tolist()
    data['error'] = error.tolist()
    data['error_norm'] = error_norm
    return data


def drive_through_waypoints(waypoints, publisher_twist):
    """
    Given a list of waypoints [x, y, theta] in world frame, navigate the robot
    through those

    Args
    - waypoints (list): each waypoint is a np.ndarray
    - publisher_twist: object to publish velocities

    Returns:
    - log (list): [waypoint_index, wx, wy, womz, xr, yr, omzr]
    """
    assert isinstance(waypoints, list)
    
    # In this global variable we maintain the most updated state of the robot given april tags
    global current_state_global
    
    pid = PIDcontroller(
        Kp = PID_K_D,
        Ki = PID_K_I,
        Kd = PID_K_D,
        maximumNormValue = MAX_PID_VAL,
        minimumNormValue = MIN_PID_VAL,
        time_step = SECONDS_BETWEEN_PUBLISH
    )        
    
    velocities_frameWorld = np.asarray([0,0,0])
    iteration = 0
    log = []
    rospy.sleep(1)
    
    for waypoint_index, waypoint in enumerate(waypoints):
        pid.setTarget(waypoint)
        print("\nGoing to waypoint: {}".format(np.round(waypoint, DECIMALS_TO_PRINT)))

        while iteration < MAX_ITERATIONS:
            
            # To Command-C to kill this
            if rospy.is_shutdown():
                exit()

            # Compute the current status
            error = pid.getError(current_state_global, waypoint)
            error_norm = np.linalg.norm(error)
            print("\nit: {}, #w {}/{}, Wp: {}, State: {}, |e|: {}".format(
                iteration,
                waypoint_index + 1,
                len(waypoints),                
                np.round(waypoint, DECIMALS_TO_PRINT),
                np.round(current_state_global, DECIMALS_TO_PRINT),
                round(error_norm, DECIMALS_TO_PRINT))
                )
            log.append(get_log_iteration(
                iteration = iteration,
                waypoint_index = waypoint_index,
                waypoint = waypoint,
                robot_state = current_state_global,
                error = error,
                error_norm = error_norm
            ))
            
            # Plot the current state
            save_plot(log, waypoints, iteration)

            # Check reaching condition
            if error_norm < THRESHOLD_ARRIVED:
                print("-------- Reached waypoint! --------")
                break

            # Get the velocities from the PID controller and publish them
            if MOVE_ROBOT:
                # These vectors are in the world frame 
                original_velocities, original_norm, final_velocities, final_norm = pid.update(current_state_global)
                print("   pid vel origi: {}, norm: {}".format(
                    np.round(original_velocities, DECIMALS_TO_PRINT),
                    np.round(original_norm, DECIMALS_TO_PRINT)))
                print("   pid vel final: {}, norm: {}".format(
                    np.round(final_velocities, DECIMALS_TO_PRINT),
                    np.round(final_norm, DECIMALS_TO_PRINT)))                
            
            velocities_frameRobot = coord(final_velocities, current_state_global)
            publisher_twist.publish(genTwistMsg(velocities_frameRobot))

            # Update the robot position with the previous velocity
            #current_state_global += final_velocities * SECONDS_BETWEEN_PUBLISH

            # Sleep for delta t
            rospy.sleep(SECONDS_BETWEEN_PUBLISH)
            iteration += 1
    
    
    # Stop the robot and print the log    
    publisher_twist.publish(genTwistMsg(np.asarray([0,0,0])))
    save_logfile(log, "hw5log.json")
    return log


def save_plot(data, waypoints, iteration):
    """
    Plots the map and the robot in a png file
    Args:
        data (list): list of dictionaries
        waypoints (list): list of [x,y,theta]
        iteration (int)

    data['iteration'] = iteration
    data['waypoint_index'] = waypoint_index
    data['waypoint'] = waypoint.tolist()
    data['robot_state'] = robot_state.tolist()
    data['error'] = error.tolist()
    data['error_norm'] = error_norm        
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 7)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    # Plot the robot trajectory
    robot_x = []
    robot_y = []
    for t in range(len(data)):    
        x_t, y_t = data[t]["robot_state"][0:2]
        robot_x.append(x_t)
        robot_y.append(y_t)
    ax.plot(robot_x, robot_y, color='black', marker='o', markersize=1, linewidth=0.5)

    # Plot the grid
    width_cell = 0.2
    for i in range(10):
        ax.axvline(x=width_cell*(i), color='black', linestyle=":", linewidth=1)
        ax.axhline(y=width_cell*(i), color='black', linestyle=":", linewidth=1)

    # Plot the waypoints
    waypoints_x = [w[0] for w in waypoints]
    waypoints_y = [w[1] for w in waypoints]
    ax.scatter(waypoints_x, waypoints_y, s=150, color='red', marker="x", linewidths=2)

    # Plot the initial position of the robot
    ax.scatter(INITIAL_ROBOT_STATE[0], INITIAL_ROBOT_STATE[1], s=150, color='lime', marker="*", linewidths=2)

    # Save figure
    x_min, x_max, y_min, y_max = 0, 2, 0, 2
    plt.axis([x_min, x_max, y_min, y_max])
    plt.title("Robot trajectory iteration {}".format(iteration+1))
    number = str(iteration+1).zfill(5)
    plt.savefig("plots/plot" + number + ".png")
    plt.savefig("plot.png")



def save_logfile(logdata, filename):
    assert filename.endswith(".json")
    with open(filename, "w") as json_file:
        json.dump(logdata, json_file)
    print("\n\nlog saved!: {}".format(filename))



def transform_xypoints_to_waypoints(xy_points):
    """Given a list of xy_points, transform it to a list of waypoints
    Args:
        xy_points (list): list of (x, y)
    Returns:
        waypoints (list): list of np.ndarray size 3 (x, y, theta)
    """
    waypoints = []
    n = len(xy_points) # number of nodes
    assert n > 1
    for i in range(n):
        x1, y1 = xy_points[i]
        if i == 0:
            x2, y2 = xy_points[i + 1]
            theta1 = np.arctan2(y2-y1, x2-x1)
        else:
            x0, y0 = xy_points[i-1]
            theta1 = np.arctan2(y1-y0, x1-x0)
        waypoints.append(np.asarray([x1, y1, theta1]))
    return waypoints[1:] # the first node is not necessary



if __name__ == "__main__":
    if FLAG_USE_ROS:
        rospy.init_node('planning_node')
        suscriber_apriltag = rospy.Subscriber("/apriltag_detection_array",
                                            AprilTagDetectionArray, 
                                            apriltag_detection_callback, 
                                            queue_size=1)
        publisher_twist = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    
    print("\n\n\n\n\n\n")

    # Some checkings
    assert len(INITIAL_ROBOT_STATE) == 3
    assert len(current_state_global) == 3

    # Create the world representation
    world_model = WorldModel(
        cell_length_m =     CELL_LENGTH_M,
        width_m =           MODEL_WIDTH_M,
        height_m =          MODEL_HEIGHT_M,
        safety_nodes =      MODEL_SAFETY_NODES,        
        rectangle_obstacle =MODEL_OBSTACLE_CORNERS 
    )
    start_node = world_model.point_to_node(INITIAL_ROBOT_STATE[0], INITIAL_ROBOT_STATE[1])
    print("Start node is {}".format(start_node))
    planner = MineSweeperPath(grid = world_model.grid, start_node = start_node, max_length_between_waypoints = MAX_NODES_BETWEEN_WAYPOINTS)
    # This is a list of nodes
    path = planner.get_path()
    print("\nPath nodes:")
    for n in path:
        print(n)

    # Print the path with robot and target
    print("\nGRID:")
    world_model.print_grid(start_node, path)
    xy_points = [world_model.node_to_point(i,j) for (i,j) in path]
    waypoints = transform_xypoints_to_waypoints(xy_points)
    
    print("\nWaypoints:")
    for w in waypoints:
        print("  {}, {}, {} [rad] or {} [deg]".format(
            round(w[0], DECIMALS_TO_PRINT), # x
            round(w[1], DECIMALS_TO_PRINT), # y
            round(w[2], DECIMALS_TO_PRINT), # theta rad
            round(w[2]*180/np.pi), # theta deg
        ))
    
    # Now we have the list of waypoints (x, y, theta) in world frame guide the robot
    if FLAG_DRIVE_TO_WAYPOINTS:
        drive_through_waypoints(waypoints, publisher_twist)
    
    # Activate the next line to have the robot listening to the april tags
    if FLAG_USE_ROS:
        rospy.spin()