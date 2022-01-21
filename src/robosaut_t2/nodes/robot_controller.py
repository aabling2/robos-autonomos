#!/usr/bin/env python
import math
import rospy

# Enables the use of the string message type
from std_msgs.msg import String

# Twist is linear and angular velocity
from geometry_msgs.msg import Twist

# Position, orientation, linear velocity, angular velocity
from nav_msgs.msg import Odometry

# Handles LaserScan messages to sense distance to obstacles (i.e. walls)
from sensor_msgs.msg import LaserScan

# Handle Pose messages
from geometry_msgs.msg import Pose


class Controller():
    """
    Class constructor to set up the node
    """
    def __init__(self):

        self.odom_subscriber = rospy.Subscriber(
            name='/jackal_velocity_controller/odom',
            data_class=Odometry,
            callback=self.odom_callback,
            queue_size=10
        )

        self.velocity_subscriber = rospy.Subscriber(
            name='/jackal_velocity_controller/cmd_vel',
            data_class=Twist,
            callback=self.velocity_callback,
            queue_size=10
        )

        self.scan_subscription = rospy.Subscriber(
            name='/front/scan',
            data_class=LaserScan,
            callback=self.scan_callback,
            queue_size=10
        )

        self.publisher = rospy.Publisher(
            name='/jackal_velocity_controller/cmd_vel',
            data_class=Twist,
            queue_size=10)

        # Initialize the LaserScan sensor readings to some large value
        # Values are in meters.
        self.left_dist = 999999.9  # Left
        self.leftfront_dist = 999999.9  # Left-front
        self.front_dist = 999999.9  # Front
        self.rightfront_dist = 999999.9  # Right-front
        self.right_dist = 999999.9  # Right

        # ################## ROBOT CONTROL PARAMETERS ##################
        # Maximum forward speed of the robot in meters per second
        # Any faster than this and the robot risks falling over.
        #self.forward_speed = 0.025
        self.forward_speed = 0.05

        # Current position and orientation of the robot in the global
        # reference frame
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # ############ WALL FOLLOWING PARAMETERS #######################
        # Finite states for the wall following mode
        #  "turn left": Robot turns towards the left
        #  "search for wall": Robot tries to locate the wall
        #  "follow wall": Robot moves parallel to the wall
        self.wall_following_state = "turn left"

        # Set turning speeds (to the left) in rad/s
        # These values were determined by trial and error.
        self.turning_speed_wf_fast = 3.0  # Fast turn
        self.turning_speed_wf_slow = 0.05  # Slow turn

        # Wall following distance threshold.
        # We want to try to keep within this distance from the wall.
        self.dist_thresh_wf = 0.50  # in meters

        # We don't want to get too close to the wall though.
        self.dist_too_close_to_wall = 0.19  # in meters

    def odom_callback(self, msg):
        """
        Receive the odometry information containing the position and orientation
        of the robot in the global reference frame.
        The position is x, y, z.
        The orientation is a x,y,z,w quaternion.
        """
        roll, pitch, yaw = self.euler_from_quaternion(
          msg.pose.pose.orientation.x,
          msg.pose.pose.orientation.y,
          msg.pose.pose.orientation.z,
          msg.pose.pose.orientation.w)

        obs_state_vector_x_y_yaw = [msg.pose.pose.position.x,msg.pose.pose.position.y,yaw]

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians

    def velocity_callback(self, msg):
        """
        Listen to the velocity commands (linear forward velocity
        in the x direction in the robot's reference frame and
        angular velocity (yaw rate) around the robot's z-axis.
        [v,yaw_rate]
        [meters/second, radians/second]
        """
        # Forward velocity in the robot's reference frame
        v = msg.linear.x

        # Angular velocity around the robot's z axis
        yaw_rate = msg.angular.z

    def state_estimate_callback(self, msg):
        """
        Extract the position and orientation data.
        This callback is called each time
        a new message is received on the '/demo/state_est' topic
        """
        # Update the current estimated state in the global reference frame
        curr_state = msg.data
        self.current_x = curr_state[0]
        self.current_y = curr_state[1]
        self.current_yaw = curr_state[2]

        # Command the robot to keep following the wall
        self.follow_wall()

    def scan_callback(self, msg):
        """
        This method gets called every time a LaserScan message is
        received on the '/demo/laser/out' topic
        """
        # Read the laser scan data that indicates distances
        # to obstacles (e.g. wall) in meters and extract
        # 5 distinct laser readings to work with.
        # Each reading is separated by 45 degrees.
        # Assumes 181 laser readings, separated by 1 degree.
        # (e.g. -90 degrees to 90 degrees....0 to 180 degrees)

        # number_of_laser_beams = str(len(msg.ranges))
        self.left_dist = msg.ranges[180]
        self.leftfront_dist = msg.ranges[135]
        self.front_dist = msg.ranges[90]
        self.rightfront_dist = msg.ranges[45]
        self.right_dist = msg.ranges[0]

    def follow_wall(self):
        """
        This method causes the robot to follow the boundary of a wall.
        """
        # Create a geometry_msgs/Twist message
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        # Logic for following the wall
        # >d means no wall detected by that laser beam
        # <d means an wall was detected by that laser beam
        d = self.dist_thresh_wf

        if self.leftfront_dist > d and self.front_dist > d and self.rightfront_dist > d:
            self.wall_following_state = "search for wall"
            msg.linear.x = self.forward_speed
            msg.angular.z = -self.turning_speed_wf_slow  # turn right to find wall

        elif self.leftfront_dist > d and self.front_dist < d and self.rightfront_dist > d:
            self.wall_following_state = "turn left"
            msg.angular.z = self.turning_speed_wf_fast

        elif (self.leftfront_dist > d and self.front_dist > d and self.rightfront_dist < d):
            if (self.rightfront_dist < self.dist_too_close_to_wall):
                # Getting too close to the wall
                self.wall_following_state = "turn left"
                msg.linear.x = self.forward_speed
                msg.angular.z = self.turning_speed_wf_fast
            else:
                # Go straight ahead
                self.wall_following_state = "follow wall"
                msg.linear.x = self.forward_speed

        elif self.leftfront_dist < d and self.front_dist > d and self.rightfront_dist > d:
            self.wall_following_state = "search for wall"
            msg.linear.x = self.forward_speed
            msg.angular.z = -self.turning_speed_wf_slow # turn right to find wall

        elif self.leftfront_dist > d and self.front_dist < d and self.rightfront_dist < d:
            self.wall_following_state = "turn left"
            msg.angular.z = self.turning_speed_wf_fast

        elif self.leftfront_dist < d and self.front_dist < d and self.rightfront_dist > d:
            self.wall_following_state = "turn left"
            msg.angular.z = self.turning_speed_wf_fast

        elif self.leftfront_dist < d and self.front_dist < d and self.rightfront_dist < d:
            self.wall_following_state = "turn left"
            msg.angular.z = self.turning_speed_wf_fast

        elif self.leftfront_dist < d and self.front_dist > d and self.rightfront_dist < d:
            self.wall_following_state = "search for wall"
            msg.linear.x = self.forward_speed
            msg.angular.z = -self.turning_speed_wf_slow  # turn right to find wall

        else:
            pass

        # Send velocity command to the robot
        self.publisher.publish(msg)
        rospy.loginfo(self.front_dist)


def main():
    # Create the nodes
    rospy.init_node('Controller', anonymous=True)
    controller = Controller()

    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        controller.follow_wall()
        rate.sleep()

    rospy.spin()
    del controller


if __name__ == "__main__":
    main()
