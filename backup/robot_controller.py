#!/usr/bin/env python
import math
import rospy

# Twist é a velocidade linear e angular
from geometry_msgs.msg import Twist

# Odometria contendo posição, orientação, vel. linear e angular
from nav_msgs.msg import Odometry

# Amostras de distâncias obtidas pelo LaserScan
from sensor_msgs.msg import LaserScan

# Pose do robô
# from geometry_msgs.msg import Pose


class Controller():
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

        # Posições de amostragem para LaserScan e distâncias iniciais
        self.left_dist = 99999.
        self.leftfront_dist = 99999.
        self.front_dist = 99999.
        self.rightfront_dist = 99999.
        self.right_dist = 99999.

        # Posição inicial do robô
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Velocidades
        self.speed_linear_fast = 0.25  # m/s
        self.speed_linear_slow = 0.05  # m/s
        self.speed_angular_fast = 3.0  # rad/s
        self.speed_angular_slow = 0.05  # rad/s

        # Estado do seguidor de parede
        self.robot_state = "turn left"

        # Distância a ser mantida da parede
        self.dist_wall = 0.7  # metros

    # Callback de odometria do robô (posição e orientação)
    # em '/jackal_velocity_controller/odom'
    def odom_callback(self, msg):
        # Posições x, y, z.
        # Orientações x, y, z, w quaternion.
        roll, pitch, yaw = self.euler_from_quaternion(
          msg.pose.pose.orientation.x,
          msg.pose.pose.orientation.y,
          msg.pose.pose.orientation.z,
          msg.pose.pose.orientation.w)

        obs_state_vector_x_y_yaw = [
            msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

    # Callback das velocidades lineares e angulares
    # em '/jackal_velocity_controller/cmd_vel'
    def velocity_callback(self, msg):
        # Velocidade linear em frente para o robô
        v = msg.linear.x

        # Velicidade angular em torno do eixo z do robô
        yaw_rate = msg.angular.z

    # Callback do LaserScan
    # em '/front/laser'
    def scan_callback(self, msg):
        # n_laser_beams = len(msg.ranges) = 720
        # max_angle = 270
        # r = n_laser_beams/max_angle = 2.6666 -> 2.6666*angle
        self.left_dist = msg.ranges[600]  # 225
        self.leftfront_dist = msg.ranges[480]  # 180
        self.front_dist = msg.ranges[360]  # 135
        self.rightfront_dist = msg.ranges[240]  # 90
        self.right_dist = msg.ranges[120]  # 45

        # Mostra distâncias detectadas pelo LaserScan
        rospy.loginfo(
            " 180=" + str(round(self.left_dist, 2)) +
            " 135=" + str(round(self.leftfront_dist, 2)) +
            " 90=" + str(round(self.front_dist, 2)) +
            " 45=" + str(round(self.rightfront_dist, 2)) +
            " 0=" + str(round(self.right_dist, 2)) +
            " " + self.wall_following_state
        )

    # Converte quaternions para ângulos de Euler
    def euler_from_quaternion(self, x, y, z, w):
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

        return roll_x, pitch_y, yaw_z  # in radians

    # Atualização do estado do robô na referência do frame global
    def state_estimate_callback(self, msg):
        curr_state = msg.data
        self.current_x = curr_state[0]
        self.current_y = curr_state[1]
        self.current_yaw = curr_state[2]

        # Command the robot to keep following the wall
        self.follow_wall()

    # Segue parede
    def follow_wall(self):
        # Cria mensagem Twist
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        # Distância da parede
        d = self.dist_wall

        if self.leftfront_dist > d and self.front_dist > d and self.rightfront_dist > d:
            self.wall_following_state = "search for wall"
            msg.linear.x = self.speed_linear_fast
            msg.angular.z = -self.speed_angular_slow

        elif self.leftfront_dist > d and self.front_dist < d and self.rightfront_dist > d:
            self.wall_following_state = "turn left"
            msg.angular.z = self.speed_angular_fast

        elif (self.leftfront_dist > d and self.front_dist > d and self.rightfront_dist < d):
            self.wall_following_state = "follow wall"
            msg.linear.x = self.speed_linear_fast

        elif self.leftfront_dist < d and self.front_dist > d and self.rightfront_dist > d:
            self.wall_following_state = "search for wall"
            msg.linear.x = self.speed_linear_fast
            msg.angular.z = -self.speed_angular_slow

        elif self.leftfront_dist > d and self.front_dist < d and self.rightfront_dist < d:
            self.wall_following_state = "turn left"
            msg.angular.z = self.speed_angular_fast

        elif self.leftfront_dist < d and self.front_dist < d and self.rightfront_dist > d:
            self.wall_following_state = "turn left"
            msg.angular.z = self.speed_angular_fast

        elif self.leftfront_dist < d and self.front_dist < d and self.rightfront_dist < d:
            self.wall_following_state = "turn left"
            msg.angular.z = self.speed_angular_fast

        elif self.leftfront_dist < d and self.front_dist > d and self.rightfront_dist < d:
            self.wall_following_state = "search for wall"
            msg.linear.x = self.speed_linear_fast
            msg.angular.z = -self.speed_angular_slow

        else:
            pass

        # Envia mensagem da velocidade atualizada
        self.publisher.publish(msg)


def main():
    # Cria node do controlador do robô
    rospy.init_node('Controller', anonymous=True)
    controller = Controller()

    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        controller.follow_wall()
        rate.sleep()

    # Aguarda finalizar o processo
    rospy.spin()
    del controller


if __name__ == "__main__":
    main()
