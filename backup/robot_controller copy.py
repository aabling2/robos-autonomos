#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import rospy

# Odometria contendo posição, orientação, vel. linear e angular
from nav_msgs.msg import Odometry

# Twist é a velocidade linear e angular
from geometry_msgs.msg import Twist

# Amostras de distâncias obtidas pelo LaserScan
from sensor_msgs.msg import LaserScan


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
        self.speed_linear_fast = 0.15  # m/s
        self.speed_linear_slow = 0.07  # m/s
        self.speed_angular_fast = 2.0  # rad/s
        self.speed_angular_slow = 0.1  # rad/s

        # Estado do seguidor de parede
        self.robot_state = "turn left"

        # Distância a ser mantida da parede
        self.dist_wall_low = 0.5  # metros
        self.dist_wall_med = 0.7  # metros
        self.dist_wall_high = 0.75  # metros

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

        return roll_x, pitch_y, yaw_z  # radianos

    # Callback de odometria do robô (posição e orientação)
    # em '/jackal_velocity_controller/odom'
    def odom_callback(self, msg):
        # Posições x, y, z.
        # Orientações x, y, z, w quaternion
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

    # Atualização do estado do robô na referência do frame global
    def state_estimate_callback(self, msg):
        curr_state = msg.data
        self.current_x = curr_state[0]
        self.current_y = curr_state[1]
        self.current_yaw = curr_state[2]

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
        d1 = self.dist_wall_low
        d2 = self.dist_wall_med
        d3 = self.dist_wall_high

        if self.leftfront_dist < d1 or self.front_dist < d1 or self.rightfront_dist < d1:
            self.robot_state = "stop"
            msg.linear.x = 0
        elif self.leftfront_dist < d2 or self.front_dist < d2 or self.rightfront_dist < d2:
            self.robot_state = "slow"
            msg.linear.x = self.speed_linear_slow
        else:
            self.robot_state = "fast"
            msg.linear.x = self.speed_linear_fast

        if self.front_dist < d2:
            self.robot_state += " turning fast inv."
            msg.angular.z = self.speed_angular_fast
        elif self.right_dist < d1 or self.rightfront_dist < d2:
            self.robot_state += " turning slow inv."
            msg.angular.z = self.speed_angular_slow
        else:
            self.robot_state += " turning slow to wall"
            msg.angular.z = -self.speed_angular_slow

        """if self.front_dist > d:
            if self.leftfront_dist > d and self.rightfront_dist > d:
                self.robot_state = "search for wall"
                msg.linear.x = self.speed_linear_fast
                msg.angular.z = -self.speed_angular_fast

            elif self.leftfront_dist > d and self.rightfront_dist < d:
                self.robot_state = "follow wall"
                msg.linear.x = self.speed_linear_fast
                msg.angular.z = -self.speed_angular_fast

            elif self.leftfront_dist < d and self.rightfront_dist > d:
                self.robot_state = "search for wall"
                msg.linear.x = self.speed_linear_slow
                # msg.angular.z = -self.speed_angular_slow

            elif self.leftfront_dist < d and self.rightfront_dist < d:
                self.robot_state = "search for wall"
                msg.linear.x = self.speed_linear_slow
                # msg.angular.z = -self.speed_angular_slow

        else:
            msg.linear.x = self.speed_linear_slow
            if self.leftfront_dist > d and self.rightfront_dist > d:
                self.robot_state = "turn left"
                # msg.angular.z = self.speed_angular_slow

            elif self.leftfront_dist > d and self.rightfront_dist < d:
                self.robot_state = "turn left"
                # msg.angular.z = -self.speed_angular_slow

            elif self.leftfront_dist < d and self.rightfront_dist > d:
                self.robot_state = "turn left"
                msg.linear.x = 0
                # msg.angular.z = self.speed_angular_slow

            elif self.leftfront_dist < d and self.rightfront_dist < d:
                self.robot_state = "turn left"
                # msg.angular.z = 0"""

        # Envia mensagem da velocidade atualizada
        self.publisher.publish(msg)

        # Mostra distâncias detectadas pelo LaserScan
        rospy.loginfo(
            " 180=" + str(round(self.left_dist, 2)) +
            " 135=" + str(round(self.leftfront_dist, 2)) +
            " 90=" + str(round(self.front_dist, 2)) +
            " 45=" + str(round(self.rightfront_dist, 2)) +
            " 0=" + str(round(self.right_dist, 2)) +
            " " + self.robot_state
        )


def main():
    # Cria node do controlador do robô
    rospy.init_node('Controller', anonymous=True)
    controller = Controller()

    rate = rospy.Rate(10)  # 10hz

    # Espera tópico do laser abrir
    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('/front/scan', LaserScan, timeout=2)
        except Exception:
            pass

    while not rospy.is_shutdown():
        controller.follow_wall()
        rate.sleep()

    # Aguarda finalizar o processo
    rospy.spin()
    del controller


if __name__ == "__main__":
    main()
