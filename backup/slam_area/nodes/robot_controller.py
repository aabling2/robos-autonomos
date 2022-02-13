#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np


# Odometria contendo posição, orientação, vel. linear e angular
from nav_msgs.msg import Odometry

# Twist é a velocidade linear e angular
from geometry_msgs.msg import Twist

# Amostras de distâncias obtidas pelo LaserScan
from sensor_msgs.msg import LaserScan


class Controller():
    def __init__(self):

        # Posições de amostragem para LaserScan e distâncias iniciais
        self.fov = 1.5*np.pi
        self.left_dist = 99999.
        self.leftfront_dist = 99999.
        self.front_dist = 99999.
        self.rightfront_dist = 99999.
        self.right_dist = 99999.

        # Posição do robô
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Velocidades
        self.speed_linear_fast = 0.20  # m/s
        self.speed_linear_slow = 0.1  # m/s
        self.speed_angular_fast = 0.7  # rad/s
        self.speed_angular_slow = 0.3  # rad/s

        # Estado do seguidor de parede
        self.robot_state = "turn left"

        # Distância a ser mantida da parede
        self.dist_wall_close = 0.5  # metros
        self.dist_wall_thresh = 0.7  # metros

        self.searching_wall = True
        self.finish = False
        self.closed = False

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

        self.velocity_publisher = rospy.Publisher(
            name='/jackal_velocity_controller/cmd_vel',
            data_class=Twist,
            queue_size=10)

    # Converte quaternions para ângulos de Euler
    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

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

        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw

    # Callback das velocidades lineares e angulares
    # em '/jackal_velocity_controller/cmd_vel'
    def velocity_callback(self, msg):
        # Velocidade linear em frente para o robô
        v = msg.linear.x

        # Velocidade angular em torno do eixo z do robô
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
        d1 = self.dist_wall_close
        d2 = self.dist_wall_thresh

        if self.searching_wall:
            self.robot_state = "searching wall..."
            if self.front_dist > d2 and self.rightfront_dist > d2:
                msg.linear.x = self.speed_linear_fast
                if self.rightfront_dist < self.front_dist:
                    msg.angular.z = -self.speed_angular_slow
                else:
                    msg.angular.z = 0
            else:
                self.searching_wall = False

        else:
            if self.leftfront_dist < d1 or self.front_dist < d1 or\
                    self.rightfront_dist < d1:
                self.robot_state = "caution"
                msg.linear.x = 0
            elif self.leftfront_dist < d2 or self.front_dist < d2 or\
                    self.rightfront_dist < d2:
                self.robot_state = "going slow, "
                msg.linear.x = self.speed_linear_slow
            else:
                self.robot_state = "going fast, "
                msg.linear.x = self.speed_linear_fast

            if self.front_dist < d2 or self.rightfront_dist < d1:
                self.robot_state += "turning fast -"
                msg.angular.z = self.speed_angular_fast
            elif self.right_dist < d1 or self.rightfront_dist < d2:
                self.robot_state += "turning slow -"
                msg.angular.z = self.speed_angular_slow
            elif self.front_dist > d1+d2 and self.rightfront_dist > d1+d2 and\
                    self.right_dist > d1+d2:
                self.robot_state += "turning fast +"
                msg.angular.z = -self.speed_angular_fast
            else:
                self.robot_state += "turning slow +"
                msg.angular.z = -self.speed_angular_slow

        if self.finish:
            self.robot_state = "stop"
            msg.linear.x = 0
            msg.angular.z = 0
            self.closed = True

        """msg.linear.x = 1
        msg.angular.z = 0.5"""

        # Envia mensagem da velocidade atualizada
        self.velocity_publisher.publish(msg)

        # Mostra distâncias detectadas pelo LaserScan
        """print(
            "\npose:" +
            " [x=" + str(round(self.current_x, 2)) +
            "; y=" + str(round(self.current_y, 2)) +
            "; yaw=" + str(round(self.current_yaw, 2)) + "]" +
            "\nrange finder:" +
            " [s180=" + str(round(self.left_dist, 2)) +
            " s135=" + str(round(self.leftfront_dist, 2)) +
            " s90=" + str(round(self.front_dist, 2)) +
            " s45=" + str(round(self.rightfront_dist, 2)) +
            " s0=" + str(round(self.right_dist, 2)) + "] " +
            "\nrobot state: " + self.robot_state + "\n"
        )"""
