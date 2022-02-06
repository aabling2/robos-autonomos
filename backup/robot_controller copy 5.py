#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cmath import pi
from matplotlib.pyplot import axis
import rospy
import numpy as np


# Odometria contendo posição, orientação, vel. linear e angular
from nav_msgs.msg import Odometry

# Twist é a velocidade linear e angular
from geometry_msgs.msg import Twist

# Amostras de distâncias obtidas pelo LaserScan
from sensor_msgs.msg import LaserScan

# Mapping
from ekf_slam.robot import Robot
from ekf_slam.plotmap import plotMap, plotEstimate, plotMeasurement, plotError
from ekf_slam.ekf import predict, update


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

        self.velocity_publisher = rospy.Publisher(
            name='/jackal_velocity_controller/cmd_vel',
            data_class=Twist,
            queue_size=10)

        # Posições de amostragem para LaserScan e distâncias iniciais
        self.left_dist = 99999.
        self.leftfront_dist = 99999.
        self.front_dist = 99999.
        self.rightfront_dist = 99999.
        self.right_dist = 99999.

        # Posição do robô
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Velocidades
        self.speed_linear_fast = 0.20  # m/s
        self.speed_linear_slow = 0.1  # m/s
        self.speed_angular_fast = 2.0  # rad/s
        self.speed_angular_slow = 0.3  # rad/s

        # Estado do seguidor de parede
        self.robot_state = "turn left"

        # Distância a ser mantida da parede
        self.dist_wall_close = 0.5  # metros
        self.dist_wall_thresh = 0.7  # metros

        self.search_wall = True
        self.track_wall = False
        self.finish = False

        self.odometry = []
        self.range_sense = []

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

        obs_state_vector_x_y_yaw = [
            msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw

        self.odometry = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            roll,
            pitch,
            yaw
        ]

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

        self.range_sense = [
            self.left_dist,
            self.leftfront_dist,
            self.front_dist,
            self.rightfront_dist,
            self.right_dist,
        ]

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

        if self.search_wall:
            self.robot_state = "searching wall..."
            if self.front_dist > d2 and self.rightfront_dist > d2:
                msg.linear.x = self.speed_linear_fast
                if self.rightfront_dist < self.front_dist:
                    msg.angular.z = -self.speed_angular_slow
                else:
                    msg.angular.z = 0
            else:
                self.search_wall = False
                self.start_x = self.current_x
                self.start_y = self.current_y
                self.start_yaw = self.current_yaw

        elif (
                abs(self.current_x - self.start_x) > 15 and
                abs(self.current_y - self.start_y) > 15 and
                abs(self.current_yaw - self.start_yaw) > 15):
            self.track_wall = True

        elif self.track_wall and (
                abs(self.current_x - self.start_x) < 10 and
                abs(self.current_y - self.start_y) < 10 and
                abs(self.current_yaw - self.start_yaw) < 10):
            msg.linear.x = 0
            msg.angular.z = 0
            self.robot_state = "finish"
        else:
            if self.leftfront_dist < d1 or self.front_dist < d1 or\
                    self.rightfront_dist < d1:
                self.robot_state = "stop"
                msg.linear.x = 0
            elif self.leftfront_dist < d2 or self.front_dist < d2 or\
                    self.rightfront_dist < d2:
                self.robot_state = "slow"
                msg.linear.x = self.speed_linear_slow
            else:
                self.robot_state = "fast"
                msg.linear.x = self.speed_linear_fast

            if self.front_dist < d2 or self.rightfront_dist < d1:
                self.robot_state += " turning fast inv."
                msg.angular.z = self.speed_angular_fast
            elif self.right_dist < d1 or self.rightfront_dist < d2:
                self.robot_state += " turning slow inv."
                msg.angular.z = self.speed_angular_slow
            elif self.front_dist > d1+d2 and self.rightfront_dist > d1+d2 and\
                    self.right_dist > d1+d2:
                self.robot_state += " turning fast to wall"
                msg.angular.z = -self.speed_angular_fast
            else:
                self.robot_state += " turning slow to wall"
                msg.angular.z = -self.speed_angular_slow

        # Envia mensagem da velocidade atualizada
        self.velocity_publisher.publish(msg)

        # Mostra distâncias detectadas pelo LaserScan
        rospy.loginfo(
            " 180=" + str(round(self.left_dist, 2)) +
            " 135=" + str(round(self.leftfront_dist, 2)) +
            " 90=" + str(round(self.front_dist, 2)) +
            " 45=" + str(round(self.rightfront_dist, 2)) +
            " 0=" + str(round(self.right_dist, 2)) +
            " " + self.robot_state
        )


class odometry:
    rot1 = .0
    trans = .0
    rot2 = .0

class rangeBearing:
    landmark_id = 0
    range = .0
    bearing = .0


class EKFMapping():
    def __init__(self):
        self.num_landmarks = 15
        self.landmarks = np.random.rand(self.num_landmarks, 2) * 2
        self.odometry = odometry()
        self.range_bearing = rangeBearing()

    def odometry_model(self, pose, odometry):
        rx, ry, ra = pose
        direction = ra + odometry[3]
        rx += odometry.trans * np.cos(direction)
        ry += odometry.trans * np.sin(direction)
        ra += odometry.rot1 + odometry.rot2
        # ra = rem2pi(ra, RoundNearest)  # Round to [-π, π]
        return [rx, ry, ra]

    def observation_model(self, robot_pose, range_bearing):
        rx, ry, ra = robot_pose
        range, bearing = range_bearing.range, range_bearing.bearing
        mx = rx + range * np.cos(bearing + ra)
        my = ry + range * np.sin(bearing + ra)
        return [mx, my]

    def belief_init(self, num_landmarks):
        """μ = Vector{Union{Float32, Missing}}(missing, 3 + 2*num_landmarks)
        μ[1:3] .= 0
        S = zeros(Float32, 3+2*num_landmarks, 3+2*num_landmarks)
        S[diagind(S)[1:3]] .= 0
        S[diagind(S)[4:end]] .= 1000"""

        # return Belief(μ, Symmetric(S))

        belief = 1e6*np.eye(2*num_landmarks+3)
        belief[:3, :3] = np.zeros((3, 3))

        return belief

    def correct_odometry(odometry):
        self.odometry.rot1 = 
        self.odometry.tran =
        self.odometry.rot2 = 


    def prediction_step(self, belief, odometry):
        # Compute the new mu based on the noise-free
        # (odometry-based) motion model
        rx, ry, ra = np.mean(belief, axis=1)[0:3]
        belief.mean[0:3] = self.odometry_model([rx, ry, ra], self.odometry)

        # Compute the 3x3 Jacobian Gx of the motion model
        # Gx = Matrix{Float32}(I, 3, 3)
        """Gx = np.matrix((3, 3), dtype=np.float32)
        heading = ra + odometry.rot1
        Gx[1, 3] -= odometry.trans * np.sin(heading)  # ∂x'/∂θ
        Gx[2, 3] += odometry.trans * np.cos(heading)  # ∂y'/∂θ
        print("Gx", Gx)

        # Motion noise
        # Rx = Diagonal{Float32}([0.1, 0.1, 0.01])
        Rx = np.eye(3)*0.1
        print("Rx -- >", Rx)

        # Compute the predicted sigma after incorporating the motion
        Sxx = belief.covariance[1:3, 1:3]
        Sxm = belief.covariance[1:3, 4:]

        # S = Matrix(belief.covariance)
        S = belief.covariance.copy()
        # S[1:3, 1:3] = Gx * Sxx * Gx' + Rx
        S[1:3, 1:3] = Gx * Sxx * Gx + Rx
        S[1:3, 4:] = Gx * Sxm
        # belief.covariance = Symmetric(S)
        belief.covariance = S.copy()"""

    def correction_step(self, belief, range_bearings):
        rx, ry, ra = belief.mean[1:3]

        num_range_bearings = len(range_bearings)
        num_dim_state = len(belief.mean)

        # H = Matrix{Float32}(undef, 2 * num_range_bearings, num_dim_state) # Jacobian matrix ∂ẑ/∂(rx,ry)
        H = np.matrix((2*num_range_bearings, num_dim_state), dtype=np.float32)
        zs, zss = [], []  # true and predicted observations

        for (i, range_bearing) in enumerate(range_bearings):
            mid = range_bearing.landmark_id
            # if ismissing(belief.mean[2*mid+2])
            if belief.mean[2*mid+2]:
                # Initialize its pose in mu based on the measurement and the current robot pose
                mx, my = self.observation_model([rx, ry, ra], range_bearing)
                belief.mean[2*mid+2:2*mid+3] = [mx, my]

            # Add the landmark measurement to the Z vector
            zs = [zs, range_bearing.range, range_bearing.bearing]

            # Use the current estimate of the landmark pose
            # to compute the corresponding expected measurement in z̄:
            mx, my = belief.mean[2*mid+2:2*mid+3]
            d = [mx - rx, my - ry]
            q = np.dot(d, d)
            sqrtq = np.sqrt(q)

            zss = [zss, sqrtq, np.atan(d[2], d[1]) - ra]

            # Compute the Jacobian Hi of the measurement function h
            # for this observation
            dx, dy = d
            Hi = np.zeros((2, num_dim_state), dtype=np.float32)
            Hi[1:2, 1:3] = [-sqrtq * dx, -sqrtq * dy, 0, dy, -dx, -q] / q
            Hi[1:2, 2*mid+2:2*mid+3] = [sqrtq * dx, sqrtq * dy, -dy, dx] / q

            # Augment H with the new Hi
            H[2*i-1:2*i, 1:] = Hi

        # Construct the sensor noise matrix Q
        # Q = Diagonal{Float32}(ones(2 * num_range_bearings) * 0.01)
        Q = np.eye(2*num_range_bearings) * 0.01

        # Compute the Kalman gain K
        # K = belief.covariance * H' * inv(H * belief.covariance * H' + Q)
        K = belief.covariance * H * np.inv(H * belief.covariance * H + Q)

        # Compute the difference between the expected and recorded measurements
        dz = zs - zss
        # Normalize the bearings
        # dz[2:2:] = map(bearing->rem2pi(bearing, RoundNearest), dz[2:2:end])

        # Finish the correction step by computing the new mu and sigma.
        """belief.mean += K * dz
        I = Diagonal{Float32}(ones(num_dim_state))
        belief.covariance = Symmetric((I - K * H) * belief.covariance)"""

        # Normalize theta in the robot pose.
        # belief.mean[3] = rem2pi(belief.mean[3], RoundNearest)

    def update_map(self, odometry, range_bearings):
        # believes = []
        belief = self.belief_init(self.num_landmarks)
        self.prediction_step(belief, odometry)
        # self.correction_step(belief, range_bearings)
        # push!(believes, deepcopy(belief))

        # canvas = make_canvas(-1, -1, 11, 11)
        # HTML(animate_kalman_state(canvas, believes, range_bearingss, landmarks).to_jshtml())


def main():
    # Cria node do controlador do robô
    rospy.init_node('robosaut_controller', anonymous=True)
    controller = Controller()
    mapping = EKFMapping()

    rate = rospy.Rate(10)  # 10hz

    # Espera tópico do laser abrir
    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('/front/scan', LaserScan, timeout=2)
        except Exception:
            pass

    while not rospy.is_shutdown() or controller.finish:
        controller.follow_wall()
        mapping.update_map(controller.odometry, controller.range_sense)
        rate.sleep()

    # Aguarda finalizar o processo
    rospy.spin()
    del controller


if __name__ == "__main__":
    main()
