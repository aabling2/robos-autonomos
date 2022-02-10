#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from matplotlib.patches import Ellipse


# Odometria contendo posição, orientação, vel. linear e angular
from nav_msgs.msg import Odometry

# Amostras de distâncias obtidas pelo LaserScan
from sensor_msgs.msg import LaserScan


class Mapping():
    def __init__(self, fov=180, mapsize=10, plot=False, thresh_dist=5):
        self.fov = fov*(np.pi/180)
        self.mapsize = mapsize
        self.plot = plot

        self.poses = None
        self.range_bearings = None
        self.landmarks = None

        self.thresh_dist = thresh_dist

        self.checkpoints = None
        self.check_steps = 20
        self.check_count = 0

        self.finish = False

        self.odometry = None
        self.laser_scan = None

        self.odom_subscriber = rospy.Subscriber(
            name='/jackal_velocity_controller/odom',
            data_class=Odometry,
            callback=self.odom_callback,
            queue_size=10
        )

        self.scan_subscription = rospy.Subscriber(
            name='/front/scan',
            data_class=LaserScan,
            callback=self.scan_callback,
            queue_size=10
        )

    # Callback de odometria do robô (posição e orientação)
    # em '/jackal_velocity_controller/odom'
    def odom_callback(self, msg):

        # Coordenadas cartesianas
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y

        # Quaternion para Euler
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        current_yaw = np.arctan2(t3, t4)

        self.odometry = np.array([current_x, current_y, current_yaw])

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

        self.laser_scan = np.array([
            [self.left_dist, np.pi*1.25],  # 225
            [self.leftfront_dist, np.pi],  # 180
            [self.front_dist, np.pi*0.75],  # 135
            [self.rightfront_dist, np.pi*0.5],  # 90
            [self.right_dist, np.pi*0.25]  # 45
            ], dtype=np.float32
        )

    def update(self):
        if self.odometry is not None:
            self._update_pose(pose=self.odometry)
            self._update_landmarks(range_bearings=self.laser_scan)
            end_point = self._detect_end_trajectory()
            print("end_point", end_point)
            #conseguir enviar para controlador parar

        if self.plot:
            self.plot_map()

    def _observation_model(self, pose, range_bearings):
        rx, ry, ra = pose
        ranges, bearings = range_bearings[:, 0], range_bearings[:, 1]
        mx = rx + ranges * np.cos(bearings + ra)
        my = ry + ranges * np.sin(bearings + ra)
        return np.hstack([mx, my]).reshape((-1, 2))

    def _update_pose(self, pose):
        if self.poses is None:
            self.poses = np.array([pose])
        else:
            self.poses = np.vstack([self.poses, pose])

    def _update_landmarks(self, range_bearings):
        pose = self.poses[-1, :]
        landmarks = self.landmarks
        observations = self._observation_model(pose, range_bearings)
        if landmarks is None:
            landmarks = observations
        else:
            dists = distance.cdist(observations, landmarks, metric='euclidean')
            new_ids = np.all(dists > self.thresh_dist, axis=1)
            if True in new_ids:
                landmarks = np.append(
                    landmarks, observations[new_ids, :], axis=0)

        self.landmarks = landmarks

    def plot_map(self):
        a = plt.subplot(132, aspect='equal')
        a.cla()

        fov = self.fov
        mapsize = self.mapsize
        x = self.poses

        def stateToArrow(state):
            x = state[0]
            y = state[1]
            dx = 0.5*np.cos(state[2])
            dy = 0.5*np.sin(state[2])
            return x, y, dx, dy

        if self.poses is not None:
            # plot current robot state covariance
            a.arrow(*stateToArrow(x[-1, :]), head_width=0.5, color=(1, 0, 1))

            # plot current robot field of view
            plt.plot(
                [x[-1, 0], x[-1, 0]+50*np.cos(x[-1, 2] + fov/2)],
                [x[-1, 1], x[-1, 1]+50*np.sin(x[-1, 2] + fov/2)],
                color="r")
            plt.plot(
                [x[-1, 0], x[-1, 0]+50*np.cos(x[-1, 2] - fov/2)],
                [x[-1, 1], x[-1, 1]+50*np.sin(x[-1, 2] - fov/2)],
                color="r")

            # plot robot state history
            for i in range(1, len(x)):
                plt.plot(
                    [x[i-1, 0], x[i, 0]],
                    [x[i-1, 1], x[i, 1]],
                    color="c")

        # plot all landmarks ever observed
        if self.landmarks is not None:
            for lm in self.landmarks:
                plt.scatter(lm[0], lm[1], marker='s', s=12, color=(0, 0, 0))

        if self.checkpoints is not None:
            for c in self.checkpoints:
                plt.scatter(c[0], c[1], marker='o', s=12, color=(1, 0, 0))

        # plot settings
        plt.xlim([-mapsize/2, mapsize/2])
        plt.ylim([-mapsize/2, mapsize/2])
        plt.title('Mapeamento da area')
        plt.pause(0.001)
        # plt.show()

    def _detect_end_trajectory(self):
        if self.poses is not None:
            if self.check_count == self.check_steps:
                if self.checkpoints is None:
                    self.checkpoints = self.poses[-1, :].reshape((1, 3))
                else:
                    self.checkpoints = np.append(
                        self.checkpoints,
                        self.poses[-1, :].reshape((1, 3)),
                        axis=0)
                self.check_count = 0
            else:
                self.check_count += 1

            if self.checkpoints is not None:
                dists = distance.cdist(
                    self.poses[-1, :].reshape(1, 3),
                    self.checkpoints[:-1, :],
                    metric='euclidean')

                return True if np.any(dists < self.thresh_dist) else False

        return False

    def calc_area(self):
        print("Filtrando pontos")
        print("Calculando area")


def main():
    # Cria node do controlador do robô
    rospy.wait_for_service('gazebo/set_physics_properties')
    rospy.init_node('robosaut_mapping', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    mapping = Mapping(fov=270, mapsize=10, plot=True, thresh_dist=0.7)

    # Espera tópico do laser abrir
    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('/front/scan', LaserScan, timeout=2)
        except Exception:
            pass

    while not rospy.is_shutdown() or mapping.finish:
        mapping.update()
        rate.sleep()

    # Calcula area do mapa gerado
    mapping.calc_area()

    # Aguarda finalizar o processo
    rospy.spin()
    del mapping


if __name__ == "__main__":
    main()
