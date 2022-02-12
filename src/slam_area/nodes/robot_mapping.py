#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.patches import Ellipse


# Odometria contendo posição, orientação, vel. linear e angular
from nav_msgs.msg import Odometry

# Amostras de distâncias obtidas pelo LaserScan
from sensor_msgs.msg import LaserScan


class Mapping():
    def __init__(self, fov=180, mapsize=10, plot=False, thresh_dist=5,
                 steps_checkpoint=20):
        self.fov = fov*(np.pi/180)
        self.mapsize = mapsize
        self.plot = plot
        self.endpoint = False

        self.poses = None
        self.range_bearings = None
        self.landmarks = None

        self.thresh_dist = thresh_dist

        self.checkpoints = None
        self.check_steps = steps_checkpoint
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
        plt.subplot(131, aspect='equal')
        plt.clf()

        fov = self.fov
        mapsize = self.mapsize
        x = self.poses
        l_points = self.landmarks
        c_points = self.checkpoints

        def stateToArrow(state):
            x = state[0]
            y = state[1]
            dx = 0.1 * np.cos(state[2])
            dy = 0.1 * np.sin(state[2])
            return x, y, dx, dy

        if self.poses is not None:
            # plot current robot state covariance
            plt.arrow(*stateToArrow(x[-1, :]), head_width=0.2, color=(1, 0, 1))

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
        if l_points is not None:
            max_mapsize = np.max(self.landmarks)*2
            mapsize = max_mapsize*1.1 if max_mapsize > mapsize else mapsize
            plt.scatter(
                l_points[:, 0], l_points[:, 1],
                marker='D', s=12, color=(0, 0, 0))

        if c_points is not None:
            plt.scatter(
                c_points[:, 0], c_points[:, 1],
                marker='s', s=20, color=(1, 0, 0))

        # plot settings
        plt.xlim([-mapsize/2, mapsize/2])
        plt.ylim([-mapsize/2, mapsize/2])
        plt.title('Mapeando area...')
        plt.pause(0.001)
        # plt.show()
        self.mapsize = mapsize

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
                    self.poses[-1, :2].reshape(1, 2),
                    self.checkpoints[:-1, :2],
                    metric='euclidean')

                if np.any(dists < self.thresh_dist):
                    self.endpoint = True

    def _convex_hull(self, points):
        hull_points = []

        # Pega ponto mais a esquerda
        start_point = points[np.argmin(points[:, 0], axis=0)]
        point = start_point
        hull_points.append(start_point)
        far_point = None

        while np.all(far_point != start_point):
            p1 = None
            for p in points:
                if np.all(p == point):
                    continue
                else:
                    p1 = p
                    break

            far_point = p1
            for p2 in points:
                if np.all(p2 == point) or np.all(p2 == p1):
                    continue
                else:
                    diff = (
                        ((p2[0] - point[0]) * (far_point[1] - point[1]))
                        - ((far_point[0] - point[0]) * (p2[1] - point[1]))
                    )
                    if diff > 0:
                        far_point = p2

            hull_points.append(far_point)
            point = far_point

        return np.array(hull_points, dtype=np.float32)

    def _shoelace_area(self, points):
        lines = np.hstack([points, np.roll(points, -1, axis=0)])
        area = 0.5*abs(sum(x1*y2-x2*y1 for x1, y1, x2, y2 in lines))
        return round(area, 2)

    def calc_area(self):
        if self.landmarks is not None:
            # plt.figure()

            print("\nFechando pontos...")
            points = self.landmarks
            hull_points = self._convex_hull(points)

            print("Calculando area...")
            area = self._shoelace_area(hull_points)
            print("Area encontrada:", area)

            plt.plot(hull_points[:, 0], hull_points[:, 1], color=(0, 0, 1))

            mapsize = self.mapsize
            plt.xlim([-mapsize/2, mapsize/2])
            plt.ylim([-mapsize/2, mapsize/2])
            plt.title('Area encontrada: ' + str(area) + 'm2')
            plt.show()

        else:
            print("Sem pontos mapeados para calcular!")

    def update(self):
        if self.odometry is not None:
            self._update_pose(pose=self.odometry)
            self._update_landmarks(range_bearings=self.laser_scan)
            self._detect_end_trajectory()

        if self.plot:
            self.plot_map()
