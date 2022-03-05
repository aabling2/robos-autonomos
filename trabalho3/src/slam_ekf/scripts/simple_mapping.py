#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Odometria contendo posição, orientação, vel. linear e angular
from nav_msgs.msg import Odometry

# Amostras de distâncias obtidas pelo LaserScan
from sensor_msgs.msg import LaserScan


# Modelo do sensor laser
class SICK_LMS511:
    range = 80  # metros
    angle = 270  # graus
    n_laser_bins = 720
    rad_angle = angle*(np.pi/180)  # 270
    rad_min_angle = -rad_angle/2  # 90 - 270/2
    rad_max_angle = rad_angle/2


# Mapeia ambiente pela odometria e pontos do sensor
class Mapping():
    def __init__(self, plot=False, dist_thresh_min=1, dist_thresh_max=2, laser_samples=10):

        self.mapsize = 5.0  # tamanho inicial do mapa
        self.plot = plot  # habilita exibição do mapa
        self.endpoint = False  # flag de ponto final

        self.poses = None  # poses salvas
        self.landmarks = None  # pontos de detectados para o mapa
        self.odometry = None  # odometria
        self.laser_scan = None  # amostras do laser

        # Limiares de distância euclidiana
        self.dist_thresh_min = dist_thresh_min
        self.dist_thresh_max = dist_thresh_max

        # Sensor laser de alcance
        self.laser = SICK_LMS511()
        self.laser_bins = np.linspace(
            0, self.laser.n_laser_bins-1,
            laser_samples, dtype=np.int32)
        self.laser_angles = np.linspace(
            self.laser.rad_min_angle, self.laser.rad_max_angle,
            laser_samples, dtype=np.float32).reshape(-1, 1)

        # Tópicos de comunicação
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
        ranges = np.array(msg.ranges)

        self.laser_scan = np.hstack([
            ranges[self.laser_bins].reshape(-1, 1),
            self.laser_angles,
        ]).reshape(-1, 2)

    # Formata coordenada observada
    def _observation_model(self, pose, range_bearings):
        rx, ry, ra = pose
        ranges, bearings = range_bearings[:, 0], range_bearings[:, 1]

        mx = rx + ranges * np.cos(bearings + ra)
        my = ry + ranges * np.sin(bearings + ra)

        mx = mx.reshape(-1, 1)
        my = my.reshape(-1, 1)

        return np.hstack([mx, my])

    # Atualiza histórico de poses
    def _update_pose(self, pose):
        if self.poses is None:
            self.poses = np.array([pose])
        else:
            self.poses = np.vstack([self.poses, pose])

    # Atualiza histórico de landmarks
    def _update_landmarks(self, range_bearings):
        pose = self.poses[-1, :]
        landmarks = self.landmarks
        observations = self._observation_model(pose, range_bearings)
        if landmarks is None:
            landmarks = observations
        else:
            dists = distance.cdist(observations, landmarks, metric='euclidean')
            new_ids_min = np.all(dists > self.dist_thresh_min, axis=1)
            new_ids_max = np.any(dists < self.dist_thresh_max, axis=1)
            new_ids = np.logical_and(new_ids_min, new_ids_max)
            if True in new_ids:
                landmarks = np.append(
                    landmarks, observations[new_ids, :], axis=0)

        self.landmarks = landmarks

    # Plota mapa gerado
    def _plot_map(self):
        plt.subplot(131, aspect='equal')
        plt.clf()

        fov = self.laser.rad_angle
        mapsize = self.mapsize
        x = self.poses
        l_points = self.landmarks

        # Inverte eixos para melhor comparação com Gazebo
        x_id, y_id = 1, 0

        def state_arrow(state, size=0.1):
            x = state[x_id]
            y = state[y_id]
            dx = size * np.cos(state[2])
            dy = size * np.sin(state[2])
            dx, dy = dy, dx  # inverte para exibir
            return x, y, dx, dy

        if self.poses is not None:
            max_mapsize = np.max(np.abs(x[:, :2]))*2
            mapsize = max_mapsize if max_mapsize > mapsize else mapsize

            # Posição e sentido do robô
            plt.arrow(*state_arrow(x[-1, :]), head_width=0.2, color=(1, 0, 1))

            # Limites do campo de visão do sensor
            mid_fov = fov/2
            plt.plot(
                [x[-1, x_id], x[-1, x_id]+50*np.sin(x[-1, 2] + mid_fov)],
                [x[-1, y_id], x[-1, y_id]+50*np.cos(x[-1, 2] + mid_fov)],
                color="r")
            plt.plot(
                [x[-1, x_id], x[-1, x_id]+50*np.sin(x[-1, 2] - mid_fov)],
                [x[-1, y_id], x[-1, y_id]+50*np.cos(x[-1, 2] - mid_fov)],
                color="r")

            # Histórico de poses
            for i in range(1, len(x)):
                plt.plot(
                    [x[i-1, x_id], x[i, x_id]],
                    [x[i-1, y_id], x[i, y_id]],
                    color="c")

        # Landmarks observadas
        if l_points is not None:
            max_mapsize = np.max(np.abs(l_points))*2
            mapsize = max_mapsize if max_mapsize > mapsize else mapsize
            plt.scatter(
                l_points[:, x_id], l_points[:, y_id],
                marker='D', s=12, color=(0, 0, 0))

        # Plot
        plt.xlim([1.1*mapsize/2, 1.1*-mapsize/2])
        plt.ylim([1.1*-mapsize/2, 1.1*mapsize/2])
        plt.xlabel("Y-gazebo")
        plt.ylabel("X-gazebo")
        plt.title('Mapeando area...')
        plt.pause(0.001)
        # plt.show()
        self.mapsize = mapsize

    # Atualiza estados
    def update(self):
        if not self.endpoint:
            if self.odometry is not None:
                self._update_pose(pose=self.odometry)
                self._update_landmarks(range_bearings=self.laser_scan)

            if self.plot:
                self._plot_map()
