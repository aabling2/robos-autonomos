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
    def __init__(self, plot=False, dist_thresh_min=1, dist_thresh_max=2,
                 dist_trace_stop=1, dist_map_stop=2,
                 steps_checkpoint=20, laser_samples=10):

        self.mapsize = 5.0  # tamanho inicial do mapa
        self.plot = plot  # habilita exibição do mapa
        self.endpoint = False  # flag de ponto final
        self.check_steps = steps_checkpoint  # passos até próximo checkpoint
        self.check_count = 0  # contador de passos

        self.poses = None  # poses salvas
        self.landmarks = None  # pontos de detectados para o mapa
        self.checkpoints = None  # pontos de checkagem do robô
        self.odometry = None  # odometria
        self.laser_scan = None  # amostras do laser

        # Limiares de distância euclidiana
        self.dist_thresh_min = dist_thresh_min
        self.dist_thresh_max = dist_thresh_max
        self.dist_trace_stop = dist_trace_stop
        self.dist_map_stop = dist_map_stop

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

    # Ordena landmarks pela proximidade entre os vizinhos
    def _sort_landmarks(self, landmarks):
        N = landmarks.shape[0]
        dists = distance.cdist(landmarks, landmarks, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        used_idx = []
        sorted_landmarks = []
        idx = 0
        while len(used_idx) < N:
            d = dists[idx]
            for n in range(len(d)):
                min_idx = np.argmin(d)
                if min_idx not in used_idx:
                    sorted_landmarks.append(landmarks[min_idx])
                    used_idx.append(min_idx)
                    idx = min_idx
                    break
                else:
                    d[min_idx] = np.inf
                    continue

        return np.array(sorted_landmarks, dtype=np.float32)

    # Atualiza vetor de poses
    def _update_pose(self, pose):
        if self.poses is None:
            self.poses = np.array([pose])
        else:
            self.poses = np.vstack([self.poses, pose])

    # Atualiza vetor de landmarks
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

                landmarks = self._sort_landmarks(landmarks)

        self.landmarks = landmarks

    # Plota mapa gerado
    def _plot_map(self):
        plt.subplot(131, aspect='equal')
        plt.clf()

        fov = self.laser.rad_angle
        mapsize = self.mapsize
        x = self.poses
        l_points = self.landmarks
        c_points = self.checkpoints

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
            max_mapsize = np.max(x[:, :2])*2
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
            max_mapsize = np.max(l_points)*2
            mapsize = max_mapsize if max_mapsize > mapsize else mapsize
            plt.scatter(
                l_points[:, x_id], l_points[:, y_id],
                marker='D', s=12, color=(0, 0, 0))

        # Checkpoints salvos
        if c_points is not None:
            plt.scatter(
                c_points[:, x_id], c_points[:, y_id],
                marker='s', s=20, color=(1, 0, 0))

        # Plot
        plt.xlim([1.1*mapsize/2, 1.1*-mapsize/2])
        plt.ylim([1.1*-mapsize/2, 1.1*mapsize/2])
        plt.xlabel("Y-gazebo")
        plt.ylabel("X-gazebo")
        plt.title('Mapeando area...')
        plt.pause(0.001)
        # plt.show()
        self.mapsize = mapsize

    # Detecta final da trajetória pelo histórico de poses
    def _detect_end_trajectory(self, poses):
        if poses is not None:
            if self.check_count == self.check_steps:
                if self.checkpoints is None:
                    self.checkpoints = poses[-1, :].reshape((1, 3))
                else:
                    self.checkpoints = np.append(
                        self.checkpoints,
                        poses[-1, :].reshape((1, 3)),
                        axis=0)
                self.check_count = 0
            else:
                self.check_count += 1

            if self.checkpoints is not None:
                dists = distance.cdist(
                    poses[-1, :2].reshape(1, 2),
                    self.checkpoints[:-1, :2],
                    metric='euclidean')

                if np.any(dists < self.dist_trace_stop):
                    print(
                        "\nFinal do mapeamento detectado " +
                        "pela proximidade dos checkpoints!\n")
                    self.endpoint = True

    # Detecta final do mapeamento pela distância dos pontos ordenados
    def _detect_end_mapping(self, landmarks):

        if landmarks is not None:

            shift_points = np.roll(landmarks, -1, axis=0)
            diff = np.linalg.norm(landmarks - shift_points, axis=0)

            if np.all(diff < self.dist_map_stop, axis=0):
                print(
                    "\nFinal do mapeamento detectado " +
                    "pela proximidade dos landmarks!\n")
                self.endpoint = True

    # Forma polígono pelos pontos encontrados
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

    # Calcula área pela formula shoelace
    def _shoelace_area(self, points):
        lines = np.hstack([points, np.roll(points, -1, axis=0)])
        area = 0.5*abs(sum(x1*y2-x2*y1 for x1, y1, x2, y2 in lines))
        return round(area, 2)

    # Detecta polígono e calcula área
    def calc_area(self):
        if self.landmarks is not None:
            print("\nFechando pontos...")
            points = self.landmarks
            hull_points = self._convex_hull(points)

            print("Calculando area...")
            area = self._shoelace_area(hull_points)
            print("Area encontrada: " + str(area) + "m2")

            plt.plot(hull_points[:, 1], hull_points[:, 0], color=(0, 0, 1))

            mapsize = self.mapsize
            plt.xlim([1.1*mapsize/2, -1.1*mapsize/2])
            plt.ylim([-1.1*mapsize/2, 1.1*mapsize/2])
            plt.title('Area encontrada: ' + str(area) + 'm2')
            plt.show()

        else:
            print("Sem pontos mapeados para calcular!")

    # Atualiza estados
    def update(self):
        if not self.endpoint:
            if self.odometry is not None:
                self._update_pose(pose=self.odometry)
                self._update_landmarks(range_bearings=self.laser_scan)
                self._detect_end_trajectory(poses=self.poses)
                self._detect_end_mapping(landmarks=self.landmarks)

            if self.plot:
                self._plot_map()
