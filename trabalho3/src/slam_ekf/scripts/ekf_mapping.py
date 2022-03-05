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

        self._init_params()
        self._init_topics()

    # Parâmetros para filtro de Kalman
    def _init_params(self):
        self.belief = False
        self.bel_mean = None
        self.bel_cov = None
        self.hist_mu = []

        self.Rt = np.array([
            [1., 0, 0],
            [0, 1., 0],
            [0, 0, 1.]]) * .01

        self.Qt = np.array([
            [1., 0],
            [0, 1.]]) * .01

    # Tópicos de comunicação
    def _init_topics(self):
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

    def belief_init(self, num_landmarks):
        self.bel_mean = np.zeros(3+2*num_landmarks)
        self.bel_cov = 1e6*np.eye(3+2*num_landmarks)
        self.bel_cov[:3, :3] = .0
        self.c_prob = 0.5*np.ones((num_landmarks, 1))

    # Random motion noise
    def add_noise(self, odometry):
        motion_noise = np.matmul(np.random.randn(1, 3), self.Rt)[0]
        return odometry + motion_noise

    def convert_odometry(self, pose, odometry):
        drot1 = pose[2]
        drot2 = odometry[2]
        dtrans = np.linalg.norm(odometry[:2] - pose[:2])
        return drot1, dtrans, drot2

    def odometry_model(self, pose, odometry):
        rx, ry, ra = pose
        drot1, dtrans, drot2 = odometry

        rx += dtrans*np.cos(ra+drot1)
        ry += dtrans*np.sin(ra+drot1)
        ra += (drot1 + drot2 + np.pi) % (2*np.pi) - np.pi

        motion = np.array([
            dtrans*np.cos(ra+drot1),
            dtrans*np.sin(ra+drot1),
            drot1 + drot2])

        return motion

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
    def _update_landmarks(self, measurement):
        n = len(measurement)
        observations = np.hstack([
            measurement[np.arange(0, n-1, 1)].reshape(-1, 1),
            measurement[np.arange(1, n, 1)].reshape(-1, 1)]
        )

        landmarks = self.landmarks
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

    def _prediction_step(self, odometry):
        bel_mean = self.bel_mean
        bel_cov = self.bel_cov
        n = len(bel_mean)
        F = np.append(np.eye(3), np.zeros((3, n-3)), axis=1)
        Rt = self.Rt

        odometry = self.convert_odometry(bel_mean[:3], odometry)
        # odometry = self.add_noise(odometry)
        drot1, dtrans, drot2 = odometry
        motion = self.odometry_model(bel_mean[:3], odometry)

        # Compute the new mu based on the noise-free
        self.bel_mean = bel_mean + (F.T).dot(motion)

        # Define motion model Jacobian
        J = np.array([
            [0, 0, -dtrans*np.sin(bel_mean[2]+drot1)],
            [0, 0,  dtrans*np.cos(bel_mean[2]+drot1)],
            [0, 0,  0]])
        # J = np.zeros((3, 3))
        # J[:, 2] = motion
        Gt = np.eye(n) + (F.T).dot(J).dot(F)

        # Predict new covariance
        self.bel_cov = Gt.dot(bel_cov).dot(Gt.T) + (F.T).dot(Rt).dot(F)

        mu = self.bel_mean
        print(
            'Predicted location\t x: {0:.2} \t y: {1:.2} \t theta: {2:.2}'
            .format(mu[0], mu[1], mu[2]))

    def _correction_step(self, range_bearings):
        bel_mean = self.bel_mean
        bel_cov = self.bel_cov
        rx, ry, ra = bel_mean[:3]
        n_range_bearings = len(range_bearings)
        n_dim_state = len(bel_mean)
        # Qt = np.eye(2*n_range_bearings) * 0.01
        Qt = self.Qt

        observations = self._observation_model([rx, ry, ra], range_bearings)

        # H = Jacobian matrix ∂ẑ/∂(rx,ry)
        H = np.zeros((2*n_range_bearings, n_dim_state), dtype=np.float32)
        zs, z_hat = [], []  # true and predicted observations
        for (j, range_bearing) in enumerate(range_bearings):
            pos = 3 + 2*j

            if bel_cov[2*j+3][2*j+3] >= 1e6 and bel_cov[2*j+4][2*j+4] >= 1e6:
                # Initialize its pose in mu based on the
                # measurement and the current robot pose
                mx, my = observations[j]
                bel_mean[pos] = mx
                bel_mean[pos+1] = my
            else:
                mx, my = bel_mean[pos:pos+2]

            # Add the landmark measurement to the Z vector
            # zs.append([range_bearing[0], range_bearing[1]])
            zs = np.array([[range_bearing[0]], [range_bearing[1]]])

            # Use the current estimate of the landmark pose
            # to compute the corresponding expected measurement in z̄:
            d = [mx - rx, my - ry]
            q = np.dot(d, d)
            sq = np.sqrt(q)
            z_theta = np.arctan2(d[1], d[0])
            # z_hat.append([sq, z_theta-ra])
            z_hat = np.array([[sq], [z_theta-ra]])

            # Compute the Jacobian of this observation
            dx, dy = d
            F = np.zeros((5, n_dim_state))
            F[:3, :3] = np.eye(3)
            F[3, pos] = 1
            F[4, pos+1] = 1
            Hi = np.array([
                [-sq*dx, -sq*dy, 0, sq*dx, sq*dy],
                [dy, -dx, -q, -dy, dx]],
                dtype=np.float32)
            Hi = (1/q*Hi).dot(F)

            # Map to high dimensional space
            pos = pos - 3
            # H[pos:pos+2, :] = Hi
            H = Hi

            # ---------------------
            # Calculate Kalman gain
            K = bel_cov.dot(H.T).dot(np.linalg.inv(H.dot(bel_cov).dot(H.T)+Qt))

            # Calculate difference between expected and real observation
            z_dif = np.float32(zs) - np.float32(z_hat)
            z_dif = (z_dif + np.pi) % (2*np.pi) - np.pi

            # update state vector and covariance matrix
            bel_mean = bel_mean + (K.dot(z_dif)).reshape(-1)
            bel_cov = (np.eye(n_dim_state) - K.dot(H)).dot(bel_cov)

        self.bel_mean = bel_mean
        self.bel_cov = bel_cov
        mu = self.bel_mean
        print(
            'Updated location\t x: {0:.2} \t y: {1:.2} \t theta: {2:.2}'
            .format(mu[0], mu[1], mu[2]))

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
        if not self.belief:
            self.belief_init(num_landmarks=len(self.laser_scan))
            self.belief = True

        if self.odometry is not None:
            self._prediction_step(self.odometry)
            self._correction_step(self.laser_scan)
            self._update_pose(pose=self.bel_mean[:3])
            self._update_landmarks(measurement=self.bel_mean[3:])

        if self.plot:
            self._plot_map()
