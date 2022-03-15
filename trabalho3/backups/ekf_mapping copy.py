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


# Twist é a velocidade linear e angular
from geometry_msgs.msg import Twist


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
                 laser_samples=10):

        self.mapsize = 5.0  # tamanho inicial do mapa
        self.plot = plot  # habilita exibição do mapa
        self.endpoint = False  # flag de ponto final

        self.hist_poses = None  # poses salvas
        self.hist_landmarks = None  # pontos de detectados para o mapa
        self.odometry = None  # odometria
        self.laser_scan = None  # amostras do laser
        self.velocity = np.zeros(4)  # vt-1, wt-1, vt, wt

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

        self.R = np.array([
            [1., 0, 0],
            [0, 1., 0],
            [0, 0, 1.]]) * .01

        self.Q = np.array([
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
        self.velocity_subscriber = rospy.Subscriber(
            name='/jackal_velocity_controller/cmd_vel',
            data_class=Twist,
            callback=self.velocity_callback,
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

    # Callback das velocidades lineares e angulares
    # em '/jackal_velocity_controller/cmd_vel'
    def velocity_callback(self, msg):
        # Velocidade linear em frente para o robô
        v = msg.linear.x

        # Velocidade angular em torno do eixo z do robô
        yaw_rate = msg.angular.z

        # Atualiza vetor de velocidades
        self.velocity[2:4] = np.array([v, yaw_rate])

    # Probabilidades iniciais
    def _belief_init(self, num_landmarks):
        self.bel_mean = np.zeros(3+2*num_landmarks)
        self.bel_cov = 1e6*np.eye(3+2*num_landmarks)
        self.bel_cov[:3, :3] = .0
        self.c_prob = 0.5*np.ones(num_landmarks)

    # Modelos de movimento normal e Jacobiano
    def _motion_models(self, pose, odometry, velocity):
        rx0, ry0, ra0 = pose
        rx1, ry1, ra1 = odometry
        vt0, wt0, vt1, wt1 = velocity
        vw = vt1/wt1 if wt1 > .0 else .0
        dw = abs(wt1-wt0)

        # Motion model
        motion = np.array([
            -vw*np.sin(ra0) + vw*np.sin(ra0+dw),
            vw*np.cos(ra0) - vw*np.cos(ra0+dw),
            dw
        ])

        # Jacobian motion model
        jacobian = np.array([
            [0, 0, -vw*np.cos(ra0) + vw*np.cos(ra0+dw)],
            [0, 0, -vw*np.sin(ra0) + vw*np.sin(ra0+dw)],
            [0, 0, 0]
        ])

        # Atualiza velocidade anterior
        self.velocity[:2] = vt1, wt1

        return motion, jacobian

    # Cordenadas observadas pelas amostras do sensoriamento
    def _observation_model(self, pose, range_bearings):
        rx, ry, ra = pose
        ranges, bearings = range_bearings[:, 0], range_bearings[:, 1]

        mx = rx + ranges * np.cos(bearings + ra)
        my = ry + ranges * np.sin(bearings + ra)

        mx = mx.reshape(-1, 1)
        my = my.reshape(-1, 1)

        return np.hstack([mx, my])

    # Etapa de predição do estado e covariância
    def _prediction_step(self, odometry, velocity):
        bel_mean = self.bel_mean
        bel_cov = self.bel_cov
        n = len(bel_mean)
        R = self.R
        F = np.append(np.eye(3), np.zeros((3, n-3)), axis=1)

        # Define modelos de movimento normal e Jacobiano
        motion, J = self._motion_models(bel_mean[:3], odometry, velocity)

        # Predição do novo estado do robô
        self.bel_mean = bel_mean + (F.T).dot(motion)

        # Predict da nova covariância
        G = np.eye(n) + (F.T).dot(J).dot(F)
        self.bel_cov = G.dot(bel_cov).dot(G.T) + (F.T).dot(R).dot(F)

        mu = self.bel_mean
        print(
            'Predicted location\t x: {0:.2} \t y: {1:.2} \t theta: {2:.2}'
            .format(mu[0], mu[1], mu[2]))

    # Etapa de correção pela comparação entre features observadas
    def _correction_step(self, range_bearings):
        bel_mean = self.bel_mean
        bel_cov = self.bel_cov
        rx, ry, ra = bel_mean[:3]
        Q = self.Q
        thresh = 10

        # Features observadas como coordenadas cartesianas
        observations = self._observation_model([rx, ry, ra], range_bearings)

        # Landmarks previamente observados
        n = len(bel_mean[3:])
        landmarks = np.hstack([
            bel_mean[3:][np.arange(0, n-1, 2)].reshape(-1, 1),
            bel_mean[3:][np.arange(1, n, 2)].reshape(-1, 1)])

        for j, zt in enumerate(range_bearings):
            n_tot = len(bel_mean)
            pos = 3 + 2*j
            N = (n_tot-3) // 2
            mx, my = observations[j]

            zk = np.vstack([landmarks, np.array([[mx, my]])])
            Hs, Ss, Ms, zs = [], [], [], []
            for z in zk:

                # Predição do landmark esperado para a feat. observada
                delta = np.array([z[0] - rx, z[1] - ry])
                q = np.dot(delta.T, delta)
                # print(q)
                sq = np.sqrt(q)
                z_theta = np.arctan2(delta[1], delta[0])
                z_hat = np.array([sq, z_theta-ra])

                # Computa a matriz Jacobiana dessa observação
                dx, dy = delta
                F = np.zeros((5, n_tot))
                F[:3, :3] = np.eye(3)
                F[3, pos] = 1
                F[4, pos+1] = 1
                Hz = np.array([
                    [-sq*dx, -sq*dy, 0, sq*dx, sq*dy],
                    [dy, -dx, -q, -dy, dx]], dtype=np.float32)
                H = (np.nan_to_num(1/q)*Hz).dot(F)  # Jacobian matrix ∂ẑ/∂(rx,ry)
                Hs.append(H)

                # Calcula a diferença entre as obsevações reais e preditas
                z_dif = z - z_hat
                z_dif = (z_dif + np.pi) % (2*np.pi) - np.pi
                zs.append(z_dif)

                S = np.linalg.inv(H.dot(bel_cov).dot(H.T)+Q)
                M = np.dot(z_dif.T, S.dot(z_dif))  # Mahalanobis dist.
                Ss.append(S), Ms.append(M)

            # Distânciad de Mahalanobis
            """M = np.min(distance.cdist(
                landmarks, np.array([zk]),
                metric='euclidean'))"""
            """C = np.cov(landmarks.T)
            inv_C = np.linalg.pinv(C)
            x_dist = np.array([[mx], [my]]) - landmarks
            M = np.dot(x_dist, np.dot(inv_C, x_dist.T)) ** (1/2)
            dist = np.diagonal(M)"""
            print("Ss", Ss)
            print("Ms", Ms)
            i = np.argmin(Ms)
            mx, my = zk[i]
            M = Ms[i]
            H = Hs[i]
            S = Ss[i]
            z_dif = zs[i]

            # Novo landmark se houver distância maior que limiar
            print("M", M)
            if i == N+1:
                # Adiciona nova obsevação
                bel_mean = np.append(bel_mean, [mx, my])

                # Adiciona linhas e coluna para covariâncias
                bel_cov = np.vstack([bel_cov, np.zeros((2, n_tot))])
                bel_cov = np.hstack([bel_cov, np.zeros((n_tot+2, 2))])
                bel_cov[-2:, -2:] = 1e6*np.eye(2)

                # Adiciona landmark
                landmarks = np.append(landmarks, [[mx], [my]])

            elif M <= thresh:
                # Atualiza posição com a amostra observada
                bel_mean[pos] = mx
                bel_mean[pos+1] = my

                # Calcula o ganho do filtro de Kalman
                K = bel_cov.dot(H.T).dot(S)

                # Atualiza vetor de estados e matriz de covariância
                bel_mean = bel_mean + (K.dot(z_dif))
                bel_cov = (np.eye(n_tot)-K.dot(H)).dot(bel_cov)

        self.bel_mean = bel_mean
        self.bel_cov = bel_cov
        self.hist_landmarks = landmarks  #incluido
        mu = self.bel_mean
        print(
            'Updated location\t x: {0:.2} \t y: {1:.2} \t theta: {2:.2}'
            .format(mu[0], mu[1], mu[2]))

    # Atualiza histórico de poses
    def _update_hist_pose(self, pose):
        if self.hist_poses is None:
            self.hist_poses = np.array([pose])
        else:
            self.hist_poses = np.vstack([self.hist_poses, pose])

    # Atualiza histórico de landmarks
    def _update_hist_landmarks(self, measurement):
        n = len(measurement)
        observations = np.hstack([
            measurement[np.arange(0, n-1, 2)].reshape(-1, 1),
            measurement[np.arange(1, n, 2)].reshape(-1, 1)]
        )

        # landmarks = self.hist_landmarks
        """if landmarks is None:
            landmarks = observations
        else:
            dists = distance.cdist(observations, landmarks, metric='euclidean')
            new_ids_min = np.all(dists > self.dist_thresh_min, axis=1)
            new_ids_max = np.any(dists < self.dist_thresh_max, axis=1)
            new_ids = np.logical_and(new_ids_min, new_ids_max)
            if True in new_ids:
                landmarks = np.append(
                    landmarks, observations[new_ids, :], axis=0)"""

        # self.hist_landmarks = landmarks
        self.hist_landmarks = observations

    # Plota mapa gerado
    def _plot_map(self):
        plt.subplot(131, aspect='equal')
        plt.clf()

        fov = self.laser.rad_angle
        mapsize = self.mapsize
        x = self.hist_poses
        l_points = self.hist_landmarks

        # Inverte eixos para melhor comparação com Gazebo
        x_id, y_id = 1, 0

        def state_arrow(state, size=0.1):
            x = state[x_id]
            y = state[y_id]
            dx = size * np.cos(state[2])
            dy = size * np.sin(state[2])
            dx, dy = dy, dx  # inverte para exibir
            return x, y, dx, dy

        if self.hist_poses is not None:
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

        # Atualiza tamanho do mapa
        self.mapsize = mapsize

    # Atualiza estados
    def update(self):
        if not self.belief:
            self._belief_init(num_landmarks=len(self.laser_scan))
            self.belief = True

        if self.odometry is not None:
            self._prediction_step(self.odometry, self.velocity)
            self._correction_step(self.laser_scan)
            self._update_hist_pose(pose=self.bel_mean[:3])
            # self._update_hist_landmarks(measurement=self.bel_mean[3:])

        if self.plot:
            self._plot_map()
