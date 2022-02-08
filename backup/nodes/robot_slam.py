#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class EKFmapping():
    def __init__(self):
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

    def observation_model(self, pose, range_bearing):
        rx, ry, ra = pose
        range, bearing = range_bearing[0], range_bearing[1]
        mx = rx + range * np.cos(bearing + ra)
        my = ry + range * np.sin(bearing + ra)
        return [mx, my]

    def prediction_step(self, odometry):
        n = len(self.bel_mean)
        F = np.append(np.eye(3), np.zeros((3, n-3)), axis=1)
        Rt = self.Rt
        bel_mean = self.bel_mean
        bel_cov = self.bel_cov

        odometry = self.convert_odometry(self.bel_mean[:3], odometry)
        odometry = self.add_noise(odometry)
        drot1, dtrans, drot2 = odometry
        motion = self.odometry_model(self.bel_mean[:3], odometry)

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

    def correction_step(self, range_bearings):
        bel_mean = self.bel_mean
        bel_cov = self.bel_cov
        rx, ry, ra = bel_mean[:3]
        n_range_bearings = len(range_bearings)
        n_dim_state = len(bel_mean)
        # Qt = np.eye(2*n_range_bearings) * 0.01
        Qt = self.Qt

        # H = Jacobian matrix ∂ẑ/∂(rx,ry)
        H = np.zeros((2*n_range_bearings, n_dim_state), dtype=np.float32)
        zs, z_hat = [], []  # true and predicted observations
        for (j, range_bearing) in enumerate(range_bearings):
            pos = 3 + 2*j

            if bel_cov[2*j+3][2*j+3] >= 1e6 and bel_cov[2*j+4][2*j+4] >= 1e6:
                # Initialize its pose in mu based on the
                # measurement and the current robot pose
                mx, my = self.observation_model([rx, ry, ra], range_bearing)
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

    def plot_map(self, mu, x, mapsize, fov):
        a = plt.subplot(132, aspect='equal')
        a.cla()
        mu = np.array(mu).reshape((-1, len(mu)))
        print("test", mu)

        def stateToArrow(state):
            x = state[0]
            y = state[1]
            dx = 0.5*np.cos(state[2])
            dy = 0.5*np.sin(state[2])
            return x, y, dx, dy

        # plot current robot state covariance
        plt.scatter(mu[0, -1], mu[1, -1], marker='o', s=10, color=(1, 0, 0))
        plt.scatter(x[0], x[1], marker='o', s=10, color=(0, 0, 1))

        # plot current robot field of view
        # plot current robot field of view
        plt.plot(
            [x[0], x[0]+50*np.cos(x[2] + fov/2)],
            [x[1], x[1]+50*np.sin(x[2] + fov/2)],
            color="b")
        plt.plot(
            [x[0], x[0]+50*np.cos(x[2] - fov/2)],
            [x[1], x[1]+50*np.sin(x[2] - fov/2)],
            color="b")
        plt.plot(
            [mu[0, -1], mu[0, -1]+50*np.cos(mu[2, -1] + fov/2)],
            [mu[1, -1], mu[1, -1]+50*np.sin(mu[2, -1] + fov/2)],
            color="r")
        plt.plot(
            [mu[0, -1], mu[0, -1]+50*np.cos(mu[2, -1] - fov/2)],
            [mu[1, -1], mu[1, -1]+50*np.sin(mu[2, -1] - fov/2)],
            color="r")

        # plot robot state history
        for i in range(mu.shape[1]-1):
            a.arrow(*stateToArrow(mu[:3, i]), head_width=0.2, color=(0, 1, 0))

        # plot all landmarks ever observed
        n = int((len(mu)-3)/2)
        for i in range(n):
            # if cov[2*i+3, 2*i+3] < 1e6:
            zx = mu[2*i+3, -1]
            zy = mu[2*i+4, -1]
            plt.scatter(zx, zy, marker='s', s=12, color=(0, 0, 0))

        # plot settings
        plt.xlim([-mapsize/2, mapsize/2])
        plt.ylim([-mapsize/2, mapsize/2])
        plt.title('Observations and trajectory estimate')
        plt.pause(0.1)

    def update(self, odometry, range_bearings, fov=np.pi):
        if not self.belief:
            self.belief_init(num_landmarks=len(range_bearings))
            self.belief = True

        self.prediction_step(odometry)
        self.correction_step(range_bearings)

        self.hist_mu.append(self.bel_mean)

        self.plot_map(mu=self.hist_mu, x=odometry, mapsize=40, fov=fov)
