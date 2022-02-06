#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class EKFmapping():
    def __init__(self):
        self.bel_mean = None
        self.bel_cov = None

        self.Rt = np.array([
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0, 0, 0.01]])

        self.Qt = np.array([
            [0.01, 0],
            [0, 0.01]])

    def observation_model(self, pose, range_bearing):
        rx, ry, ra = pose
        range, bearing = range_bearing[0], range_bearing[1]
        mx = rx + range * np.cos(bearing + ra)
        my = ry + range * np.sin(bearing + ra)
        return [mx, my]

    def belief_init(self, num_landmarks):
        self.bel_mean = np.zeros(3+2*num_landmarks)
        self.bel_cov = 1e6*np.eye(3+2*num_landmarks)
        self.bel_cov[:3, :3] = .0

    def prediction_step(self, odometry):
        n = len(self.bel_mean)
        F = np.append(np.eye(3), np.zeros((3, n-3)), axis=1)
        Rt = self.Rt
        bel_mean = self.bel_mean
        bel_cov = self.bel_cov
        motion = odometry

        # Compute the new mu based on the noise-free
        # (odometry-based) motion model
        self.bel_mean = bel_mean + (F.T).dot(motion)

        # Define motion model Jacobian
        """
            [0, 0, -dtrans*np.sin(mu[2][0]+drot1)],
            [0, 0,  dtrans*np.cos(mu[2][0]+drot1)],
            [0, 0,  0]
        """
        J = np.zeros((3, 3))
        J[:, 2] = motion
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
        rx, ry, ra = bel_mean[0:3]
        n_range_bearings = len(range_bearings)
        n_dim_state = len(bel_mean)
        # Qt = np.eye(2*n_range_bearings) * 0.01
        Qt = self.Qt

        # H = Jacobian matrix ∂ẑ/∂(rx,ry)
        H = np.zeros((2*n_range_bearings, n_dim_state), dtype=np.float32)
        zs, z_hat = [], []  # true and predicted observations
        for (i, range_bearing) in enumerate(range_bearings):
            # mid = range_bearing.landmark_id
            mid = i

            # Initialize its pose in mu based on the
            # measurement and the current robot pose
            mx, my = self.observation_model([rx, ry, ra], range_bearing)
            pos = 3 + 2*mid
            bel_mean[pos] = mx
            bel_mean[pos+1] = my

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

    def update(self, odometry, range_bearings):
        self.belief_init(num_landmarks=len(range_bearings))
        self.prediction_step(odometry)
        self.correction_step(range_bearings)

        self.plot_measurement(
            self.bel_mean,
            self.bel_cov,
            range_bearings)

    def plot_measurement(self, mu, cov, obs):
        a = plt.subplot(132, aspect='equal')
        n = len(obs)
        for j, z in enumerate(obs):
            zx = mu[2*j+3]
            zy = mu[2*j+4]
            if j < n:
                plt.plot([mu[0], zx], [mu[1], zy], color=(0, 1, 1))
            else:
                plt.plot([mu[0], zx], [mu[1], zy], color=(0, 1, 0))

            landmark_cov = Ellipse(
                xy=[zx, zy],
                width=cov[2*j+3][2*j+3],
                height=cov[2*j+4][2*j+4],
                angle=0)

            landmark_cov.set_edgecolor((0, 0, 0))
            landmark_cov.set_fill(0)
            a.add_artist(landmark_cov)

        plt.pause(0.1)
