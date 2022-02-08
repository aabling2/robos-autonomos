#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class Mapping():
    def __init__(self, fov=270):
        self.fov = fov*(np.pi/180)
        self.poses = np.zeros((3, 1), dtype=np.float32)
        self.range_bearings = None
        self.landmarks = []

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

    def update(self, )