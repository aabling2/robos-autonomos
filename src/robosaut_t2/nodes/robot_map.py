#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from matplotlib.patches import Ellipse


class Mapping():
    def __init__(self, fov=180, mapsize=10, plot=False, thresh_dist=5):
        self.fov = fov*(np.pi/180)
        self.mapsize = mapsize
        self.plot = plot
        self.thresh_dist = thresh_dist
        self.checkpoints = np.zeros((1, 3), dtype=np.float32)
        self.check_steps = 20

        self.poses = np.zeros((1, 3), dtype=np.float32)
        self.range_bearings = None
        self.landmarks = None

    def _observation_model(self, pose, range_bearings):
        rx, ry, ra = pose
        ranges, bearings = range_bearings[:, 0], range_bearings[:, 1]
        mx = rx + ranges * np.cos(bearings + ra)
        my = ry + ranges * np.sin(bearings + ra)
        return np.hstack([mx, my]).reshape((-1, 2))

    def _update_pose(self, pose):
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

    def update(self, pose, range_bearings):
        self._update_pose(pose)
        self._update_landmarks(range_bearings)

        if self.plot:
            self.plot_map()

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

        # plot current robot state covariance
        # plt.scatter(x[-1, 0], x[-1, 1], marker='o', s=12, color=(0, 0, 1))
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
            # a.arrow(*stateToArrow(pose), head_width=0.2, color=(0, 1, 0))
            plt.plot(
                [x[i-1, 0], x[i, 0]],
                [x[i-1, 1], x[i, 1]],
                color="c")

        # plot all landmarks ever observed
        for lm in self.landmarks:
            plt.scatter(lm[0], lm[1], marker='s', s=12, color=(0, 0, 0))

        # plot settings
        plt.xlim([-mapsize/2, mapsize/2])
        plt.ylim([-mapsize/2, mapsize/2])
        plt.title('Mapeamento da area')
        plt.pause(0.1)
        # plt.show()

    def detect_end_trajectory(self):
        if self.check_steps == 0:
            self.checkpoints = np.append(self.checkpoints, self.poses[-1, :])
            self.check_steps = 20
        else:
            self.check_steps -= 1

        #-------

    def calc_area(self):
        print("Filtrando pontos")
        print("Calculando area")


if __name__ == "__main__":
    mapping = Mapping(fov=270, mapsize=50, plot=True, thresh_dist=3)

    pose = np.array([.0, .0, .0])
    mapping.update(pose, range_bearings=np.array([[16, np.pi*0.25]]))
    mapping.update(pose, range_bearings=np.array([[17, np.pi*0.25]]))
    mapping.update(pose, range_bearings=np.array([[19, np.pi*0.24], [25, np.pi*0.5]]))
