#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


# Calcula área pela formula shoelace
def shoelace_area(points):
    lines = np.hstack([points, np.roll(points, -1, axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1, y1, x2, y2 in lines))
    return round(area, 2)


# Forma polígono pelos pontos encontrados
def convex_hull(points):
    hull_points = []

    # Pega ponto mais a esquerda
    start_point = points[np.argmin(points[:, 0], axis=0)]
    point = start_point
    hull_points.append(start_point)
    far_point = None

    # Busca pontos mais distantes não adicionados ainda
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
