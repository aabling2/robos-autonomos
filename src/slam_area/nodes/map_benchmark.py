#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from map_area import shoelace_area


class Benchmark():

    # IOU entre o poligono real do ambiente e o encontrado
    def _IOU(self, polygon1, polygon2):
        polygon1 = Polygon(polygon1)
        polygon2 = Polygon(polygon2)
        intersect = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        iou = intersect / union
        return iou

    # Erro percentual relativo entre valor real e encontrado
    def _relative_error(selfs, x_true, x_found):
        return abs(x_true - x_found) / x_true

    # Estima área pelos pontos e calcula metricas de desempenho
    def run(self, points_true, points_found):
        print("\nBenchmarking...\n")

        # Estima valores de área pelos pontos fornecidos
        area_true = shoelace_area(points_true)
        area_found = shoelace_area(points_found)

        # Métricas de desempenho
        err = self._relative_error(x_true=area_true, x_found=area_found)
        iou = self._IOU(polygon1=points_true, polygon2=points_found)
        err = round(err*100, 2)
        iou = round(iou*100, 2)

        # Resultados
        print("Area real: {}m2".format(area_true))
        print("Area encontrada: {}m2".format(area_found))
        print("Erro percentual relativo: {}%".format(err))
        print("IOU: {}%".format(iou))

        # Plot
        self.plot_area(points_true, points_found, iou, err)

    # Plot erro
    def plot_area(self, x_true, x_found, iou, err):
        x_true = np.vstack([x_true, x_true[0]])
        x_found = np.vstack([x_found, x_found[0]])

        plt.plot(x_true[:, 1], x_true[:, 0], color=(0, 1, 0), linewidth=4)
        plt.plot(x_found[:, 1], x_found[:, 0], color=(0, 0, 1), linewidth=2)

        mapsize = np.max(np.abs(np.vstack([x_true, x_found])))*2
        plt.xlim([1.1*mapsize/2, -1.1*mapsize/2])
        plt.ylim([-1.1*mapsize/2, 1.1*mapsize/2])
        plt.xlabel("Y-gazebo")
        plt.ylabel("X-gazebo")
        plt.title('Error {}%, IOU {}%'.format(err, iou))
        plt.legend(['true', 'found'])
        plt.show()


if __name__ == "__main__":
    bench = Benchmark(mapsize=15)
    bench.run(
        points_true=np.float32([[3, 3], [3, -3], [-3, -3], [-3, 3]]),
        points_found=np.float32([[3, 3], [3, -3], [-3, -3], [-3, 2]]))
