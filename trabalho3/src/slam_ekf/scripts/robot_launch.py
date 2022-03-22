#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import rospy
import numpy as np
from ekf_mapping import Mapping as EKF_map
from simple_mapping import Mapping as Simple_map

# Amostras de distâncias obtidas pelo LaserScan
from sensor_msgs.msg import LaserScan

METHOD = os.environ.get('MAP_METHOD')
if METHOD == '':
    METHOD = 'simple'
print("method", METHOD)


def main():

    # Cria node de controlador e mapeamento do robô
    rospy.wait_for_service('gazebo/set_physics_properties')
    rospy.init_node('ekf_slam', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    # Coordenadas de representação do mapa 0
    offset = np.array([-1.5, -1.6])
    draw_map = offset + np.array([
        [0., 0.],
        [7.7, 0.],
        [7.7, 7.7],
        [4.85, 7.7],
        [4.85, 9.7],
        [2.85, 9.7],
        [2.85, 7.7],
        [0., 7.7],
        [0., 0.]])

    if METHOD == 'simple':
        mapping = Simple_map(
            dist_thresh=0.2, laser_samples=25,
            map_size=10, offset=[2.5, 3.2], draw_map=draw_map)
    elif METHOD == 'ekf':
        mapping = EKF_map(
            dist_thresh=1.7, laser_samples=25,
            map_size=10, offset=[2.5, 3.2], draw_map=draw_map, plot_cov=False)

    # Espera tópico do laser abrir
    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('/front/scan', LaserScan, timeout=2)
        except Exception:
            pass

    # Mantém nodes enquanto não finalizar
    while not rospy.is_shutdown():

        # Atualiza mapeamento do ambiente
        mapping.update()

        try:
            rate.sleep()
        except Exception:
            break

    # Deleta objetos da memória
    del mapping

    # Aguarda finalizar o processo
    rospy.spin()


if __name__ == "__main__":
    main()
