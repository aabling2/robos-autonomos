#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from ekf_mapping import Mapping

# Amostras de distâncias obtidas pelo LaserScan
from sensor_msgs.msg import LaserScan


def main():

    # Cria node de controlador e mapeamento do robô
    rospy.wait_for_service('gazebo/set_physics_properties')
    rospy.init_node('ekf_slam', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    mapping = Mapping(
        plot=True, dist_thresh_min=0.3, dist_thresh_max=0.4, laser_samples=10)

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
