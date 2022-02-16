#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from robot_controller import Controller
from robot_mapping import Mapping
from map_benchmark import Benchmark

# Amostras de distâncias obtidas pelo LaserScan
from sensor_msgs.msg import LaserScan


def main():

    # Cria node de controlador e mapeamento do robô
    rospy.wait_for_service('gazebo/set_physics_properties')
    rospy.init_node('slam_area', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    benchmark = Benchmark()
    controller = Controller()
    mapping = Mapping(
        plot=True, dist_thresh_min=0.3, dist_thresh_max=0.4,
        dist_trace_stop=0.5, dist_map_stop=2.,
        steps_checkpoint=40, laser_samples=35)

    # Espera tópico do laser abrir
    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('/front/scan', LaserScan, timeout=2)
        except Exception:
            pass

    # Mantém nodes enquanto não finalizar
    while not rospy.is_shutdown() and not controller.closed:

        # Atualiza controles do robô
        controller.follow_wall()

        # Atualiza mapeamento do ambiente
        mapping.update()

        # Finaliza controle do robô se ponto final detectado
        if mapping.endpoint:
            controller.finish = True

        try:
            rate.sleep()
        except Exception:
            break

    # Calcula area do mapa gerado
    mapping.calc_area()

    # Calcula taxa de acerto, precisão, erro de mapeamento
    benchmark.run(
        points_true=np.float32(
            [[2.18, 2.05], [2.18, -1.15], [-1.55, -1.15], [-1.55, 2.05]]),
        points_pred=mapping.hull_points)

    # Deleta objetos da memória
    del controller
    del mapping

    # Aguarda finalizar o processo
    rospy.spin()


if __name__ == "__main__":
    main()
