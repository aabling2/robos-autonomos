#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from robot_controller import Controller
from robot_mapping import Mapping

# Amostras de distâncias obtidas pelo LaserScan
from sensor_msgs.msg import LaserScan


def main():

    # Cria node do controlador do robô
    rospy.wait_for_service('gazebo/set_physics_properties')
    rospy.init_node('slam_area', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    controller = Controller()
    mapping = Mapping(
        mapsize=5, plot=True, thresh_dist=0.5,
        steps_checkpoint=40, laser_samples=20)

    # Espera tópico do laser abrir
    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('/front/scan', LaserScan, timeout=2)
        except Exception:
            pass

    while not rospy.is_shutdown() and not controller.closed:
        controller.follow_wall()
        mapping.update()

        if mapping.endpoint:
            controller.finish = True

        try:
            rate.sleep()

        except Exception:
            break

    # Calcula area do mapa gerado
    mapping.calc_area()

    # Aguarda finalizar o processo
    del controller
    del mapping

    rospy.spin()


if __name__ == "__main__":
    main()
