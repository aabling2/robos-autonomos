# usr/bin/bash!
WORLD=$1
ROS_VERSION=melodic

source /opt/ros/$ROS_VERSION/setup.bash
source devel/setup.bash
export JACKAL_LASER=1

chmod +x $(rospack find slam_ekf)/scripts/*.py
roslaunch slam_ekf demo.launch world:=$WORLD &
wait

# roslaunch jackal_viz view_robot.launch config:=navigation
# roslaunch jackal_viz view_robot.launch
# rqt_graph