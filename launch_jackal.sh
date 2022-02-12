# usr/bin/bash!
WORLD=$1
ROS_VERSION=melodic

source /opt/ros/$ROS_VERSION/setup.bash
source devel/setup.bash
export JACKAL_LASER=1

chmod +x $(rospack find slam_area)/nodes/*.py
roslaunch slam_area demo.launch world:=$WORLD &
wait

# roslaunch jackal_viz view_robot.launch