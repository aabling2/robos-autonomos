# usr/bin/bash!

ROS_VERSION=melodic

# rosdep update
# rospack profile
source /opt/ros/$ROS_VERSION/setup.bash
source devel/setup.bash
export JACKAL_LASER=1

chmod +x $(rospack find robosaut_t2)/nodes/*.py
roslaunch robosaut_t2 demo.launch &
wait