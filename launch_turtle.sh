# usr/bin/bash!
# rosdep update
source /opt/ros/melodic/setup.bash
gnome-terminal -- sh -c "roscore"
sleep 2
gnome-terminal -- sh -c "rosrun turtlesim turtlesim_node"
gnome-terminal -- sh -c "rosrun turtlesim turtle_teleop_key"
rqt_graph