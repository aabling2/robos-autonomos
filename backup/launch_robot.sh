# usr/bin/bash!
#export $ROS_PACKAGE_PATH=$PWD
export JACKAL_LASER=1

# rosdep update
# rospack profile

source /opt/ros/melodic/setup.bash
source devel/setup.bash

# roslaunch jackal_gazebo jackal_world.launch
# roslaunch jackal_gazebo jackal_world.launch config:=front_laser
# roslaunch jackal_gazebo t2.launch config:=front_laser
# roslaunch jackal_viz view_robot.launch
# roslaunch robos_autonomos_t2 t2.launch

# gnome-terminal -- sh -c "roscore"
# roslaunch jackal_gazebo $(pwd)/src/robosaut_t2/jackal.launch config:=front_laser platform:=jackal
# roslaunch robosaut_t2 $(pwd)/src/robosaut_t2/jackal.launch config:=front_laser platform:=jackal

#gnome-terminal -- sh -c "roscore"
#sleep 2
# gnome-terminal -- sh -c "roslaunch jackal_gazebo robosaut_t2/world.launch config:=front_laser"
gnome-terminal -- sh -c "roslaunch robosaut_t2 jackal.launch platform:=jackal config:=front_laser; bash"

rqt_graph