# usr/bin/bash!

# rosdep update
# rospack profile
source /opt/ros/melodic/setup.bash
source devel/setup.bash

chmod +x $(rospack find robosaut_t2)/nodes/*.py

rqt_graph &
roslaunch robosaut_t2 demo.launch &
wait
