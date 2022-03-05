# usr/bin/bash!

# Robot simulation - Gazebo
sudo apt-get install ros-melodic-gazebo-ros-pkgs ros-melodic-gazebo-ros-control

# Clearpath Robotics - Jackal Robot
sudo apt-get install ros-melodic-jackal-simulator ros-melodic-jackal-desktop ros-melodic-jackal-navigation

# Path generator
ROOT_DIR=$(dirname $(dirname $(realpath ${BASH_SOURCE})))
sudo git clone https://github.com/ricardocmello/path_generator.git $ROOT_DIR/src/path_generator/
sudo chown -Rc $USER $ROOT_DIR/src/path_generator
