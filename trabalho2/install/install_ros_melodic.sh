# usr/bin/bash!

# remove previous installations of ROS
# sudo apt-get remove ros-*
# sudo apt-get remove ros-melodic-*
# sudo apt-get autoremove

# Configure your Ubuntu repositories
# Configure your Ubuntu repositories to allow "restricted," "universe," and "multiverse." You can follow the Ubuntu guide for instructions on doing this.
# Setup your sources.list
# Setup your computer to accept software from packages.ros.org.
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update

# Desktop-Full Install: (Recommended) : ROS, rqt, rviz, robot-generic libraries, 2D/3D simulators and 2D/3D perception
sudo apt install ros-melodic-desktop-full

# Desktop Install: ROS, rqt, rviz, and robot-generic libraries
# sudo apt install ros-melodic-desktop

# ROS-Base: (Bare Bones) ROS package, build, and communication libraries. No GUI tools.
# sudo apt install ros-melodic-ros-base

# Individual Package: You can also install a specific ROS package (replace underscores with dashes of the package name):
# sudo apt install ros-melodic-PACKAGE
# sudo apt install ros-melodic-slam-gmapping

# To find available packages, use:
# apt search ros-melodic

# Environment setup
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
source /opt/ros/melodic/setup.bash
export ROS_PYTHON_VERSION=3

# Dependencies for building packages
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential

# With the following, you can initialize rosdep.
sudo rosdep init
rosdep update