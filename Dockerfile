FROM osrf/ros:noetic-desktop-full

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full* \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y apt-utils curl wget git bash-completion build-essential sudo && rm -rf /var/lib/apt/lists/*

# Now create the same user as the host itself
ARG UID=1000
ARG GID=1000
RUN addgroup --gid ${GID} daniel
RUN adduser --gecos "ROS User" --disabled-password --uid ${UID} --gid ${GID} daniel
RUN usermod -a -G dialout daniel
RUN mkdir config && echo "ros ALL=(ALL) NOPASSWD: ALL" > config/99_aptget
RUN cp config/99_aptget /etc/sudoers.d/99_aptget
RUN chmod 0440 /etc/sudoers.d/99_aptget && chown root:root /etc/sudoers.d/99_aptget

# Change HOME environment variable
ENV HOME /home/daniel
RUN mkdir -p ${HOME}/ros_ws/src

# Install pip
RUN apt-get update
RUN apt-get install -y python3-pip

# Pytorch with CPU
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Pytorch with CUDA --> CUDA Pytorch Version: 12.1. // CUDA Nvidia Version: 12.2
#                   --> NVIDIA-SMI 535.129.03 // Driver Version: 535.129.03 // NVIDIA GeForce RTX 3060
RUN pip3 install torch

# Update bash to add ROS2 to the path
RUN echo "" >> ~/.bashrc
RUN echo "## ROS" >> ~/.bashrc
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc


# Install pybullet, gym and graphic interface
RUN pip3 install pybullet
RUN pip3 install pyb_utils  
RUN pip3 install gymnasium
RUN apt-get install -y libgl1-mesa-glx

# Install opencv-python
RUN apt-get install -y python3-opencv

# Install Robotic Toolbox
RUN pip3 install roboticstoolbox-python
RUN pip3 install pynput

# Update Numpy
RUN pip3 install numpy==1.24

# Install Stable Baselines 3
RUN pip3 install stable-baselines3[extra]
RUN pip3 install sb3-contrib

# DQ Robotics
RUN python3 -m pip install --user dqrobotics
RUN python3 -m pip install --user dqrobotics --upgrade

RUN apt-get update
RUN apt-get install -y ros-noetic-moveit-resources-panda-moveit-config ros-noetic-rosparam-shortcuts

RUN apt-get install -y ros-noetic-ros-control ros-noetic-ros-controllers ros-noetic-controller-manager  ros-noetic-position-controllers  ros-noetic-joint-state-controller  ros-noetic-joint-trajectory-controller  ros-noetic-joint-limits-interface  ros-noetic-transmission-interface
RUN apt-get install -y ros-noetic-rviz-visual-tools
RUN apt-get install -y ros-noetic-gazebo-ros-pkgs

RUN apt install -y ros-noetic-catkin python3-catkin-tools python3-wstool
RUN apt-get install -y libpopt-dev ros-noetic-libpcan
RUN apt-get install -y ros-noetic-socketcan-interface
RUN apt-get install -y ros-noetic-ddynamic-reconfigure
RUN apt-get install -y ros-noetic-realsense2-camera && apt-get install -y ros-noetic-realsense2-description
RUN apt-get install -y ros-noetic-industrial-robot-status-interface
RUN apt-get install -y ros-noetic-scaled-joint-trajectory-controller
RUN apt-get install -y ros-noetic-ruckig
RUN apt-get install -y ros-noetic-pybind11-catkin
RUN apt-get install -y ros-noetic-geometric-shapes
RUN apt-get install -y ros-noetic-moveit-msgs
RUN apt-get install -y ros-noetic-srdfdom
RUN apt-get install -y ros-noetic-ompl
RUN apt-get install -y ros-noetic-warehouse-ros
RUN apt-get install -y ros-noetic-eigenpy
RUN apt-get install -y ros-noetic-speed-scaling-interface
RUN apt install -y python3-tk
RUN apt-get install -y ros-noetic-moveit-ros-planning-interface
RUN apt-get install -y ros-noetic-speed-scaling-state-controller
RUN apt-get install -y ros-noetic-ur-msgs
RUN apt-get install -y ros-noetic-pass-through-controllers
RUN apt-get install -y ros-noetic-ur-client-library
RUN apt-get install -y ros-noetic-gazebo-ros-control

# Customization
RUN apt-get update
RUN apt-get install -y ncurses-term


# GPU configuration
RUN export CUDA_VISIBLE_DEVICES=[0]
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES all 

#
RUN echo 'export PS1="\[\033[1;33m\]\u@\h\[\033[0m\]:\[\033[1;35m\]\w\[\033[0m\]\$ "' >> ~/.bashrc
# RUN source ~/.bashrc
WORKDIR /daniel/Desktop/ur5-rl/ros_ws



# RUN rm /usr/local/lib/python3.10/dist-packages/roboticstoolbox/mobile/EKF.py
# COPY EKF.py /usr/local/lib/python3.10/dist-packages/roboticstoolbox/mobile


# LANZAMIENTO: sudo docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm -it --name docker_rl --net host --cpuset-cpus="0-11" -v ~/:/daniel -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev:/dev --runtime=nvidia --user=$(id -u $USER):$(id -g $USER) --pid=host --privileged docker_rl


