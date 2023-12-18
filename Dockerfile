FROM osrf/ros:humble-desktop-full

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop-full* \
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

# set up environment
COPY ./update_bashrc /sbin/update_bashrc
RUN sudo chmod +x /sbin/update_bashrc ; sudo chown ros /sbin/update_bashrc ; sync ; /bin/bash -c /sbin/update_bashrc ; sudo rm /sbin/update_bashrc

# Install pip
RUN apt-get update
RUN apt-get install -y python3-pip

# Install pybullet, gym and graphic interface
RUN pip3 install pybullet
RUN pip3 install gymnasium
RUN apt-get install -y libgl1-mesa-glx

# Install opencv-python
RUN apt-get install -y python3-opencv

# Install Robotic Toolbox
RUN pip3 install roboticstoolbox-python
RUN pip3 install pynput

# Update Numpy
RUN pip3 install -U numpy

# GPU configuration
RUN export CUDA_VISIBLE_DEVICES=[0]
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES all 




WORKDIR /daniel/Desktop/ur5-rl