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

# Install pip
RUN apt-get update
RUN apt-get install -y python3-pip

# Pytorch with CPU
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Pytorch with CUDA --> CUDA Pytorch Version: 12.1. // CUDA Nvidia Version: 12.2
#                   --> NVIDIA-SMI 535.129.03 // Driver Version: 535.129.03 // NVIDIA GeForce RTX 3060
RUN pip3 install torch torchvision

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

# Install IKPY
RUN pip3 install ikpy

# GPU configuration
RUN export CUDA_VISIBLE_DEVICES=[0]
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES all 




WORKDIR /daniel/Desktop/ur5-rl