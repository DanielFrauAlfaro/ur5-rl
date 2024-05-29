FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y apt-utils curl wget git bash-completion build-essential sudo && rm -rf /var/lib/apt/lists/*

# Now create the same user as the host itself
ARG UID=1000
ARG GID=1000
RUN addgroup --gid ${GID} ur5_rl_docker
RUN adduser --gecos "ROS User" --disabled-password --uid ${UID} --gid ${GID} ur5_rl_docker
RUN usermod -a -G dialout ur5_rl_docker
RUN mkdir config && echo "ros ALL=(ALL) NOPASSWD: ALL" > config/99_aptget
RUN cp config/99_aptget /etc/sudoers.d/99_aptget
RUN chmod 0440 /etc/sudoers.d/99_aptget && chown root:root /etc/sudoers.d/99_aptget

# Install pip
RUN apt-get update
RUN apt-get install -y python3-pip

# Pytorch with CPU
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Pytorch with CUDA 
RUN pip3 install torch

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

# Customization
RUN apt-get update
RUN apt-get install -y ncurses-term

# GPU configuration
RUN export CUDA_VISIBLE_DEVICES=[0]
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES all 

WORKDIR /ur5_rl_docker/Desktop/ur5-rl/

# LANZAMIENTO: sudo docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm -it --name docker_rl --net host --cpuset-cpus="0-11" -v ~/:/ur5_rl_docker -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev:/dev --runtime=nvidia --user=$(id -u $USER):$(id -g $USER) --pid=host --privileged docker_rl


