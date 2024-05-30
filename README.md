# DRL for Manipulation Using a UR5e and Dual Quaternions

This repository contains code for robotic manipulation using a UR5e robot and deep reinforcement learning (DRL) with dual quaternions. Dual quaternions provide a compact and efficient representation for the position and orientation of objects in 3D space, making them suitable for robotic manipulation tasks and distance computation for reward estimation.

![intro](https://github.com/DanielFrauAlfaro/ur5-rl/assets/98766327/2e3998f9-636a-4b82-b99c-48a1b9cbff76)


## Setup and Installation

This projected can be executed using the provided Dockerfile. This way it is ensured that every user can run this code without errors or problems.

### Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/DanielFrauAlfaro/ur5-rl
    cd ur5-rl/
    ```

2. Build the Docker image:

    ```bash
    sudo docker build -t docker_ur5e_rl .
    ```

3. Launch Docker image:

    ```bash
    sudo docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm -it --name docker_ur5e_rl --net host --cpuset-cpus="0-11" -v ~/:/ur5-rl -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev:/dev --user=$(id -u $USER):$(id -g $USER) --pid=host --privileged docker_ur5e_rl
    ```

## Training

To train the deep reinforcement learning model for robotic manipulation:

    ```bash
    sudo docker build -t docker_ur5e_rl .
    ```

Training logs and checkpoints will be saved in the `logs/`. You can use a visualization tool like TensorBoard to monitor the training progress:

    ```bash
    tensorboard --logdir logs/
    ```

## Evaluation

To evaluate the trained model there are two ways of doing it; one that generates metrics of all models saved during training and another one that allows to see how the agent performs in the environment.

    ```bash
    python3 test.py             # See the agent
    python3 loop_test.py        # Metrics over all agents
    ```

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please open an issue or create a pull request.
