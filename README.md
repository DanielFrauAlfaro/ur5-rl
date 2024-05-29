# DRL for Manipulation Using a UR5e and Dual Quaternions

This repository contains code for robotic manipulation using a UR5e robot and deep reinforcement learning (DRL) with dual quaternions. Dual quaternions provide a compact and efficient representation for the position and orientation of objects in 3D space, making them suitable for robotic manipulation tasks.

## Setup

### Prerequisites

Ensure you have the following software and libraries installed:

- Python 3.8+
- PyTorch
- NumPy
- Gym
- ROS (Robot Operating System)
- UR5e ROS drivers
- Additional Python libraries specified in `requirements.txt`

### Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/robotic_manipulation_drl.git
    cd robotic_manipulation_drl
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up the ROS environment:

    Follow the instructions for installing ROS and setting up the UR5e ROS drivers from the official [ROS documentation](http://wiki.ros.org/).

## Training

To train the deep reinforcement learning model for robotic manipulation:

1. Ensure your UR5e robot is properly set up and connected.
2. Launch the ROS nodes required for controlling the UR5e.
3. Run the training script:

    ```bash
    python train.py --config configs/train_config.yaml
    ```

    The training configuration is specified in the `configs/train_config.yaml` file. You can adjust the hyperparameters, learning rate, and other settings in this file.

4. Monitor the training process:

    Training logs and checkpoints will be saved in the `logs/` and `checkpoints/` directories, respectively. Use a visualization tool like TensorBoard to monitor the training progress:

    ```bash
    tensorboard --logdir logs/
    ```

## Evaluation

To evaluate the trained model:

1. Ensure your UR5e robot is properly set up and connected.
2. Launch the ROS nodes required for controlling the UR5e.
3. Run the evaluation script:

    ```bash
    python evaluate.py --config configs/eval_config.yaml --checkpoint checkpoints/best_model.pth
    ```

    The evaluation configuration is specified in the `configs/eval_config.yaml` file. Adjust the settings as needed.

4. Review the evaluation results:

    The evaluation script will output the performance metrics and visualizations of the robot's manipulation tasks. Check the `results/` directory for detailed evaluation reports.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please open an issue or create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
