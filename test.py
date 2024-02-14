import ur5_rl
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, VecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from networks_SB import CustomCombinedExtractor
import numpy as np
import cv2 as cv
import os
import pybullet as p


# Define GUI elements
def user_interface():

    x_gui = p.addUserDebugParameter('X', -1, 1, 0.0)
    y_gui = p.addUserDebugParameter('Y', -1, 1, 0.0)
    z_gui = p.addUserDebugParameter('Z', -1, 1, 0.0)
    roll_gui = p.addUserDebugParameter('Roll', -1, 1, 0)
    pitch_gui = p.addUserDebugParameter('Pitch', -1, 1, 0)
    yaw_gui = p.addUserDebugParameter('Yaw', -1, 1, 0)
    gripper_gui = p.addUserDebugParameter('Gripper', -1, 1, 0)

    return [x_gui, y_gui, z_gui, roll_gui, pitch_gui, yaw_gui, gripper_gui]


# Read GUI elements
def read_gui(gui_joints):
    j1 = p.readUserDebugParameter(gui_joints[0])
    j2 = p.readUserDebugParameter(gui_joints[1])
    j3 = p.readUserDebugParameter(gui_joints[2])
    j4 = p.readUserDebugParameter(gui_joints[3])
    j5 = p.readUserDebugParameter(gui_joints[4])
    j6 = p.readUserDebugParameter(gui_joints[5])
    g =  p.readUserDebugParameter(gui_joints[6])

    return np.array([j1, j2, j3, j4, j5, j6, g])

if __name__ == "__main__":
    

    # Test
    # print("|| Loading model for testing ...")
    # model = SAC.load("./my_models_eval/rl_model_37500_steps.zip")
    
    # model.policy.eval()
    # print("|| Testing ...")

    

    r = 0
    vec_env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")
    obs, info = vec_env.reset()
    
    gui_joints = user_interface()

    while True:
        # action, _states = model.predict(obs, deterministic = True)
        # action = read_gui(gui_joints)

        obs, reward, terminated, truncated, info = vec_env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        
        # print(reward)
        r += reward
        img = vec_env.render()

        cv.imshow("AA", img)
        cv.waitKey(1)

        if terminated or truncated:
            print(r, "--")
            r = 0
            obs, info = vec_env.reset()

