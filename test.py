import ur5_rl
import gymnasium as gym
from stable_baselines3 import SAC, TD3
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
import time


# Define GUI elements
def user_interface():

    x_gui = p.addUserDebugParameter('X', -1, 1, 0.0)
    y_gui = p.addUserDebugParameter('Y', -1, 1, 0.0)
    z_gui = p.addUserDebugParameter('Z', -1, 1, 0.0)
    roll_gui = p.addUserDebugParameter('Roll', -1, 1, 0)
    pitch_gui = p.addUserDebugParameter('Pitch', -1, 1, 0)
    yaw_gui = p.addUserDebugParameter('Yaw', -1, 1, 0)
    gripper_gui = p.addUserDebugParameter('Gripper', -1, 1, 0)

    return [x_gui, y_gui, z_gui, roll_gui, pitch_gui, yaw_gui]

# Read GUI elements
def read_gui(gui_joints):
    j1 = p.readUserDebugParameter(gui_joints[0])
    j2 = p.readUserDebugParameter(gui_joints[1])
    j3 = p.readUserDebugParameter(gui_joints[2])
    j4 = p.readUserDebugParameter(gui_joints[3])
    j5 = p.readUserDebugParameter(gui_joints[4])
    j6 = p.readUserDebugParameter(gui_joints[5])
    # g =  p.readUserDebugParameter(gui_joints[6])

    return np.array([j1, j2, j3, j4, j5, j6])

COMPLETE = True

if __name__ == "__main__":
    

    # Test
    print("|| Loading model for testing ...")
    model = SAC.load("./5.2_aux/rl_model_33000_steps.zip")       # rl_31500_5.5 --> 8 / 10
                                                                 # rl_30000_5.5 --> 8o9 / 10
                                                                 # rl_28500_5.5 --> 8o9 / 10
                                                                 # rl_12500_5.5 --> 7o8 / 10
                                                                 # rl_33000_5.2 --> 7o8 / 10

    model.policy.eval()
    print("|| Testing ...")

    

    r = 0
    vec_env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")
    obs, info = vec_env.reset()
    
    gui_joints = user_interface()

    list_actions = []
    

    t = time.time()
    while True:
        # obs["ee_position"] = np.append(obs["ee_position"], 0)
        action, _states = model.predict(obs, deterministic = True)
        # action = read_gui(gui_joints)

        list_actions.append(action)
        obs, reward, terminated, truncated, info = vec_env.step(action)
        
        

        print(reward)
        print("--")
        r += reward
        img = vec_env.render()

        cv.imshow("AA", img)
        cv.waitKey(1)

        if terminated or truncated:

            if COMPLETE:

                grasped = False

                while len(list_actions) > 0:
                    if not grasped:
                        obs, reward, terminated, truncated, info = vec_env.step(np.zeros(6))
                        vec_env.grasping()

                    else:
                        obs, reward, terminated, truncated, info = vec_env.step(list_actions.pop())

                    img = vec_env.render()

                    cv.imshow("AA", img)
                    cv.waitKey(1)





            print("Tiempo: ", time.time() - t)
            t = time.time()
            print(r, "--")
            r = 0
            list_actions = []

            

            obs, info = vec_env.reset()

