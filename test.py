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


if __name__ == "__main__":
    

    # Test
    print("|| Loading model for testing ...")
    model = SAC.load("./my_models_eval/rl_model_3000_steps.zip")
    
    model.policy.eval()
    print("|| Testing ...")

    r = 0
    vec_env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "GUI")
    obs, info = vec_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic = True)
        obs, reward, terminated, truncated, info = vec_env.step(action)
        
        # print(reward)
        r += reward
        img = vec_env.render()

        cv.imshow("AA", img)
        cv.waitKey(1)

        if terminated or truncated:
            print(r, "--")
            r = 0
            obs, info = vec_env.reset()

