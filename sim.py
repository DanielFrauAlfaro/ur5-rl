import yaml
import ur5_rl
import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import VecNormalize, VecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from networks_SB import CustomCombinedExtractor
import numpy as np
import cv2 as cv
import os
import torch

class Env():
    def __init__(self, data = {}):
        self.__env_ids = {"PyBullet": "ur5_rl/Ur5Env-v0"}
        self.__data = self.load_data(data = data)

        

        self.__env = self.set_env()

    def get_env(self):
        return self.__env
    
    def set_env(self):
        return make_vec_env(self.__env_ids[self.__data["sim"]], n_envs=self.__data["n_envs"], 
                                  vec_env_cls = SubprocVecEnv, seed=0, 
                                  env_kwargs={"render_mode": self.__data["render_mode"], "data": self.__data})
    
    def load_data(self, data = {}):
        with open(data["sim_config"], 'r') as s:
            data.update(yaml.load(s, Loader=yaml.SafeLoader))

        return data
    