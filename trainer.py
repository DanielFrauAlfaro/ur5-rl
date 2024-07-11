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
import torch
import yaml

class Trainer():
    
    def __init__(self, vec_env, data = {}):
        self.__data_agent, self.__data_model = self.load_data(data = data)
        self.__data = data
        self.__vec_env = vec_env.get_env()

        self.__agent = self.set_agent(data = data, data_model = self.__data_model,
                                      data_agent = self.__data_agent, vec_env = self.__vec_env)
        
        self.__callbacks = self.set_callbacks(data_agent = self.__data_agent)



    def train(self):
        # Training 

        total_timesteps = self.__data_agent["train_timesteps"]
        log_interval = self.__data_agent["log_interval"]
        checkpoint_log_dir = self.__data_agent["save_dir"]
        logs_name = self.__data_agent["logs_name"]
        pretrained_model = self.__data_agent["pretrained_model"]

        if pretrained_model != "":
            self.__agent.set_parameters(pretrained_model)

        self.__agent.learn(total_timesteps = total_timesteps, log_interval = log_interval, 
                           tb_log_name = logs_name, callback = self.__callbacks, progress_bar = True)
        self.__agent.save(checkpoint_log_dir + "best_model.zip")


    def test(self):
        pass



    def load_data(self, data = {}):
        agent, model = None, None

        with open(data["agent_config"], "r") as agent, open(data["model_config"], "r") as model:
            agent = yaml.load(agent, Loader=yaml.SafeLoader)
            model = yaml.load(model, Loader=yaml.SafeLoader)

        return agent, model
    
    def set_agent(self, data, data_model, data_agent, vec_env):

        # --- Arquitecture ---    
        
        residual = data_agent["residual"]
        channels = data_agent["channels"]
        channels.insert(0, len(data["cam_mode"]))

        kernel = data_agent["kernel"]         
        pool_kernel = data_agent["pool_kernel"]
        n_layers = len(channels) - 1

        out_vector_features = data_agent["vector_features"]
        features_dim = data_agent["features_dim"] 
        mapping_arch = data_agent["mapping_arch"]
        share_feature_extractor = data_agent["share_feature_extractor"]

        extractor_class = None

        if data["observations"] == ["ee_position", "image"]:
            extractor_class = CustomCombinedExtractor

        else:
            print("--- ERROR: Observation of the type", {data["observations"]} ,"is not supported.")
            raise

        
        # Use custom feature extractor in the policy_kwargs
        policy_kwargs = dict(
            features_extractor_class=extractor_class,
            features_extractor_kwargs=dict(features_dim = features_dim,
                                        residual = residual, 
                                        channels = channels, kernel = kernel, m_kernel = pool_kernel,
                                        n_layers = n_layers, out_vector_features = out_vector_features),
            net_arch=dict(
                pi=mapping_arch,
                vf=mapping_arch,
                qf=mapping_arch),
            share_features_extractor = share_feature_extractor,
        )

        model = self.build_model(policy_kwargs = policy_kwargs, vec_env = vec_env, data_model = data_model, data_agent = data_agent)

        return model

    
    def build_model(self, policy_kwargs, vec_env, data_model, data_agent):
        model = None

        # Get params
        learning_starts = data_agent["learning_starts"]
        verbose = data_agent["verbose"]
        logs_dir = data_agent["logs_dir"]
        seed = data_agent["seed"]
        learning_rate = data_agent["learning_rate"]
        batch_size = data_agent["batch_size"]
        

        # Model declaration
        if data_model["model"] == "sac":
            buffer_size = data_model["buffer_size"]
            train_freq = data_model["train_freq"]

            model = SAC("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, learning_starts = learning_starts,
                        verbose = verbose, buffer_size = buffer_size, tensorboard_log = logs_dir, seed = seed, learning_rate = learning_rate,
                        train_freq = train_freq, batch_size = batch_size)         # See logs: tensorboard --logdir logs/

        else:
            print("--- ERROR: ", {data_model["model"]}, " models are not supported.")
            raise

        return model

    def set_callbacks(self, data_agent):
        # --- Callbacks ---
        save_freq = data_agent["save_freq"]
        n_training_envs = data_agent["n_training_envs"]
        checkpoint_log_dir = data_agent["save_dir"]
        save_replay_buffer = data_agent["save_buffer"]
        save_vecnormalize = data_agent["save_vec"]
        name_prefix = data_agent["name_prefix"]

        # eval_log_dir = "./models_eval/"
        # eval_callback = EvalCallback(eval_env, best_model_save_path="./models_eval/",
        #                          log_path="./logs/", eval_freq=max(save_freq // n_training_envs, 1),
        #                          deterministic=True, render=False)
        
        
        checkpoint_callback = CheckpointCallback(
            save_freq = max(save_freq // n_training_envs, 1),
            save_path = checkpoint_log_dir,
            name_prefix = name_prefix,
            save_replay_buffer = save_replay_buffer,
            save_vecnormalize = save_vecnormalize
        )

        return [checkpoint_callback]

    def get_model_data(self):
        return self.__data_model
    
    def set_env(self, env):
        self.__env = env

    def get_env(self):
        return self.__env