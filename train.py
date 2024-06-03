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


env_id = "ur5_rl/Ur5Env-v0"
n_training_envs = 1
n_eval_envs = 0


# --- Main ---
if __name__ == "__main__":
    print("|| Compiling ...")
    
    # Training environments
    vec_env  = make_vec_env(env_id, n_envs=n_training_envs, vec_env_cls = SubprocVecEnv, seed=0, env_kwargs={"render_mode": "DIRECT", "show": False})

    # Evaluation environments
    eval_env = make_vec_env(env_id, n_envs=n_eval_envs, vec_env_cls = SubprocVecEnv, seed=0, env_kwargs={"render_mode": "DIRECT", "show": True})


    # Observation space
    q_space = vec_env.observation_space["ee_position"]
    image_space = vec_env.observation_space["image"]

    q_shape = q_space.shape
    in_channels, frame_w, frame_h = image_space.shape


    # --- Arquitecture ---    
    residual = True
    channels = [2, 16, 32, 32, 48]
    kernel = 3          
    m_kernel = 3
    n_layers = len(channels) - 1

    out_vector_features = 100
    features_dim = 256   
    
    # Use custom feature extractor in the policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim = features_dim,
                                       residual = residual, 
                                       channels = channels, kernel = kernel, m_kernel = m_kernel,
                                       n_layers = n_layers, out_vector_features = out_vector_features),
        net_arch=dict(
            pi=[features_dim, 64, 32],
            vf=[features_dim, 64, 32],
            qf=[features_dim, 64, 32]),
        share_features_extractor = True,
    )

    # --- Callbacks ---
    save_freq = 1500

    eval_log_dir = "./models_eval/"
    eval_callback = EvalCallback(eval_env, best_model_save_path="./models_eval/",
                             log_path="./logs/", eval_freq=max(save_freq // n_training_envs, 1),
                             deterministic=True, render=False)
    
    checkpoint_log_dir = "./models/"
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_training_envs, 1),
        save_path=checkpoint_log_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    
    # Model declaration
    model = SAC("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, learning_starts = 5000,
                verbose=100, buffer_size = 15000, tensorboard_log="logs/", seed = 42, learning_rate = 0.0003,
                train_freq=3)         # See logs: tensorboard --logdir logs/
    
    # Training 
    print("|| Training ...")
    model.learn(total_timesteps=100000, log_interval=5, tb_log_name= "Test", callback = [checkpoint_callback], progress_bar = True)
    model.save(checkpoint_log_dir + "best_model.zip")


    # Close enviroments
    vec_env.close()
    eval_env.close()
