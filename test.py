import ur5_rl
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from networks_SB import CustomCombinedExtractor
import numpy as np
import cv2 as cv


TEST = False

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode="DIRECT")
        env.reset(seed=seed + rank)
        return env

    return _init



if __name__ == "__main__":
    print("|| Compiling ...")
    env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")
    




    env_id = "ur5_rl/Ur5Env-v0"
    num_cpu = 3  # Number of processes to use
    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])




    q_space = env.observation_space["ee_position"]
    image_space = env.observation_space["image"]

    q_shape = q_space.shape
    in_channels, frame_w, frame_h = image_space.shape
    
    residual = False
    channels = [in_channels, 16, 32]
    kernel = 3          
    m_kernel = 3
    n_layers = len(channels) - 1

    out_vector_features = 70
    features_dim = 128          

    # Use your custom feature extractor in the policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim = features_dim,
                                       residual = residual, 
                                       channels = channels, kernel = kernel, m_kernel = m_kernel,
                                       n_layers = n_layers, out_vector_features = out_vector_features),
        net_arch=dict(
            pi=[features_dim, 16],  # Adjust the size of these layers based on your requirements
            vf=[features_dim, 16],  # Adjust the size of these layers based on your requirements
            qf=[features_dim, 16]),
        share_features_extractor = True
    )

    model = SAC("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, 
                verbose=100, buffer_size = 10000,  batch_size = 128, tensorboard_log="logs/", train_freq=10,
                learning_rate = 0.001, gamma = 0.99, seed = 42,
                use_sde = False, sde_sample_freq = 8)         # See logs: tensorboard --logdir logs/
    

    if not TEST:
        model.learn(total_timesteps=6300, log_interval=5, tb_log_name= "Test", progress_bar = True)
        model.save("./models/sac_ur5_stage_teset")
    else:
        model = SAC.load("./models/sac_ur5_stage_1.2")
    
    
    model.policy.eval()
    print("... Testing ...")
    
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
            

    
    

    # print("|| Sampling ...")
    # for j in range(100):
    #     print("--- Epoch ", j)
    #     obs, info = env.reset(seed=0, options={})

    #     with warnings.catch_warnings(record = True) as w:
    #         while True:

    #             # TODO: función del agente
    #             action = env.action_space.sample()

    #             env.set_warning(w)
    #             obs_, reward, done, truncated, info = env.step(action)
                
    #             env.render()
                
    #             if truncated or done:
    #                 break

    #             obs = obs_

    # print("|| Success")
    # env.close()
    

