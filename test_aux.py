import ur5_rl
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from networks_SB import CustomCombinedExtractor
import numpy as np
import cv2 as cv
import os


TEST = True
env_id = "ur5_rl/Ur5Env-v0"
n_training_envs = 1
n_eval_envs = 1



class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, **kwargs):
        super(CustomEvalCallback, self).__init__(eval_env, **kwargs)
        self.best_mean_reward = -float('inf')

    def on_step(self) -> bool:

        # Call the parent on_step method to handle the default behavior
        continue_training = super().on_step()

        # Custom logic to save the best model based on mean rewards
        if self.best_mean_reward < self.best_mean_reward:
            self.best_mean_reward = self.best_mean_reward
            # Save the best model
            self.model.save(self.best_model_save_path)
        
        if self.n_calls % self.eval_freq == 0:
            self.eval_env.close()
            aux_env = make_vec_env(env_id, n_envs = n_eval_envs, seed=0, env_kwargs={"render_mode": "DIRECT", "show": True})
            self.eval_env = VecNormalize(aux_env, norm_obs=True, norm_reward=True)
            self.training_env.reset()


        return continue_training




if __name__ == "__main__":
    print("|| Compiling ...")
    env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")
    
    
    # # Create the vectorized environment
    # vec_env  = make_vec_env(env_id, n_envs=n_training_envs, seed=0, env_kwargs={"render_mode": "DIRECT", "show": False})
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0, env_kwargs={"render_mode": "DIRECT", "show": True})
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)


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

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))

    # eval_log_dir = "my_models_eval/"
    # eval_callback = CustomEvalCallback(eval_env, best_model_save_path=eval_log_dir,
    #                               log_path=eval_log_dir, eval_freq=max(20 // n_training_envs, 1),
    #                               n_eval_episodes=1, deterministic=True,
    #                               render=False)
    

    # # Use your custom feature extractor in the policy_kwargs
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

    model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, 
                verbose=100, buffer_size = 10000,  batch_size = 128, tensorboard_log="logs/", 
                train_freq=10, learning_rate = 0.001, gamma = 0.99, seed = 42,
                use_sde = False, sde_sample_freq = 8, action_noise = action_noise)         # See logs: tensorboard --logdir logs/


    # if not TEST:
    #     model.learn(total_timesteps=300, log_interval=5, tb_log_name= "Test", callback = eval_callback, progress_bar = True)
    #     model.save("./models/sac_ur5_stage_teset")
    # else:
    #     model = SAC.load("./models/sac_ur5_stage_1.2")
    
    
    model.policy.eval()
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)



    # env = model.get_env()
    print("... Testing ...")
    env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT", show = True)
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # env.render()

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
    
