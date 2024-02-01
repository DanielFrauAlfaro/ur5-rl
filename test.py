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


TEST = False
env_id = "ur5_rl/Ur5Env-v0"
n_training_envs = 3
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
            aux_env = make_vec_env(env_id, n_envs = n_eval_envs, seed=0, env_kwargs={"render_mode": "DIRECT", "show": False})
            self.eval_env = VecNormalize(aux_env, norm_obs=True, norm_reward=True)
            self.training_env.reset()


        return continue_training


if __name__ == "__main__":
    print("|| Compiling ...")
    
    # Training environments
    vec_env  = make_vec_env(env_id, n_envs=n_training_envs, vec_env_cls = SubprocVecEnv, seed=0, env_kwargs={"render_mode": "DIRECT", "show": False}) #vec_env_cls = SubprocVecEnv

    # Evaluation environments
    eval_env = make_vec_env(env_id, n_envs=n_eval_envs, vec_env_cls = SubprocVecEnv, seed=0, env_kwargs={"render_mode": "DIRECT", "show": True})


    # Observation space
    q_space = vec_env.observation_space["ee_position"]
    image_space = vec_env.observation_space["image"]

    q_shape = q_space.shape
    in_channels, frame_w, frame_h = image_space.shape

    # --- Arquitecture ---    
    residual = True
    channels = [in_channels, 16, 32, 32]
    kernel = 3          
    m_kernel = 3
    n_layers = len(channels) - 1

    out_vector_features = 100
    features_dim = 256   
    
    # Use  custom feature extractor in the policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim = features_dim,
                                       residual = residual, 
                                       channels = channels, kernel = kernel, m_kernel = m_kernel,
                                       n_layers = n_layers, out_vector_features = out_vector_features),
        net_arch=dict(
            pi=[features_dim, 32],  # Adjust the size of these layers based on the requirements
            vf=[features_dim, 32],  # Adjust the size of these layers based on the requirements
            qf=[features_dim, 32]),
        share_features_extractor = False
    )

    # --- Callbacks ---
    save_freq = 1000

    eval_log_dir = "my_models_eval/"
    eval_callback = EvalCallback(eval_env, best_model_save_path="./models_eval/",
                             log_path="./logs/", eval_freq=max(save_freq // n_training_envs, 1),
                             deterministic=True, render=False)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_training_envs, 1),
        save_path="./my_models_eval/",
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    
    # --- Action noise ---
    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

    # Model declaration
    model = SAC("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, 
                verbose=100, buffer_size = 14000,  batch_size = 256, tensorboard_log="logs/", 
                train_freq=10, learning_rate = 0.00073, gamma = 0.99, seed = 42,
                use_sde = False, sde_sample_freq = 8, action_noise = None)         # See logs: tensorboard --logdir logs/
    
    # Training 
    if not TEST:
        print("|| Training ...")
        model.learn(total_timesteps=50000, log_interval=5, tb_log_name= "Test", callback = [checkpoint_callback], progress_bar = True)
        model.save("./my_models_eval/best_model.zip")

    # Test
    else:
        print("|| Loading model for testing ...")
        model = SAC.load("./my_models_eval/rl_model_9000_steps.zip")
    

    # Close enviroments
    vec_env.close()
    eval_env.close()

    
    if TEST:
        model.policy.eval()
        print("|| Testing ...")

        r = 0
        vec_env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")
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

