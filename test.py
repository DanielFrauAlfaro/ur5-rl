import ur5_rl
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from networks_SB import CustomCombinedExtractor
import numpy as np
import cv2 as cv
import os


TEST = False
env_id = "ur5_rl/Ur5Env-v0"
n_training_envs = 1
n_eval_envs = 2


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

# class SaveVecVideoCallback(BaseCallback):
#     """
#     Callback for saving a video of the environment to TensorBoard during training.

#     :param eval_env: The environment used for evaluation
#     :param eval_freq: Frequency at which to save videos (in terms of training steps)
#     :param n_eval_episodes: Number of episodes to run for each evaluation
#     :param deterministic: Whether to use deterministic or stochastic policy during evaluation
#     """

#     def __init__(self, eval_env: VecEnv, eval_freq: int = 1000, n_eval_episodes: int = 1, deterministic: bool = False):
#         super(SaveVecVideoCallback, self).__init__(callback_order=eval_freq)
#         self.eval_env = eval_env
#         self.eval_freq = eval_freq
#         self.n_eval_episodes = n_eval_episodes
#         self.deterministic = deterministic

#     def _on_step(self) -> bool:
#         if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
#             self.eval_policy()
#         return True

#     def eval_policy(self) -> None:
#         """
#         Evaluate the policy and save video to TensorBoard.
#         """
#         # Save video to TensorBoard
#         video_frames = []
#         obs, info = self.eval_env.reset()
#         for _ in range(self.n_eval_episodes):
#             terminated, truncated = False, False
#             while not (terminated or truncated):
#                 action, _ = self.model.predict(obs, deterministic=self.deterministic)
#                 obs, _, terminated, truncated, _ = self.eval_env.step(action)
#                 video_frames.append(self.eval_env.render())

#         # Convert video frames to uint8
#         video_frames = np.array(video_frames, dtype=np.uint8)

#         # Write video to TensorBoard
#         self.model.logger.record('logs/Videos', video_frames, len(video_frames))

class SaveVecVideoCallback(BaseCallback):
    """
    Callback for saving a video of the environment to a file during training.

    :param eval_env: The environment used for evaluation
    :param eval_freq: Frequency at which to save videos (in terms of training steps)
    :param n_eval_episodes: Number of episodes to run for each evaluation
    :param deterministic: Whether to use deterministic or stochastic policy during evaluation
    :param video_file: Name of the video file to save
    """

    def __init__(self, eval_env: VecEnv, eval_freq: int = 1000, n_eval_episodes: int = 1, deterministic: bool = False, video_file: str = 'output_video.mp4'):
        super(SaveVecVideoCallback, self).__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.video_file = video_file

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            self.eval_policy()
        return True

    def eval_policy(self) -> None:
        """
        Evaluate the policy and save video to a file.
        """
        # Save video to a file using imageio
        video_frames = []
        obs, info = self.eval_env.reset()
        for _ in range(self.n_eval_episodes):
            terminated, truncated = False, False
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                video_frames.append(self.eval_env.render())

        # Convert video frames to uint8
        video_frames = np.array(video_frames, dtype=np.uint8)

        # Write video frames to a file using imageio
        imageio.mimsave(self.video_file, video_frames, fps=30)



if __name__ == "__main__":
    print("|| Compiling ...")
    
    
    vec_env  = make_vec_env(env_id, n_envs=n_training_envs, seed=0, env_kwargs={"render_mode": "DIRECT", "show": False})

    eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0, env_kwargs={"render_mode": "DIRECT", "show": False})



    q_space = vec_env.observation_space["ee_position"]
    image_space = vec_env.observation_space["image"]

    q_shape = q_space.shape
    in_channels, frame_w, frame_h = image_space.shape
    
    residual = False
    channels = [in_channels, 16, 32, 32]
    kernel = 3          
    m_kernel = 3
    n_layers = len(channels) - 1

    out_vector_features = 100
    features_dim = 256   

    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
       
    eval_log_dir = "my_models_eval/"
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=eval_log_dir,
                                  log_path=eval_log_dir, eval_freq=max(500 // n_training_envs, 1),
                                  n_eval_episodes=1, deterministic=False,
                                  render=False)
    
    # video_callback = SaveVecVideoCallback(eval_env, eval_freq=max(500 // n_training_envs, 1), n_eval_episodes=1, deterministic=False)
    video_callback = SaveVecVideoCallback(eval_env, eval_freq=1000, n_eval_episodes=1, deterministic=False, video_file='output_video.mp4')


    # Use your custom feature extractor in the policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim = features_dim,
                                       residual = residual, 
                                       channels = channels, kernel = kernel, m_kernel = m_kernel,
                                       n_layers = n_layers, out_vector_features = out_vector_features),
        net_arch=dict(
            pi=[features_dim, 32],  # Adjust the size of these layers based on your requirements
            vf=[features_dim, 32],  # Adjust the size of these layers based on your requirements
            qf=[features_dim, 32]),
        share_features_extractor = False
    )

    model = SAC("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, 
                verbose=100, buffer_size = 16000,  batch_size = 256, tensorboard_log="logs/", 
                train_freq=10, learning_rate = 0.00073, gamma = 0.99, seed = 42,
                use_sde = False, sde_sample_freq = 8, action_noise = None)         # See logs: tensorboard --logdir logs/
    

    if not TEST:
        model.learn(total_timesteps=100000, log_interval=5, tb_log_name= "Test", callback = None, progress_bar = True)
        model.save("./my_models_eval/best_model.zip")
    else:
        model = SAC.load("./my_models_eval/best_model(1).zip")
    
    
    model.policy.eval()
    print("... Testing ...")
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print("Mean Reward: ", mean_reward)
    # print("STD reward:  ", std_reward, "\n\n")

    # Close enviroments
    # vec_env.close()
    # eval_env.close()
    # r = 0

    # vec_env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")
    # obs, info = vec_env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic = True)
    #     obs, reward, terminated, truncated, info = vec_env.step(action)
        
    #     print(reward)
    #     r += reward
    #     img = vec_env.render()
    #     cv.imshow("AA", img)
    #     cv.waitKey(1)

    #     if terminated or truncated:
    #         print(r, "--")
    #         r = 0
    #         obs, info = vec_env.reset()
            

    
    

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
    

