import ur5_rl
import gymnasium as gym
from stable_baselines3 import SAC
from networks_SB import CustomCombinedExtractor

if __name__ == "__main__":
    print("|| Compiling ...")
    env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")
    env_ = gym.make_vec("ur5_rl/Ur5Env-v0", num_envs=3)
    print("\n\n")

    q_space = env.observation_space["q_position"]
    image_space = env.observation_space["image"]

    q_shape = q_space.shape
    in_channels, frame_w, frame_h = image_space.shape

    print(in_channels)
    print(frame_h)
    print(frame_w)
    
    residual = True
    channels = [in_channels, 16, 32]
    kernel = 3
    m_kernel = 2
    n_layers = len(channels) - 1

    out_q_features = 18
    features_dim = 128      

    

    # Use your custom feature extractor in the policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim = features_dim,
                                       residual = residual, 
                                       channels = channels, kernel = kernel, m_kernel = m_kernel,
                                       n_layers = n_layers, out_q_features = out_q_features),
        net_arch=dict(
            pi=[features_dim, 64],  # Adjust the size of these layers based on your requirements
            vf=[features_dim, 64],  # Adjust the size of these layers based on your requirements
            qf=[features_dim, 64])
    )

    model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, 
                verbose=100, buffer_size=50,  tensorboard_log="logs/", train_freq=1,
                learning_rate = 0.001, gamma = 0.99, seed = 42,
                use_sde = True, sde_sample_freq = 10)         # See logs: tensorboard --logdir logs/

    model.learn(total_timesteps=106, log_interval=1, tb_log_name= "ESTA ES", progress_bar = True)


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


