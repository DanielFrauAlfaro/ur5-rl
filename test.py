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
    
    residual = False
    channels = [in_channels, 16, 32]
    kernel = 3
    m_kernel = 2
    n_layers = len(channels) - 1

    out_q_features = 10
    features_dim = 128          

    # Use your custom feature extractor in the policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim = features_dim,
                                       residual = residual, 
                                       channels = channels, kernel = kernel, m_kernel = m_kernel,
                                       n_layers = n_layers, out_q_features = out_q_features),
        net_arch=dict(
            pi=[features_dim, 32],  # Adjust the size of these layers based on your requirements
            vf=[features_dim, 32],  # Adjust the size of these layers based on your requirements
            qf=[features_dim, 32]),
        share_features_extractor = True
    )

    model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, 
                verbose=100, buffer_size = 30000,  batch_size = 256, tensorboard_log="logs/", train_freq=10,
                learning_rate = 0.00073, gamma = 0.98, seed = 42,
                use_sde = True, sde_sample_freq = 8)         # See logs: tensorboard --logdir logs/
    
    model.learn(total_timesteps=100000, log_interval=5, tb_log_name= "Test", progress_bar = True)
    
    model.save("./models/sac_ur5_stage_1.0")
    


    # Testing
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        # action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

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


