import ur5_rl
import gymnasium as gym

print("|| Compiling ...")
env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")
env_ = gym.make_vec("ur5_rl/Ur5Env-v0", num_envs=3)
print("|| Reseting environment ...")


print("|| Sampling ...")
for j in range(3):
    print("--- Epoch ", j)
    obs, __ = env.reset(seed=0, options={})
    
    while True:

        # TODO: función del agente
        action = env.action_space.sample()
        
        obs, reward, done, truncated, info = env.step(action)

        env.render()
        
        # if truncated:
        #     obs, __ = env.reset(seed=0, options={})
        
        # TODO: actualizar estado del agente --> después o durante el episodio

print("|| Success")
env.close()


