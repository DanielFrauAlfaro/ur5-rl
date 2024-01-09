import ur5_rl
import gymnasium as gym
from buffer import ReplayBuffer

print("|| Compiling ...")
env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")
env_ = gym.make_vec("ur5_rl/Ur5Env-v0", num_envs=3)
print("|| Reseting environment ...")

buffer = ReplayBuffer(max_size = 256)

print("|| Sampling ...")
for j in range(3):
    print("--- Epoch ", j)
    obs, __ = env.reset(seed=0, options={})

    while True:

        # TODO: función del agente
        action = env.action_space.sample()

        obs_, reward, done, truncated, info = env.step(action)
        
        buffer.store_transition(obs, action, reward, obs_, (done or truncated))

        env.render()

        # IMPORTANTE: a la hora de incluir el estado en el buffer, hacer obs.update(info) para incluir la imagen
        
        # if truncated:
        #     break

        obs = obs_

        # TODO: actualizar estado del agente --> después o durante el episodio

print("|| Success")
env.close()


