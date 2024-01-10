import ur5_rl
import gymnasium as gym
from buffer import ReplayBuffer
import warnings

print("|| Compiling ...")
env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")
env_ = gym.make_vec("ur5_rl/Ur5Env-v0", num_envs=3)
print("|| Reseting environment ...")

buffer = ReplayBuffer(max_size = 256)

print("|| Sampling ...")
for j in range(100):
    print("--- Epoch ", j)
    obs, info = env.reset(seed=0, options={})
    obs.update(info)

    with warnings.catch_warnings(record = True) as w:
        while True:

            # TODO: función del agente
            action = env.action_space.sample()

            env.set_warning(w)
            obs_, reward, done, truncated, info = env.step(action)
            obs_.update(info)
            

            buffer.store_transition(obs, action, reward, obs_, (done or truncated))
            
            # env.render()
            
            if truncated or done:
                break

            obs = obs_

print("|| Success")
env.close()


