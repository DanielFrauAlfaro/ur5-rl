from my_parser import *
from sim import *
from trainer import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path for the configuration file.", type=str)
    args = parser.parse_args()
    
    params_parser = Param_manager()
    params_parser.set_data(path = args.path)

    vec_env = Env(data = params_parser.get_data())
    # e = vec_env.get_env()
    # e = gym.make("ur5_rl/Ur5Env-v0", render_mode = "human" ,data=params_parser.get_data())
    # states = e.reset()

    trainer = Trainer(vec_env=vec_env, data=params_parser.get_data())



    
    

    # # Endless loop
    # while True:

    #     # Generate action
    #     # action, _states = model.predict(obs, deterministic = True)

    #     # Apply action (a) to the environment and recieve:
    #     #   state (s), reward (r), end flags (terminated and truncated)
    #     a = np.zeros(6)
    #     obs, rew, done, info = e.step(np.array([[0.0,0.0,0.0,0.0,0.0,0.0]]))
        
        
        


    #     # # Episode end
    #     if done:

    #         # Reset environment
    #         obs, info = e.reset()


