import ur5_rl
import gymnasium as gym
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize, VecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from networks_SB import CustomCombinedExtractor
import numpy as np
import cv2 as cv
import os
import pybullet as p
import time
import json
import os

def grasp(env, list_actions, render):
    grasped = False
    is_touching = False
    cnt = 0

    while len(list_actions) > 1:
        if not grasped:
            obs, reward, terminated, truncated, info = vec_env.step(np.zeros(6))
            is_touching, g = vec_env.unwrapped.grasping()

            grasped = cnt == 4

            if is_touching:
                cnt += 1
            else:
                cnt = 0

            if g >= 100:
                break

        else:
            action = list_actions.pop()
            print("Action: ", len(list_actions))
            obs, reward, terminated, truncated, info = vec_env.step(action)

        if render:
            img = vec_env.render()
            cv.imshow("AA", img)
            cv.waitKey(1)

    is_touching, g = vec_env.unwrapped.grasping(0)

    return is_touching



if __name__ == "__main__":

    path = "./6.0/"
    dir_list = os.listdir(path)

    
    vec_env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")
    obs, info = vec_env.reset()
    

    results = {}
    num_tests = 10
    render = False

    
    for idx, model_name in enumerate(dir_list):
        
        r = 0
        success = 0
        print(path + model_name)
        # Loading model
        model = SAC.load(path + model_name)      
        model.policy.eval()
        print(f"{idx}. Loading model: " + model_name)

        

        for n in range(num_tests):
            
            obs, info = vec_env.reset()
            list_actions = []

            while True:
                action, _states = model.predict(obs, deterministic = True)

                list_actions.append(-action)
                obs, reward, terminated, truncated, info = vec_env.step(action)
                
                r += reward
                
                if render:
                    img = vec_env.render()
                    cv.imshow("AA", img)
                    cv.waitKey(1)

                if terminated or truncated:
                    success += grasp(vec_env, list_actions, render)
                    obs, info = vec_env.reset()
                    break



        results[model_name] = {
            "idx": idx,
            "mean_reward" : r / num_tests,
            "success_rate": success / num_tests
        }

        print(f"   -- Mean Reward: {r / num_tests}")
        print(f"   -- Success Rate: {success / num_tests}\n")


    # Serializing json
    json_object = json.dumps(results, indent=4)
    
    # Writing to sample.json
    with open("test_results.json", "w") as outfile:
        outfile.write(json_object)
    