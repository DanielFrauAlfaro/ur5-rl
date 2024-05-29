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
import re

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
            obs, reward, terminated, truncated, info = vec_env.step(action)

        if render:
            img = vec_env.render()
            cv.imshow("AA", img)
            cv.waitKey(1)

    is_touching, g = vec_env.unwrapped.grasping(0)

    return is_touching

def get_down(env, list_actions, render):
    while True:
        obs, reward, terminated, truncated, info = vec_env.step(np.array([0.0, 0.0, -0.4, 0.0, 0.0, 0.0]))
        list_actions.append(-action)

        if render:
            img = vec_env.render()
            cv.imshow("AA", img)
            cv.waitKey(1)

        if info["limit"]:
            break

    return list_actions


if __name__ == "__main__":

    path = "./5.0_aux_D/"
    dir_list = os.listdir(path)

    pattern = r'\d+'
    numbers_list = []

    # dir_list.remove("best_model.zip")
    numbers_list = [re.findall(pattern, n)[0] for n in dir_list]
    # dir_list = ["pito.zip"]
    
    vec_env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")
    # obs, info = vec_env.reset()
    

    results = {}
    num_tests = 25
    render = False

    
    for idx, model_name in enumerate(dir_list):
        
        r = 0
        error_pos = 0
        error_or = 0
        error_dq = 0

        # Loading model
        model = SAC.load(path + model_name)      
        model.policy.eval()
        print(f"{idx}. Loading model: " + model_name)

        

        for n in range(num_tests):
            print(f"   Test {n}")
            
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

                    get_down(vec_env, list_actions, render)
                    success = grasp(vec_env, list_actions, render)
                    obs, info = vec_env.reset()
                    dq_error, d_error, or_error = vec_env.unwrapped.get_error()
                    error_pos += d_error
                    error_or += or_error
                    error_dq += dq_error
                    break



        results[model_name] = {
            "idx": numbers_list[idx],
            "mean_reward" : r / num_tests,
            "distance_error": error_pos / num_tests,
            "orientation_error": error_or / num_tests,
            "dq_error": error_dq / num_tests
        }

        print(f"   -- Mean Reward: {r / num_tests}")
        print(f"   -- Mean Error: {error_pos / num_tests}\n")


    # # Serializing json
    # json_object = json.dumps(results, indent=4)
    
    # # Writing to sample.json
    # with open("test_results_withError.json", "w") as outfile:
    #     outfile.write(json_object)
    