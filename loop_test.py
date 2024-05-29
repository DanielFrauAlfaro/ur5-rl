import ur5_rl
import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import cv2 as cv
import os
import json
import re

# Function for image rendering
def image_render(env, render):
    
    if render:
        img = vec_env.render()
        cv.imshow("Environment", img)
        cv.waitKey(1)


# Function to grasp the object
def grasp(env, list_actions, render):
    # Starts the grasping process
    grasped = False
    is_touching = False
    cnt = 0

    # While the list of actions has actions ...
    while len(list_actions) > 1:

        # ... if the gripper is not touching the object ...
        if not grasped:

            # ... advance simulation with no action (zero vector)
            __, __, __, __, __ = vec_env.step(np.zeros(6))

            # Check if it is in contact with the object
            is_touching, g = vec_env.unwrapped.grasping()

            # Increase or reset the counter
            if is_touching:
                cnt += 1
            else:
                cnt = 0

            # It is grasping if is touching during 4 steps
            grasped = cnt == 4

            # Gripper limit
            if g >= 100:
                break

         # If it is grasping the object ...
        else:
            # Applies the action in reverse order
            action = list_actions.pop()
            __, __, __, __, __ = vec_env.step(action)

        image_render(env=env, render=render)


# Function to move the robot downwards
def get_down(env, list_actions, render):
    # Endless loop
    while True:
        # Apply an action to move the robot downwards
        obs, reward, terminated, truncated, info = vec_env.step(np.array([0.0, 0.0, -0.4, 0.0, 0.0, 0.0]))

        # Save inverse action
        list_actions.append(-action)

        # Image render
        image_render(env=env, render=render)
        
        # Check for limits
        if info["limit"]:
            break

    return list_actions


# --- Main ---
if __name__ == "__main__":

    # Get all the models in a directory (all must have a number indicating their saved step)
    path = "./models/"
    dir_list = os.listdir(path)

    pattern = r'\d+'
    numbers_list = []

    dir_list.remove("best_model.zip")
    numbers_list = [re.findall(pattern, n)[0] for n in dir_list]
    
    # Environment
    vec_env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")    

    # --- Config ---
    results = {}
    num_tests = 25
    render = False

    # Models for testing
    for idx, model_name in enumerate(dir_list):
        
        r = 0
        error_pos = 0
        error_or = 0
        error_dq = 0

        # Loading model
        model = SAC.load(path + model_name)      
        model.policy.eval()
        print(f"{idx}. Loading model: " + model_name)

        
        # Testing
        for n in range(num_tests):
            print(f"   Test {n}")
            
            # Reset environment
            obs, info = vec_env.reset()
            list_actions = []

            # Endless loop
            while True:

                # Generate action
                action, _states = model.predict(obs, deterministic = True)

                # Apply action (a) to the environment and recieve:
                #   state (s), reward (r), end flags (terminated and truncated)
                obs, reward, terminated, truncated, info = vec_env.step(action)
                
                # Save inverse action
                list_actions.append(-action)

                # Accumulate reward
                r += reward
                
                # Image render
                image_render(env=vec_env, render=render)

                # Episode end
                if terminated or truncated:
                    
                    # Take the robot vertically downwards
                    get_down(vec_env, list_actions, render)

                    # Perform a simple grasp
                    grasp(vec_env, list_actions, render)

                    # Reset environment
                    obs, info = vec_env.reset()

                    # Save metrics
                    dq_error, d_error, or_error = vec_env.unwrapped.get_error()
                    error_pos += d_error
                    error_or += or_error
                    error_dq += dq_error

                    break

        # Save results
        results[model_name] = {
            "idx": numbers_list[idx],
            "mean_reward" : r / num_tests,
            "distance_error": error_pos / num_tests,
            "orientation_error": error_or / num_tests,
            "dq_error": error_dq / num_tests
        }

        print(f"   -- Mean Reward: {r / num_tests}")
        print(f"   -- Mean Error: {error_pos / num_tests}\n")


    # Serializing json
    json_object = json.dumps(results, indent=4)
    
    # Writing to sample.json
    with open("results.json", "w") as outfile:
        outfile.write(json_object)
    