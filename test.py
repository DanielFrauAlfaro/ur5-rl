import ur5_rl
import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import cv2 as cv


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

    path = "./models/"
    model_name = "best_model.zip"

    # Environment
    vec_env = gym.make("ur5_rl/Ur5Env-v0", render_mode = "DIRECT")    

    # --- Config ---
    render = False
    r = 0
    list_actions = []


    # Loading model
    model = SAC.load(path + model_name)      
    model.policy.eval()
    print(f"Loading model: " + model_name)


    # Reset environment
    obs, info = vec_env.reset()
    
    
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
            
            break
    