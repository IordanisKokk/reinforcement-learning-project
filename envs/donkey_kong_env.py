import ale_py
import gymnasium as gym
import supersuit as ss
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def make_donkey_kong_env(sticky_actions=True, frame_stacking=4, render_mode="rgb", observation_space="image") -> gym.Env:
    """_summary_

    Args:
        sticky_actions (bool, optional): _description_. Defaults to True.
        frame_stacking (int, optional): _description_. Defaults to 4.
        normalize (bool, optional): _description_. Defaults to True.
        grayscale (bool, optional): _description_. Defaults to True.
        render_mode (str, optional): _description_. Defaults to "rgb".
        
    Returns:
        gym.Env: Preprocessed gym environment.
    """
    if observation_space == "image":
        env_id = "ALE/Pong-v5"
        grayscale = True
        frame_stacking = 4
    elif observation_space == "ram":
        env_id = "ALE/Pong-ram-v5"
        grayscale = False
        frame_stacking = 0
    else:
        raise ValueError("Invalid observation type. Choose 'image' or 'ram'.")
    
    print(f"\n\nCreating environment with observation space: {observation_space}")
    print(f"Sticky actions: {sticky_actions}")
    print(f"Frame stacking: {frame_stacking}")
    print(f"Grayscale: {grayscale}")
    print(f"Render mode: {render_mode}\n\n")
    
    env = gym.make(env_id, render_mode=render_mode)
    if sticky_actions:
        print("Adding sticky actions")
        env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    if grayscale:
        print("Converting to grayscale")
        env = ss.color_reduction_v0(env, mode='full')
        env = ss.resize_v1(env, x_size=84, y_size=84)
    if frame_stacking > 0:
        print(f"Stacking {frame_stacking} frames")
        env = ss.frame_stack_v1(env, frame_stacking)
        
    return env