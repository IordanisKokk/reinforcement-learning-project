from envs.environment import make_env
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN  # Modify this based on the model type you're testing
import supersuit as ss

env_id = "ALE/SpaceInvaders-v5"

def evaluate_model(model, env, n_episodes=10):
    """Evaluate a trained model."""
    total_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs)  # Get action from model
            obs, reward, done, _, _ = env.step(action)  # Take action in environment
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {n_episodes} episodes: {avg_reward}")


if __name__ == "__main__":
        env = gym.make(env_id, render_mode="rgb_array")
        print("Adding sticky actions")
        env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
        print("Converting to grayscale")
        env = ss.color_reduction_v0(env, mode="full")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        print(f"Stacking {4} frames")
        env = ss.frame_stack_v1(env, 4)
        model = PPO.load("models/trained_model-PPO-space-invaders-image-20003328.zip")
        
        evaluate_model(model, env, n_episodes=100)