from envs.donkey_kong_env import make_donkey_kong_env
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN  # Modify this based on the model type you're testing

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
        env = make_donkey_kong_env(render_mode="human", observation_space="ram")
        model = PPO.load("models/PPO/ram/model_14200000_steps.zip")
        
        evaluate_model(model, env, n_episodes=5)