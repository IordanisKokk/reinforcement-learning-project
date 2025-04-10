import yaml
import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from callbacks import SaveModelCallback
from envs.donkey_kong_env import make_donkey_kong_env
from envs.donkey_kong_env import make_donkey_kong_env

TOTAL_TIMESTEPS = 50_000_000

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def train(env, algorithm, observation_space, config_file):
    config = load_config(config_file)
    algorithm = algorithm.upper()
    policy = "CnnPolicy" if observation_space == "image" else "MlpPolicy"

    if algorithm == "PPO":
        model = PPO(policy, env, verbose=1, tensorboard_log=f"./logs/{algorithm}/{observation_space}")
    elif algorithm == "DQN":
        model = DQN(policy, env, verbose=1, tensorboard_log=f"./logs/{algorithm}/{observation_space}")
    else:
        raise ValueError("Invalid algorithm. Choose 'PPO' or 'DQN'")
    
    new_logger = configure(f"./logs/{algorithm}/{observation_space}", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    callback = SaveModelCallback(save_freq=100_000, save_path=f"./models/{algorithm}/{observation_space}")
    print("Beginning training")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True
    )
    
    model.save(f"./models/trained_model-{algorithm}-{observation_space}-{model.num_timesteps}")
    print("Training complete! Model saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--algorithm", type=str, choices=["PPO", "DQN"], default="PPO", help="Algorithm to use for training")
    parser.add_argument("--observation_space", type=str, default="image", choices=["image", "ram"], help="Observation space type")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS, help="Total timesteps for training")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="Render mode for the environment")
    args = parser.parse_args()
    
    env = make_donkey_kong_env(render_mode=args.render_mode, observation_space=args.observation_space)
    env.reset()
    
    
    train(env, args.algorithm, args.observation_space, args.config)

    
env.reset()
