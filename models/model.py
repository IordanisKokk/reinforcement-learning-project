import os
import re
import yaml
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.logger import configure
from utils import device

class ModelMaker:
    
    def __init__(self, env, algorithm, observation_space, config, load_model):
        self.env = env
        self.algorithm = algorithm
        self.observation_space = observation_space
        self.config = config
        self.load_model = load_model
    
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
        
    def get_latest_model(self, algorithm, observation_space):
        model_dir = f"./models/{algorithm}/{observation_space}"
        if not os.path.exists(model_dir):
            print("No Directory found for the specified algorithm and observation space.")
            return None
        models = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
        if not models:
            print("No models found in the specified directory.")
            return None
        # Extract numeric value from the filename using regex.
        latest_model = max(models, key=lambda x: int(re.findall(r'\d+', x)[-1]))
        return os.path.join(model_dir, latest_model)

    def make_model(self):
        model = None
    
        config = self.load_config(self.config)
        algorithm = self.algorithm.upper()
        policy = "CnnPolicy" if self.observation_space == "image" else "MlpPolicy"

        if self.load_model:
            model_path = self.get_latest_model(algorithm, self.observation_space)
            if model_path:
                print(f"\nLoading model from {model_path}\n")
                if algorithm == "PPO":
                    model = PPO.load(model_path, env=self.env, device=device.get_device())
                elif algorithm == "DQN":
                    model = DQN.load(model_path, env=self.env, device=device.get_device())
            else:
                print("\nNo saved model found. Starting training from scratch.\n")
                
        if model == None:    
            if algorithm == "PPO":
                model = PPO(policy, self.env, verbose=1, tensorboard_log=f"./logs/{algorithm}/{self.observation_space}/pong", device=device.get_device(), **config)
            elif algorithm == "DQN":
                model = DQN(policy, self.env, verbose=1, tensorboard_log=f"./logs/{algorithm}/{self.observation_space}/pong", device=device.get_device(), **config)
            else:
                raise ValueError("Invalid algorithm. Choose 'PPO' or 'DQN'")

        new_logger = configure(f"./logs/{algorithm}/{self.observation_space}", ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        return model