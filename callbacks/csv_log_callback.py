import os
import csv
from stable_baselines3.common.callbacks import BaseCallback

class CSVLogCallback(BaseCallback):
    """_summary_

    Args:
        BaseCallback (_type_): _description_
    """
    def __init__(self, save_freq, csv_file_path, verbose=1):
        super(CSVLogCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.csv_file_path = csv_file_path
        
        # Initialize CSV file and write header if it's the first run
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timesteps", "Episode Reward", "Episode Length"])
        
        # To track episode rewards and lengths
        self.episode_reward = 0
        self.episode_length = 0

    def _on_step(self) -> bool:
        # Tracking reward and episode length manually at each step
        self.episode_reward += self.locals.get("rewards", 0)
        self.episode_length += 1
        
        if self.n_calls % self.save_freq == 0:
            # Writing data to CSV file
            with open(self.csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.n_calls, self.episode_reward, self.episode_length])
        
        # If episode ends, reset the reward and length for the next episode
        if self.locals.get("done", False):
            self.episode_reward = 0
            self.episode_length = 0
        
        return True
