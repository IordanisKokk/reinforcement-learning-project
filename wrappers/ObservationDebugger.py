import gymnasium as gym
import numpy as np

class ObservationDebugger(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        print(f"Obs shape: {obs.shape}, Min: {np.min(obs)}, Max: {np.max(obs)}, Sample: {obs[:10]}")
        return obs
