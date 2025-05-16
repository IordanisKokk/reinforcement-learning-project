import gymnasium as gym

class RewardClipper(gym.RewardWrapper):
    def __init__(self, env, min_reward=-1, max_reward=1):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, reward):
        return max(self.min_reward, min(self.max_reward, reward))
