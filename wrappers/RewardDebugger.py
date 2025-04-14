import gymnasium as gym

class RewardDebugger(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        print(f"Reward: {reward}, Done: {done}")
        return obs, reward, terminated, truncated, info
