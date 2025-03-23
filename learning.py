import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset()

n_episodes = 10
n_steps_per_episode = 100

print(f"Available Actions: {env.action_space}")
print(f"Available Actions: {env.action_space.n}")
print(f"Observation Space: {env.observation_space.shape}")


for _ in range(n_episodes):
    print(f"Episode: {_}")
    for _ in range(n_steps_per_episode):
        action = env.action_space.sample()
        # print(f"Action: {action}")
        obs, reward, done, truncated, info = env.step(action)
        if _ % 10 == 0:
            print(f"Observation: {obs} Reward: {reward} Done: {done} Truncated: {truncated}")
        
        if done or truncated:
            obs, info = env.reset()
            break
    
env.close()
        