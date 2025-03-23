import pettingzoo.butterfly.knights_archers_zombies_v10 as kaz
import numpy as np

# Initialize environment
env = kaz.parallel_env(render_mode="human")  # Use "rgb_array" for image-based observations
env.reset()
for agent in env.agents:
    print(f"Agent: {agent}")
    print(f"Observation Space: {env.observation_space(agent)}")
    print(f"Action Space: {env.action_space(agent)}")
    continue  # Only print for one agent


env.close()


from stable_baselines3 import PPO
from pettingzoo.utils.wrappers import parallel_to_aec
import supersuit as ss

# Convert ParallelEnv to AECEnv (needed for Stable-Baselines3)
aec_env = parallel_to_aec(kaz.parallel_env())

# Preprocess environment (frame stacking, normalization)
aec_env = ss.frame_stack_v1(aec_env, 4)
aec_env = ss.normalize_obs_v0(aec_env, env_min=-1, env_max=1)

# Train PPO agent
model = PPO("MlpPolicy", aec_env, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save("ppo_knight")
