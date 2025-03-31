import os
import time
from callbacks import CSVLogCallback, SaveOnTimestepCallback
import gymnasium as gym
from pettingzoo.butterfly import knights_archers_zombies_v10
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from supersuit import frame_stack_v2, color_reduction_v0, black_death_v3, resize_v1, pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

TIMESTEPS = 4_000_000

def create_env():
    """
    Create a new environment for the agents to interact with.
    Returns:
        gym.Env: The environment for the agents.
    """
    def _env_fn():
        env = knights_archers_zombies_v10.parallel_env(
            vector_state=False,
            max_cycles=1000,
            render_mode=None,
        )
        env = black_death_v3(env)
        env = color_reduction_v0(env, 'full')  # Convert to grayscale
        env = resize_v1(env, x_size=84, y_size=84)
        env = frame_stack_v2(env, 4)  # Stack 4 frames
        
        env = pettingzoo_env_to_vec_env_v1(env)
        env = concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")
        env = VecMonitor(env, filename=os.path.join("logs", "monitor", "centralized_ppo_train.csv"))
        return env

    return _env_fn()

def get_callbacks():
    """
    Get the callbacks for training the PPO agent.
    Returns:
        list: List of callbacks.
    """
    log_dir = os.path.join("logs", "csv")
    os.makedirs(log_dir, exist_ok=True)

    csv_log_callback = CSVLogCallback(
        save_freq=1000,
        csv_file_path=os.path.join(log_dir, "ppo_log.csv"),
        verbose=1
    )

    save_on_timestep_callback = SaveOnTimestepCallback(
        save_freq=1000,
        save_path=os.path.join("logs", "models"),
        model_algo="ppo",
        verbose=1
    )

    return [csv_log_callback, save_on_timestep_callback]

def create_ppo_model(env):
    """
    Create a PPO model for the given environment.
    Args:
        env (gym.Env): The environment to train on.
    Returns:
        PPO: The PPO model.
    """

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join("logs", "tensorboard")
    )
    
    return model

def train_centralized_ppo(env, model, callbacks):
    """
    Train a centralized PPO agent on the given environment.
    Args:
        env (gym.Env): The environment to train on.
        model (PPO): The PPO model to train.
    """

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callbacks,
        tb_log_name="PPO_Centralized",
        log_interval=1000,
        progress_bar=True,
    )
    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


# Create environment
env = create_env()
env.reset()

model = create_ppo_model(env)
callbacks = get_callbacks()
train_centralized_ppo(env, model, callbacks)

env.close()
