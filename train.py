import argparse
import gymnasium as gym
from callbacks import SaveModelCallback
from envs.donkey_kong_env import make_donkey_kong_env
from models.model import ModelMaker
TOTAL_TIMESTEPS = 5_000_000
    
def calculate_remaining_timesteps(model, total_timesteps):
    """Calculate the remaining timesteps for training."""
    if model is None:
        return total_timesteps
    else:
        return total_timesteps - model.num_timesteps    

def train(env, algorithm, observation_space, config_file, load_model=False):
    
    model_maker = ModelMaker(
        env,
        algorithm,
        observation_space,
        config_file,
        load_model
    )
    model = model_maker.make_model()
    callback = SaveModelCallback(save_freq=100_000, save_path=f"./models/{algorithm}/{observation_space}/{'pong'}")
    
    remaining_timesteps = calculate_remaining_timesteps(model, TOTAL_TIMESTEPS)

    if remaining_timesteps <= 0:
        print("Model already trained for the specified total timesteps. Exiting.")
        return
    print(f"Remaining timesteps: {remaining_timesteps}")
    print("Beginning training\n")
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name=f"{algorithm}-{observation_space}",
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
    parser.add_argument("--load", action="store_true", default=False, help="If set, load the latest saved model and resume training. Default is to start from scratch.")
    args = parser.parse_args()
    
    env = make_donkey_kong_env(render_mode=args.render_mode, observation_space=args.observation_space)
    obs = env.reset()
    train(env, args.algorithm, args.observation_space, args.config, args.load)
    
env.close()
