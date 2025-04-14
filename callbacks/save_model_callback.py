import os
from stable_baselines3.common.callbacks import BaseCallback

class SaveModelCallback(BaseCallback):
    """Class for saving models at regular intervals during training.
    This callback inherits from BaseCallback and overrides the _on_step method to save the model.

    Args:
        BaseCallback (BaseCallback): BaseCallback class from stable_baselines3.
    """
    
    def __init__(self, save_path: str, save_freq: int, verbose: int = 1):
        """Initialize the SaveModelCallback.

        Args:
            save_path (str): Path to save the model.
            save_freq (int): Frequency of saving the model.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        
        os.makedirs(self.save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        """Called at each step of training. Saves the model if the save frequency is reached.

        Returns:
            bool: True to continue training, False to stop.
        """
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"model_{self.model.num_timesteps}_steps.zip")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved to {model_path} at step {self.model.num_timesteps}")
        return True