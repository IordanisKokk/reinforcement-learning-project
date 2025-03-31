import os
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnTimestepCallback(BaseCallback):
    def __init__(self, save_freq, save_path, model_algo, verbose=1):
        super(SaveOnTimestepCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.model_algo = model_algo
        
    def _on_step(self) -> bool:
        # Save model and logs every self.save_freq steps
        if self.n_calls % self.save_freq == 0:
            model_save_path = os.path.join(self.save_path, f"model_{self.model_algo}_{self.n_calls}.zip")
            self.model.save(model_save_path)
            print(f"Model saved at {model_save_path}")
        return True
