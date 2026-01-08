"""
Deep Q-Network Agent for SDWAN Path Selection
"""

from stable_baselines3 import DQN
import sys
sys.path.append('..')
from simulator.network_env import SDWANPathEnv

class SDWANOptimizer:
    def __init__(self, num_paths=4):
        self.env = SDWANPathEnv(num_paths=num_paths)
        self.model = None
    
    def train(self, total_timesteps=100000):
        print("Training SDWAN Path Optimizer...")
        
        self.model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=1e-3,
            buffer_size=50000,
            batch_size=64,
            verbose=1
        )
        
        self.model.learn(total_timesteps=total_timesteps)
        print("✓ Training complete")
        return self.model
    
    def predict_best_path(self, observation):
        if self.model is None:
            raise ValueError("Model not trained")
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
    
    def save(self, path="sdwan_optimizer"):
        self.model.save(path)
    
    def load(self, path="sdwan_optimizer"):
        self.model = DQN.load(path, env=self.env)

if __name__ == "__main__":
    optimizer = SDWANOptimizer(num_paths=4)
    print("SDWAN Optimizer ready for training")
