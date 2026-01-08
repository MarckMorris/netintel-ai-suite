"""
SDWAN Network Environment for Reinforcement Learning
Simulates Walmart's multi-ISP, multi-cloud topology
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SDWANPathEnv(gym.Env):
    def __init__(self, num_paths=4):
        super().__init__()
        
        self.num_paths = num_paths
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_paths * 5,), dtype=np.float32)
        self.action_space = spaces.Discrete(num_paths)
        
        self.isp_profiles = [
            {'name': 'Comcast', 'base_latency': 25, 'reliability': 0.95, 'cost': 1.0},
            {'name': 'ATT', 'base_latency': 30, 'reliability': 0.93, 'cost': 1.2},
            {'name': 'Verizon', 'base_latency': 28, 'reliability': 0.94, 'cost': 1.1},
            {'name': '5G', 'base_latency': 40, 'reliability': 0.85, 'cost': 2.0}
        ]
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.path_states = []
        for profile in self.isp_profiles[:self.num_paths]:
            state = self._generate_path_state(profile)
            self.path_states.append(state)
        
        self.current_step = 0
        return self._get_observation(), {}
    
    def _generate_path_state(self, profile):
        return {
            'latency': np.clip(np.random.normal(profile['base_latency'], 10), 5, 200) / 200,
            'bandwidth_util': np.random.uniform(0.3, 0.9),
            'packet_loss': np.random.exponential(0.01) if np.random.random() > (1-profile['reliability']) else 0,
            'cost': profile['cost'] / 2.0,
            'jitter': np.random.exponential(5) / 50
        }
    
    def step(self, action):
        self.current_step += 1
        selected_path = self.path_states[action]
        
        reward = self._calculate_reward(selected_path)
        
        for i in range(self.num_paths):
            self.path_states[i] = self._generate_path_state(self.isp_profiles[i])
        
        done = self.current_step >= 1000
        truncated = False
        
        return self._get_observation(), reward, done, truncated, {}
    
    def _calculate_reward(self, path_state):
        latency_penalty = -path_state['latency'] * 2.0
        packet_loss_penalty = -path_state['packet_loss'] * 5.0
        cost_penalty = -path_state['cost'] * 0.5
        bandwidth_bonus = (1 - path_state['bandwidth_util']) * 0.5
        
        return latency_penalty + packet_loss_penalty + cost_penalty + bandwidth_bonus
    
    def _get_observation(self):
        obs = []
        for path in self.path_states:
            obs.extend([path['latency'], path['bandwidth_util'], path['packet_loss'], 
                       path['cost'], path['jitter']])
        return np.array(obs, dtype=np.float32)
