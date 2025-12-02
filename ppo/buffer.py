import torch
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute GAE (Generalized Advantage Estimation) and Returns.
        """
        returns = []
        advantages = []
        
        gae = 0
        # Iterate backwards
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = last_value
            else:
                next_value = self.values[i + 1]
                
            mask = 1 - self.dones[i]
            delta = self.rewards[i] + gamma * next_value * mask - self.values[i]
            gae = delta + gamma * gae_lambda * mask * gae
            
            returns.insert(0, gae + self.values[i])
            advantages.insert(0, gae)
            
        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)

    def get_tensors(self):
        return (
            torch.tensor(np.array(self.states), dtype=torch.float32),
            torch.tensor(np.array(self.actions), dtype=torch.long),
            torch.tensor(np.array(self.log_probs), dtype=torch.float32),
            torch.tensor(np.array(self.values), dtype=torch.float32)
        )
