import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from .networks import PolicyNetwork, ValueNetwork

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.value_net = ValueNetwork(state_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.mse_loss = nn.MSELoss()
        
    def select_action(self, state):
        """Select action for training (stochastic)."""
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():  # ✅ 추가: 메모리 절약
            probs = self.policy_old(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.value_net(state)
        
        return action.item(), log_prob.item(), value.item()
    
    def select_action_eval(self, state):
        """Select action for evaluation (deterministic)."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            probs = self.policy(state)
            action = torch.argmax(probs)
        return action.item()
        
    def train_step(self, buffer, returns, advantages):
        """
        Perform PPO update step.s
        """
        old_states, old_actions, old_logprobs, _ = buffer.get_tensors()
        old_states = old_states.to(self.device)
        old_actions = old_actions.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # ✅ 1e-7 -> 1e-8
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            probs = self.policy(old_states)
            dist = Categorical(probs)
            logprobs = dist.log_prob(old_actions)
            state_values = self.value_net(old_states).squeeze()
            entropy = dist.entropy()  # ✅ 엔트로피 계산
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # ✅ Final loss (엔트로피 보너스 양수로 수정)
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * self.mse_loss(state_values, returns)
            entropy_bonus = 0.05 * entropy.mean()  #  탐험 장려
            
            loss = actor_loss + critic_loss - entropy_bonus  # 마이너스 (엔트로피 최대화)
            
            # take gradient step
            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            
            # ✅ Gradient Clipping (안정성)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            self.value_optimizer.step()
            
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def save(self, path):
        """✅ state_dict만 저장 (권장 방식)"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'lr': self.lr,
            'gamma': self.gamma,
            'eps_clip': self.eps_clip,
            'K_epochs': self.K_epochs
        }, path)
        
    @classmethod
    def load(cls, path, device=None):
        """✅ state_dict 로드 (권장 방식)"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        checkpoint = torch.load(path, map_location=device)
        
        # 에이전트 재생성
        agent = cls(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            lr=checkpoint['lr'],
            gamma=checkpoint['gamma'],
            eps_clip=checkpoint['eps_clip'],
            K_epochs=checkpoint['K_epochs']
        )
        
        # 가중치 로드
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        agent.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        agent.policy.eval()  # ✅ 평가 모드로 전환
        agent.policy_old.eval()
        agent.value_net.eval()
        
        return agent