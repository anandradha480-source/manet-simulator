"""
Deep Deterministic Policy Gradient (DDPG) Agent for MANET Next-Hop Routing.
"""

import copy
import random
from collections import deque
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.networks import ActorNetwork, CriticNetwork


class ReplayBuffer:
    """Experience Replay Buffer for off-policy DDPG updates."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class GaussianNoise:
    """Exploration noise for continuous action scores."""

    def __init__(self, action_dim: int, sigma: float = 0.2, decay: float = 0.9995, min_sigma: float = 0.05):
        self.action_dim = action_dim
        self.sigma = sigma
        self.decay = decay
        self.min_sigma = min_sigma

    def sample(self) -> np.ndarray:
        noise = np.random.normal(0, self.sigma, size=self.action_dim)
        self.sigma = max(self.min_sigma, self.sigma * self.decay)
        return noise

    def reset(self, initial_sigma: Optional[float] = None):
        if initial_sigma is not None:
            self.sigma = initial_sigma


class DDPGAgent:
    """
    DDPG Agent.
    - Evaluates candidate neighbor states.
    - Outputs continuous score vector for neighbors.
    - Updates Actor and Critic networks with soft target updates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        noise_sigma: float = 0.2,
        device: Optional[torch.device] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

        # Actor & Target
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic & Target
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-2)

        # Replay buffer & noise
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.noise = GaussianNoise(action_dim, sigma=noise_sigma)

        # Tracking
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Takes state and returns continuous score vector.
        Action shape: (action_dim,)
        """
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy().squeeze(0)
        self.actor.train()

        if add_noise:
            action = action + self.noise.sample()
            action = np.clip(action, -1.0, 1.0)

        return action

    def update(self) -> Optional[Tuple[float, float]]:
        """Perform one step of gradient descent on Actor and Critic."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # --- Update Critic ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            expected_q = rewards + (1.0 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, expected_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- Update Actor ---
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # --- Soft Target Updates ---
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        c_loss = float(critic_loss.item())
        a_loss = float(actor_loss.item())
        self.critic_losses.append(c_loss)
        self.actor_losses.append(a_loss)

        return c_loss, a_loss

    def _soft_update(self, source_net: nn.Module, target_net: nn.Module):
        """theta_target = tau * theta_source + (1 - tau) * theta_target"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filepath: str):
        """Save model checkpoints."""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoints."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
