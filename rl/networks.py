"""
PyTorch Actor and Critic Neural Networks for DDPG Routing Agent.
Actor outputs continuous scores for candidate neighbors.
Critic evaluates (state, action) pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ActorNetwork(nn.Module):
    """
    DDPG Actor Network.
    Maps state representation to continuous scores for each candidate neighbor.
    Action: continuous score vector in [-1, 1]^K.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        init_w: float = 3e-3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Output layer initialization
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        state: [batch_size, state_dim]
        returns: [batch_size, action_dim] scores in [-1, 1]
        """
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        scores = torch.tanh(self.fc3(x))
        return scores


class CriticNetwork(nn.Module):
    """
    DDPG Critic Network.
    Maps (state, action) to scalar Q-value Q(s, a).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        init_w: float = 3e-3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        # Combine state embedding with action
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        state: [batch_size, state_dim]
        action: [batch_size, action_dim]
        returns: [batch_size, 1] Q-value
        """
        x_state = F.relu(self.ln1(self.fc1(state)))
        x = torch.cat([x_state, action], dim=-1)
        x = F.relu(self.ln2(self.fc2(x)))
        q_value = self.fc3(x)
        return q_value
