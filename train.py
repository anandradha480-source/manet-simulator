"""
Training script for DDPG Routing Agent in MANET.
"""

import os
import json
import time
from typing import Dict, Any, List
import numpy as np
import torch

from manet.simulator import MANETSimulator
from rl.environment import MANETRoutingEnv
from rl.agent import DDPGAgent


def train(
    num_episodes: int = 120,
    steps_per_episode: int = 50,
    batch_size: int = 64,
    warmup_steps: int = 300,
    save_dir: str = "models",
    results_dir: str = "results",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains the DDPG agent in the MANET environment.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"=== Starting DDPG Training ({num_episodes} episodes, {steps_per_episode} steps/ep) ===")
    start_time = time.time()

    sim = MANETSimulator(num_nodes=30, seed=seed)
    env = MANETRoutingEnv(simulator=sim, max_neighbors=10)

    agent = DDPGAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=128,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=batch_size,
    )

    history: Dict[str, List[float]] = {
        "episode_rewards": [],
        "critic_losses": [],
        "actor_losses": [],
        "pdr": [],
        "link_breaks": [],
        "delivered": [],
    }

    total_steps = 0
    best_reward = -float("inf")

    # Warmup phase: fill replay buffer with exploration transitions
    print(f"Collecting {warmup_steps} warmup steps for replay buffer...")
    state = env.reset(seed=seed)
    for _ in range(warmup_steps):
        action = np.random.uniform(-1.0, 1.0, size=env.action_dim)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_steps += 1

    print("Warmup complete. Beginning training loop...")

    for ep in range(1, num_episodes + 1):
        # Vary episode seeds for generalized mobility patterns
        state = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_c_losses = []
        ep_a_losses = []

        for step in range(steps_per_episode):
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, info = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            total_steps += 1

            # Update networks
            loss_tuple = agent.update()
            if loss_tuple is not None:
                c_loss, a_loss = loss_tuple
                ep_c_losses.append(c_loss)
                ep_a_losses.append(a_loss)

        # Episode metrics
        metrics = sim.get_metrics()
        avg_c_loss = float(np.mean(ep_c_losses)) if ep_c_losses else 0.0
        avg_a_loss = float(np.mean(ep_a_losses)) if ep_a_losses else 0.0

        history["episode_rewards"].append(float(ep_reward))
        history["critic_losses"].append(avg_c_loss)
        history["actor_losses"].append(avg_a_loss)
        history["pdr"].append(float(metrics["pdr"]))
        history["link_breaks"].append(float(metrics["link_breaks"]))
        history["delivered"].append(float(metrics["total_delivered"]))

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(save_dir, "best_ddpg.pt"))

        if ep % 10 == 0 or ep == 1:
            print(
                f"Episode {ep:3d}/{num_episodes} | "
                f"Reward: {ep_reward:7.2f} | "
                f"PDR: {metrics['pdr']:5.1f}% | "
                f"Delivered: {int(metrics['total_delivered'])} | "
                f"Breaks: {int(metrics['link_breaks'])} | "
                f"Critic Loss: {avg_c_loss:6.4f} | "
                f"Noise Sigma: {agent.noise.sigma:.3f}"
            )

    agent.save(os.path.join(save_dir, "latest_ddpg.pt"))

    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start_time
    print(f"=== Training Completed in {elapsed:.1f}s | Best Reward: {best_reward:.2f} ===")
    return history


if __name__ == "__main__":
    train()
