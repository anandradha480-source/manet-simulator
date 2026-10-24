"""
MANET DDPG Reward Function.
Reward = throughput - delay - energy + route_stability - penalty_for_link_breaks.
"""

from typing import Dict, Any


class MANETRewardCalculator:
    """
    Computes reward for next-hop routing decision:
    Reward = w_throughput * throughput
             - w_delay * delay
             - w_energy * energy
             + w_stability * route_stability
             - w_break * link_break_penalty
    """

    def __init__(
        self,
        w_throughput: float = 2.0,
        w_delay: float = 1.0,
        w_energy: float = 1.0,
        w_stability: float = 1.5,
        w_break: float = 5.0,
        delivery_bonus: float = 5.0,
        max_hop_energy: float = 1e-4,  # Typical hop energy scale in Joules
    ):
        self.w_throughput = w_throughput
        self.w_delay = w_delay
        self.w_energy = w_energy
        self.w_stability = w_stability
        self.w_break = w_break
        self.delivery_bonus = delivery_bonus
        self.max_hop_energy = max_hop_energy

    def compute_reward(
        self,
        curr_dist_to_dest: float,
        next_dist_to_dest: float,
        hop_delay: float,
        energy_consumed: float,
        link_expiration_time: float,
        is_link_break: bool,
        is_delivered: bool,
        radio_range: float = 250.0,
    ) -> Dict[str, float]:
        """
        Calculates composite reward and decomposes individual terms.
        """
        if is_link_break:
            # Harsh penalty for link break
            break_penalty = self.w_break
            total_reward = -break_penalty
            return {
                "reward": float(total_reward),
                "throughput_term": 0.0,
                "delay_term": 0.0,
                "energy_term": 0.0,
                "stability_term": 0.0,
                "break_penalty": float(break_penalty),
            }

        # 1. Throughput component: geographic progress towards destination + delivery bonus
        progress = (curr_dist_to_dest - next_dist_to_dest) / radio_range
        throughput = progress
        if is_delivered:
            throughput += self.delivery_bonus
        throughput_term = self.w_throughput * throughput

        # 2. Delay component: normalized transmission + propagation delay
        # Normalizing delay (hop_delay is typically ~2-5 ms)
        norm_delay = hop_delay / 0.01  # normalized by 10ms
        delay_term = self.w_delay * norm_delay

        # 3. Energy component: normalized energy spent
        norm_energy = min(1.0, energy_consumed / self.max_hop_energy)
        energy_term = self.w_energy * norm_energy

        # 4. Route stability component: Link Expiration Time (LET)
        # Normalize LET (e.g. 5 seconds is very stable in high-mobility MANET)
        norm_stability = min(1.0, link_expiration_time / 5.0)
        stability_term = self.w_stability * norm_stability

        # Total Composite Reward:
        # throughput minus delay minus energy plus route stability
        total_reward = throughput_term - delay_term - energy_term + stability_term

        return {
            "reward": float(total_reward),
            "throughput_term": float(throughput_term),
            "delay_term": float(delay_term),
            "energy_term": float(energy_term),
            "stability_term": float(stability_term),
            "break_penalty": 0.0,
        }
