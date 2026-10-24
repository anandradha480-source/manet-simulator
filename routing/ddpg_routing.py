"""
DDPG Routing Protocol Wrapper.
Integrates the trained PyTorch DDPG agent as a standard BaseRoutingProtocol.
"""

from typing import Optional, Any
import numpy as np
import torch
from routing.base import BaseRoutingProtocol
from manet.packet import Packet
from rl.agent import DDPGAgent


class DDPGRouting(BaseRoutingProtocol):
    """
    DDPG-based Next-Hop Routing Protocol.
    Uses the trained continuous Actor network to evaluate neighbors and route packets.
    """

    def __init__(
        self,
        agent: DDPGAgent,
        max_neighbors: int = 10,
        features_per_neighbor: int = 7,
    ):
        super().__init__("DDPG-Routing")
        self.agent = agent
        self.max_neighbors = max_neighbors
        self.features_per_neighbor = features_per_neighbor
        self.diag = 1414.21356  # sqrt(1000^2 + 1000^2)

    def reset(self):
        pass

    def select_next_hop(
        self,
        current_node: int,
        destination: int,
        simulator: Any,
        packet: Optional[Packet] = None,
    ) -> Optional[int]:
        neighbors = simulator.get_neighbors(current_node)
        if not neighbors:
            return None

        # 1-hop direct delivery shortcut
        if destination in neighbors:
            return destination

        # Query rich details for neighbors
        details = simulator.get_neighbor_details(current_node, destination)
        details.sort(key=lambda d: d["dist_to_dest"])

        state_matrix = np.zeros((self.max_neighbors, self.features_per_neighbor), dtype=np.float32)
        slot_map = [None] * self.max_neighbors

        for slot_idx in range(min(len(details), self.max_neighbors)):
            d = details[slot_idx]
            slot_map[slot_idx] = d["neighbor_id"]
            norm_dist = np.clip(d["distance"] / simulator.radio_range, 0.0, 1.0)
            norm_speed = np.clip(d["speed"] / simulator.speed_range[1], 0.0, 1.0)
            lq = d["link_quality"]
            battery = d["battery"]
            norm_dest_dist = np.clip(d["dist_to_dest"] / self.diag, 0.0, 1.0)
            norm_let = np.clip(d["let"] / 10.0, 0.0, 1.0)
            mask = 1.0

            state_matrix[slot_idx] = [
                norm_dist,
                norm_speed,
                lq,
                battery,
                norm_dest_dist,
                norm_let,
                mask,
            ]

        flat_state = state_matrix.flatten()

        # DDPG Actor scores (no noise during inference)
        scores = self.agent.select_action(flat_state, add_noise=False)

        # Mask empty slots
        masked_scores = np.copy(scores)
        valid_slots = [idx for idx, nid in enumerate(slot_map) if nid is not None]

        if not valid_slots:
            return None

        for idx in range(self.max_neighbors):
            if idx not in valid_slots:
                masked_scores[idx] = -float("inf")

        best_slot = int(np.argmax(masked_scores))
        chosen_neighbor = slot_map[best_slot]
        return chosen_neighbor
