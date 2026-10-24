"""
Gym-like MANET RL Environment for DDPG Next-Hop Routing.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from manet.simulator import MANETSimulator
from manet.packet import Packet
from rl.reward import MANETRewardCalculator


class MANETRoutingEnv:
    """
    MANET Next-Hop Routing Environment.
    State: candidate neighbor features (distance, speed, link quality, battery, dest distance, LET).
    Action: continuous scores for each candidate neighbor slot.
    Selection: slot with the highest score is chosen as the next hop.
    """

    def __init__(
        self,
        simulator: MANETSimulator,
        max_neighbors: int = 10,
        features_per_neighbor: int = 7,
    ):
        self.sim = simulator
        self.max_neighbors = max_neighbors
        self.features_per_neighbor = features_per_neighbor
        self.state_dim = max_neighbors * features_per_neighbor
        self.action_dim = max_neighbors

        self.reward_calculator = MANETRewardCalculator()
        self.diag = float(np.hypot(self.sim.area_size[0], self.sim.area_size[1]))

        # State tracking for active packet routing
        self.active_packet: Optional[Packet] = None
        self.slot_to_neighbor_id: List[Optional[int]] = [None] * max_neighbors

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset simulator and generate a new routing task."""
        self.sim.reset(seed=seed)
        self.active_packet = None
        return self._prepare_new_packet_state()

    def _prepare_new_packet_state(self) -> np.ndarray:
        """Generates a packet and advances mobility until a valid packet is in transit."""
        max_attempts = 50
        for _ in range(max_attempts):
            self.sim.step_mobility()
            pkt = self.sim.generate_packet()
            if pkt is not None:
                neighbors = self.sim.get_neighbors(pkt.current_node)
                if neighbors:
                    self.active_packet = pkt
                    state, _ = self.get_state_for_node(pkt.current_node, pkt.destination)
                    return state

        # Fallback state if no neighbors found
        state = np.zeros(self.state_dim, dtype=np.float32)
        return state

    def get_state_for_node(
        self, current_node: int, destination: int
    ) -> Tuple[np.ndarray, List[Optional[int]]]:
        """
        Builds the normalized state vector for current_node towards destination.
        Returns:
            state_vector: [state_dim] flattened float array
            slot_map: list of neighbor node IDs corresponding to each slot (or None)
        """
        neighbors = self.sim.get_neighbors(current_node)
        # Prioritize neighbors closer to destination or random shuffle
        details = self.sim.get_neighbor_details(current_node, destination)

        # Sort candidate neighbors by distance to destination for consistency
        details.sort(key=lambda d: d["dist_to_dest"])

        state_matrix = np.zeros((self.max_neighbors, self.features_per_neighbor), dtype=np.float32)
        slot_map: List[Optional[int]] = [None] * self.max_neighbors

        for slot_idx in range(min(len(details), self.max_neighbors)):
            d = details[slot_idx]
            v_id = d["neighbor_id"]
            slot_map[slot_idx] = v_id

            # State features:
            # 1. Neighbor distance (normalized by radio range 250m)
            norm_dist = np.clip(d["distance"] / self.sim.radio_range, 0.0, 1.0)
            # 2. Speed (normalized by speed_max 50m/s)
            norm_speed = np.clip(d["speed"] / self.sim.speed_range[1], 0.0, 1.0)
            # 3. Link quality [0.0, 1.0]
            lq = d["link_quality"]
            # 4. Battery ratio [0.0, 1.0]
            battery = d["battery"]
            # 5. Distance from neighbor to destination [0.0, 1.0]
            norm_dest_dist = np.clip(d["dist_to_dest"] / self.diag, 0.0, 1.0)
            # 6. Route stability (LET normalized by 10s)
            norm_let = np.clip(d["let"] / 10.0, 0.0, 1.0)
            # 7. Valid slot mask
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

        self.slot_to_neighbor_id = slot_map
        return state_matrix.flatten(), slot_map

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes action:
        - action is a continuous score vector for candidate neighbor slots.
        - the valid slot with the highest score is chosen as the next hop.
        - advances packet by one hop and updates simulator.
        Returns:
            next_state: np.ndarray
            reward: float
            done: bool
            info: dict
        """
        if self.active_packet is None or self.active_packet.delivered or self.active_packet.dropped:
            next_state = self._prepare_new_packet_state()
            return next_state, 0.0, True, {"status": "packet_completed"}

        pkt = self.active_packet
        curr = pkt.current_node
        dest = pkt.destination
        curr_dist_to_dest = float(self.sim.distance_matrix[curr, dest])

        # Mask invalid slots (slots without active neighbor)
        masked_scores = np.copy(action)
        valid_slots = [idx for idx, nid in enumerate(self.slot_to_neighbor_id) if nid is not None]

        if not valid_slots:
            # Node has no neighbors (isolated in dynamic topology)
            reward_info = self.reward_calculator.compute_reward(
                curr_dist_to_dest=curr_dist_to_dest,
                next_dist_to_dest=curr_dist_to_dest,
                hop_delay=0.01,
                energy_consumed=0.0,
                link_expiration_time=0.0,
                is_link_break=True,
                is_delivered=False,
                radio_range=self.sim.radio_range,
            )
            pkt.dropped = True
            pkt.drop_reason = "No neighbors available"
            self.sim.packets_dropped.append(pkt)
            next_state = self._prepare_new_packet_state()
            return next_state, reward_info["reward"], True, {"reason": "no_neighbors"}

        for idx in range(self.max_neighbors):
            if idx not in valid_slots:
                masked_scores[idx] = -float("inf")

        # Highest score is the next hop
        best_slot = int(np.argmax(masked_scores))
        chosen_neighbor = self.slot_to_neighbor_id[best_slot]

        # Mobility step advances simulation time
        self.sim.step_mobility()

        # Transmit hop to chosen neighbor
        success, metrics = self.sim.transmit_hop(pkt, chosen_neighbor)

        next_dist_to_dest = float(self.sim.distance_matrix[chosen_neighbor, dest])
        is_delivered = pkt.delivered
        is_link_break = bool(metrics["link_break"] > 0) or pkt.dropped

        # Compute composite reward
        reward_dict = self.reward_calculator.compute_reward(
            curr_dist_to_dest=curr_dist_to_dest,
            next_dist_to_dest=next_dist_to_dest,
            hop_delay=metrics["delay"],
            energy_consumed=metrics["energy"],
            link_expiration_time=metrics["let"],
            is_link_break=is_link_break,
            is_delivered=is_delivered,
            radio_range=self.sim.radio_range,
        )

        done = is_delivered or is_link_break or pkt.dropped

        if not done:
            # Next state at new hop
            next_state, _ = self.get_state_for_node(pkt.current_node, pkt.destination)
        else:
            # Prepare next packet
            next_state = self._prepare_new_packet_state()

        info = {
            "chosen_neighbor": chosen_neighbor,
            "metrics": metrics,
            "reward_breakdown": reward_dict,
            "delivered": is_delivered,
            "link_break": is_link_break,
        }

        return next_state, reward_dict["reward"], done, info
