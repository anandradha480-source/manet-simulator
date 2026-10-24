"""
GPSR (Greedy Perimeter Stateless Routing) Protocol.
Combines greedy distance-progress forwarding with planar perimeter routing for local minima recovery.
"""

import numpy as np
from typing import Optional, List, Tuple, Any, Set
from routing.base import BaseRoutingProtocol
from manet.packet import Packet


class GPSRRouting(BaseRoutingProtocol):
    """
    Greedy Perimeter Stateless Routing (GPSR).
    Uses greedy geographic forwarding towards destination.
    When trapped in a local minimum (no neighbor is closer to destination),
    falls back to perimeter forwarding over a planar Gabriel Graph (GG)
    using the Right-Hand Rule until closer progress is resumed.
    """

    def __init__(self):
        super().__init__("GPSR")

    def reset(self):
        pass

    def _is_gabriel_edge(self, u: int, v: int, simulator: Any) -> bool:
        """
        Tests whether edge (u, v) belongs to the planar Gabriel Graph (GG).
        Edge exists in GG iff the midpoint circle contains no other mutual neighbor w.
        Condition: dist(u, w)^2 + dist(v, w)^2 >= dist(u, v)^2 for all mutual neighbors w.
        """
        dist_uv_sq = simulator.distance_matrix[u, v] ** 2
        u_neighbors = simulator.get_neighbors(u)
        v_neighbors = simulator.get_neighbors(v)
        mutual = set(u_neighbors).intersection(set(v_neighbors))

        for w in mutual:
            dist_uw_sq = simulator.distance_matrix[u, w] ** 2
            dist_vw_sq = simulator.distance_matrix[v, w] ** 2
            if dist_uw_sq + dist_vw_sq < dist_uv_sq - 1e-4:
                return False
        return True

    def _get_planar_neighbors(self, u: int, simulator: Any) -> List[int]:
        """Returns the subset of neighbors forming planar Gabriel Graph edges."""
        neighbors = simulator.get_neighbors(u)
        return [v for v in neighbors if self._is_gabriel_edge(u, v, simulator)]

    def _angle(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Angle of vector p1 -> p2 in radians [0, 2pi)."""
        rad = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        return float(rad % (2 * np.pi))

    def _select_perimeter_hop(
        self,
        curr: int,
        dest: int,
        ref_point: Tuple[float, float],
        simulator: Any,
        visited_hops: Set[int],
    ) -> Optional[int]:
        """
        Applies Right-Hand Rule on planar Gabriel Graph.
        Sweeps counterclockwise from ref_point to find the next planar neighbor.
        """
        planar_neighbors = self._get_planar_neighbors(curr, simulator)
        if not planar_neighbors:
            # Fall back to all neighbors if planar graph is isolated
            planar_neighbors = simulator.get_neighbors(curr)
        if not planar_neighbors:
            return None

        curr_pos = simulator.nodes[curr].position
        ref_angle = self._angle(curr_pos, ref_point)

        best_next = None
        min_sweep_angle = float("inf")

        for v in planar_neighbors:
            v_pos = simulator.nodes[v].position
            v_angle = self._angle(curr_pos, v_pos)
            # Counterclockwise sweep angle from ref_angle
            sweep = (v_angle - ref_angle) % (2 * np.pi)
            if sweep < 1e-5:
                sweep = 2 * np.pi

            if sweep < min_sweep_angle:
                min_sweep_angle = sweep
                best_next = v

        return best_next

    def select_next_hop(
        self,
        current_node: int,
        destination: int,
        simulator: Any,
        packet: Optional[Packet] = None,
    ) -> Optional[int]:
        curr_pos = simulator.nodes[current_node].position
        dest_pos = simulator.nodes[destination].position
        curr_dist_to_dest = float(simulator.distance_matrix[current_node, destination])

        neighbors = simulator.get_neighbors(current_node)
        if not neighbors:
            return None

        if destination in neighbors:
            return destination

        # 1. Greedy Mode: Find neighbor closest to destination
        best_neighbor = None
        min_dist_to_dest = curr_dist_to_dest

        for v in neighbors:
            v_dist = float(simulator.distance_matrix[v, destination])
            if v_dist < min_dist_to_dest:
                min_dist_to_dest = v_dist
                best_neighbor = v

        if packet is not None:
            # Check packet routing mode
            mode = packet.meta.get("gpsr_mode", "GREEDY")
            lp_dist = packet.meta.get("gpsr_lp_dist", curr_dist_to_dest)

            if mode == "PERIMETER":
                # Exit perimeter mode if strictly closer to destination than entry point
                if curr_dist_to_dest < lp_dist:
                    packet.meta["gpsr_mode"] = "GREEDY"
                    mode = "GREEDY"

            if mode == "GREEDY":
                if best_neighbor is not None:
                    return best_neighbor
                else:
                    # Trapped in local minimum -> switch to PERIMETER mode
                    packet.meta["gpsr_mode"] = "PERIMETER"
                    packet.meta["gpsr_lp_dist"] = curr_dist_to_dest
                    packet.meta["gpsr_perimeter_visited"] = []

            # In PERIMETER mode
            visited_list = packet.meta.get("gpsr_perimeter_visited", [])
            visited_list.append(current_node)
            packet.meta["gpsr_perimeter_visited"] = visited_list

            # Reference point: previous hop or destination direction
            prev_hop = packet.path_history[-2] if len(packet.path_history) >= 2 else None
            ref_pos = simulator.nodes[prev_hop].position if prev_hop is not None else dest_pos

            next_hop = self._select_perimeter_hop(
                current_node, destination, ref_pos, simulator, set(visited_list)
            )
            return next_hop
        else:
            # Default greedy choice
            return best_neighbor
