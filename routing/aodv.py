"""
AODV (Ad hoc On-Demand Distance Vector) Routing Protocol.
RFC 3561 inspired implementation for dynamic mobile ad-hoc networks.
"""

from typing import Dict, Optional, List, Tuple, Any
from collections import deque
from routing.base import BaseRoutingProtocol
from manet.packet import Packet


class AODVRouteEntry:
    def __init__(self, dest: int, next_hop: int, hop_count: int, seq_num: int, expiration: float):
        self.dest = dest
        self.next_hop = next_hop
        self.hop_count = hop_count
        self.seq_num = seq_num
        self.expiration = expiration
        self.is_valid = True


class AODVRouting(BaseRoutingProtocol):
    """
    AODV Routing Protocol.
    Maintains a routing table per node. When a route is missing or expired,
    a route discovery process (RREQ / RREP) is initiated over the ad-hoc topology.
    """

    def __init__(self, route_lifetime: float = 3.0):
        super().__init__("AODV")
        self.route_lifetime = route_lifetime
        # routing_tables[node_id][dest_id] -> AODVRouteEntry
        self.routing_tables: Dict[int, Dict[int, AODVRouteEntry]] = {}
        # sequence numbers per node
        self.seq_numbers: Dict[int, int] = {}
        # broadcast counter
        self.broadcast_id: int = 0

    def reset(self):
        self.routing_tables.clear()
        self.seq_numbers.clear()
        self.broadcast_id = 0

    def _get_seq_num(self, node_id: int) -> int:
        if node_id not in self.seq_numbers:
            self.seq_numbers[node_id] = 1
        return self.seq_numbers[node_id]

    def _inc_seq_num(self, node_id: int) -> int:
        self.seq_numbers[node_id] = self._get_seq_num(node_id) + 1
        return self.seq_numbers[node_id]

    def _discover_route(self, source: int, dest: int, simulator: Any) -> bool:
        """
        Simulate AODV RREQ flood and reverse path RREP return.
        Explores current reachable topology via BFS.
        """
        self.broadcast_id += 1
        now = simulator.current_time

        # Queue for RREQ flood: (current_node, hop_count, path_from_source)
        queue = deque([(source, 0, [source])])
        visited = {source: 0}
        reverse_path: Dict[int, int] = {}  # node -> previous_hop towards source

        found_dest = False
        final_path: List[int] = []

        while queue:
            curr, hops, path = queue.popleft()

            if curr == dest:
                found_dest = True
                final_path = path
                break

            if hops >= 16:  # RREQ hop limit
                continue

            for neighbor in simulator.get_neighbors(curr):
                if neighbor not in visited or visited[neighbor] > hops + 1:
                    visited[neighbor] = hops + 1
                    reverse_path[neighbor] = curr
                    queue.append((neighbor, hops + 1, path + [neighbor]))

        if not found_dest:
            return False

        # RREP unicast back along the reverse path:
        # Establish forward routing table entries along final_path:
        # final_path is [source, n1, n2, ..., dest]
        dest_seq = self._inc_seq_num(dest)
        total_hops = len(final_path) - 1

        for idx in range(len(final_path) - 1):
            u = final_path[idx]
            next_hop = final_path[idx + 1]
            remaining_hops = total_hops - idx

            if u not in self.routing_tables:
                self.routing_tables[u] = {}

            self.routing_tables[u][dest] = AODVRouteEntry(
                dest=dest,
                next_hop=next_hop,
                hop_count=remaining_hops,
                seq_num=dest_seq,
                expiration=now + self.route_lifetime,
            )

        return True

    def select_next_hop(
        self,
        current_node: int,
        destination: int,
        simulator: Any,
        packet: Optional[Packet] = None,
    ) -> Optional[int]:
        now = simulator.current_time

        # If destination is a direct 1-hop neighbor
        neighbors = simulator.get_neighbors(current_node)
        if destination in neighbors:
            return destination

        # Check existing route entry
        node_table = self.routing_tables.get(current_node, {})
        entry = node_table.get(destination)

        # Validate entry: must be valid, not expired, and next_hop must still be within radio range
        if entry is not None and entry.is_valid and entry.expiration > now:
            if entry.next_hop in neighbors:
                return entry.next_hop
            else:
                # Link broke! Next hop moved out of range
                entry.is_valid = False

        # If no valid route, trigger route discovery
        discovered = self._discover_route(current_node, destination, simulator)
        if discovered:
            entry = self.routing_tables.get(current_node, {}).get(destination)
            if entry and entry.next_hop in simulator.get_neighbors(current_node):
                return entry.next_hop

        return None
