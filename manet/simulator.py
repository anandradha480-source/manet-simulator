"""
MANET Simulator.
30 mobile nodes moving with Random Waypoint at 20-50 m/s with 250m radio range.
Provides neighbor discovery, packet generation, and metrics tracking (PDR, delay).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from manet.node import Node
from manet.packet import Packet, PacketType
from manet.channel import WirelessChannel


class MANETSimulator:
    """
    MANET Simulator environment.
    Simulates 30 mobile nodes moving in a 2D terrain with Random Waypoint mobility.
    """

    def __init__(
        self,
        num_nodes: int = 30,
        area_size: Tuple[float, float] = (1000.0, 1000.0),
        radio_range: float = 250.0,
        speed_range: Tuple[float, float] = (20.0, 50.0),
        dt: float = 0.1,  # Simulation step size in seconds
        initial_battery: float = 100.0,
        seed: Optional[int] = None,
    ):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.radio_range = radio_range
        self.speed_range = speed_range
        self.dt = dt
        self.initial_battery = initial_battery
        self.seed = seed

        self.rng = np.random.RandomState(seed)
        self.channel = WirelessChannel(radio_range=radio_range)
        
        self.current_time: float = 0.0
        self.packet_counter: int = 0
        
        # Initialize nodes
        self.nodes: List[Node] = []
        for i in range(num_nodes):
            node = Node(
                node_id=i,
                area_size=area_size,
                speed_range=speed_range,
                initial_battery_joules=initial_battery,
                rng=self.rng,
            )
            self.nodes.append(node)

        # Distance matrix and neighbor cache
        self.distance_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        self.neighbors_cache: Dict[int, List[int]] = {}
        self._update_topology()

        # Global statistics
        self.packets_generated: List[Packet] = []
        self.packets_delivered: List[Packet] = []
        self.packets_dropped: List[Packet] = []
        self.link_breaks_count: int = 0

    def reset(self, seed: Optional[int] = None):
        """Reset the simulator to initial state, optionally with a new seed."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.RandomState(seed)

        self.current_time = 0.0
        self.packet_counter = 0
        self.packets_generated.clear()
        self.packets_delivered.clear()
        self.packets_dropped.clear()
        self.link_breaks_count = 0

        self.nodes.clear()
        for i in range(self.num_nodes):
            node = Node(
                node_id=i,
                area_size=self.area_size,
                speed_range=self.speed_range,
                initial_battery_joules=self.initial_battery,
                rng=self.rng,
            )
            self.nodes.append(node)

        self._update_topology()

    def _update_topology(self):
        """Compute pairwise distances and update neighbors for all nodes within radio_range."""
        positions = np.array([node.position for node in self.nodes], dtype=np.float64)
        # Pairwise Euclidean distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        self.distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        self.neighbors_cache.clear()
        for i in range(self.num_nodes):
            # Node i's neighbors are those within radio_range (excluding itself)
            neighbors = [
                j for j in range(self.num_nodes)
                if i != j and self.distance_matrix[i, j] <= self.radio_range and self.nodes[j].is_alive()
            ]
            self.neighbors_cache[i] = neighbors

    def get_neighbors(self, node_id: int) -> List[int]:
        """
        Lists node_id's neighbors within radio_range (250m).
        Required function that lists each node's neighbors.
        """
        if node_id not in self.neighbors_cache:
            return []
        return list(self.neighbors_cache[node_id])

    def get_all_neighbors(self) -> Dict[int, List[int]]:
        """Returns neighbor list dictionary for all nodes."""
        return {i: list(self.neighbors_cache.get(i, [])) for i in range(self.num_nodes)}

    def get_distance(self, node_a: int, node_b: int) -> float:
        """Get distance between two nodes."""
        return float(self.distance_matrix[node_a, node_b])

    def get_neighbor_details(self, node_id: int, dest_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Returns rich state features for each neighbor of node_id:
        - neighbor_id
        - distance: distance between node_id and neighbor (m)
        - dist_to_dest: distance from neighbor to destination (m)
        - speed: neighbor's speed (m/s)
        - link_quality: normalized link quality [0.0, 1.0]
        - battery: neighbor's normalized battery ratio [0.0, 1.0]
        - let: Link Expiration Time (seconds)
        """
        neighbors = self.get_neighbors(node_id)
        u_node = self.nodes[node_id]
        details = []

        for v_id in neighbors:
            v_node = self.nodes[v_id]
            dist_uv = float(self.distance_matrix[node_id, v_id])
            dist_dest = float(self.distance_matrix[v_id, dest_id]) if dest_id is not None else 0.0
            lq = self.channel.get_link_quality(dist_uv)
            let = self.channel.calculate_link_expiration_time(
                u_node.position, u_node.velocity,
                v_node.position, v_node.velocity,
                self.radio_range
            )

            details.append({
                "neighbor_id": v_id,
                "distance": dist_uv,
                "dist_to_dest": dist_dest,
                "speed": v_node.speed,
                "link_quality": lq,
                "battery": v_node.battery_ratio,
                "let": let,
                "position": v_node.position,
                "velocity": v_node.velocity,
            })
        return details

    def generate_packet(self, source: Optional[int] = None, destination: Optional[int] = None) -> Optional[Packet]:
        """Generate a new data packet between distinct alive nodes."""
        alive_nodes = [i for i, node in enumerate(self.nodes) if node.is_alive()]
        if len(alive_nodes) < 2:
            return None

        if source is None:
            source = int(self.rng.choice(alive_nodes))
        
        possible_dests = [i for i in alive_nodes if i != source]
        if not possible_dests:
            return None

        if destination is None:
            destination = int(self.rng.choice(possible_dests))

        self.packet_counter += 1
        packet = Packet(
            packet_id=self.packet_counter,
            packet_type=PacketType.DATA,
            source=source,
            destination=destination,
            created_at=self.current_time,
            current_node=source,
        )
        self.packets_generated.append(packet)
        self.nodes[source].packets_originated += 1
        return packet

    def step_mobility(self):
        """Advance time and update node mobility and network topology."""
        self.current_time += self.dt
        for node in self.nodes:
            node.step_mobility(self.dt)
        self._update_topology()

    def transmit_hop(self, packet: Packet, next_hop: int) -> Tuple[bool, Dict[str, float]]:
        """
        Attempt to transmit a packet from packet.current_node to next_hop.
        Returns:
            success (bool): whether transmission succeeded
            metrics (dict): delay, energy, link_break (0 or 1), stability
        """
        curr = packet.current_node
        curr_node = self.nodes[curr]
        next_node = self.nodes[next_hop]

        dist = float(self.distance_matrix[curr, next_hop])
        in_range = dist <= self.radio_range and next_node.is_alive()

        # Calculate link stability (LET)
        let = self.channel.calculate_link_expiration_time(
            curr_node.position, curr_node.velocity,
            next_node.position, next_node.velocity,
            self.radio_range
        )

        # Transmission and reception energy
        tx_energy = self.channel.calculate_transmission_energy(dist, packet.size_bytes)
        rx_energy = self.channel.calculate_reception_energy(packet.size_bytes)
        total_hop_energy = tx_energy + rx_energy

        # Delay
        hop_delay = self.channel.calculate_hop_delay(dist, packet.size_bytes)

        # Check if link broke
        if not in_range:
            self.link_breaks_count += 1
            curr_node.consume_energy(tx_energy * 0.5)  # partially spent trying to transmit
            packet.dropped = True
            packet.drop_reason = f"Link break between {curr} and {next_hop}"
            self.packets_dropped.append(packet)
            return False, {
                "delay": hop_delay,
                "energy": tx_energy * 0.5,
                "link_break": 1.0,
                "let": let,
                "dist": dist,
            }

        # Energy consumption
        curr_node.consume_energy(tx_energy)
        next_node.consume_energy(rx_energy)
        packet.total_energy_consumed += total_hop_energy

        # Advance packet
        packet.hop_count += 1
        packet.current_node = next_hop
        packet.path_history.append(next_hop)

        # Check delivery
        if next_hop == packet.destination:
            packet.delivered = True
            packet.delivery_time = self.current_time + hop_delay
            self.packets_delivered.append(packet)
            next_node.packets_delivered += 1
            return True, {
                "delay": hop_delay,
                "energy": total_hop_energy,
                "link_break": 0.0,
                "let": let,
                "dist": dist,
                "delivered": 1.0,
            }
        else:
            curr_node.packets_forwarded += 1
            # Check max hops TTL
            if packet.hop_count >= packet.max_hops:
                packet.dropped = True
                packet.drop_reason = "TTL exceeded"
                self.packets_dropped.append(packet)
                return False, {
                    "delay": hop_delay,
                    "energy": total_hop_energy,
                    "link_break": 0.0,
                    "let": let,
                    "dist": dist,
                }
            return True, {
                "delay": hop_delay,
                "energy": total_hop_energy,
                "link_break": 0.0,
                "let": let,
                "dist": dist,
                "delivered": 0.0,
            }

    @property
    def packet_delivery_ratio(self) -> float:
        """Packet Delivery Ratio (PDR) in percentage [0.0, 100.0]."""
        if not self.packets_generated:
            return 0.0
        return float((len(self.packets_delivered) / len(self.packets_generated)) * 100.0)

    @property
    def average_delay(self) -> float:
        """Average end-to-end delay (in seconds) for delivered packets."""
        delays = [p.delay for p in self.packets_delivered if p.delay is not None]
        if not delays:
            return 0.0
        return float(np.mean(delays))

    @property
    def total_energy(self) -> float:
        """Total energy consumed across all nodes (Joules)."""
        return float(sum(n.energy_consumed for n in self.nodes))

    def get_metrics(self) -> Dict[str, float]:
        """Summary metrics dictionary."""
        return {
            "pdr": self.packet_delivery_ratio,
            "delay": self.average_delay,
            "total_generated": float(len(self.packets_generated)),
            "total_delivered": float(len(self.packets_delivered)),
            "total_dropped": float(len(self.packets_dropped)),
            "link_breaks": float(self.link_breaks_count),
            "total_energy": self.total_energy,
        }
