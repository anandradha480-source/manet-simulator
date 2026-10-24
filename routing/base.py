"""
Base interface for MANET Routing Protocols.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from manet.packet import Packet


class BaseRoutingProtocol(ABC):
    """Abstract base class for all routing algorithms in the MANET simulator."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_next_hop(
        self,
        current_node: int,
        destination: int,
        simulator: Any,
        packet: Optional[Packet] = None,
    ) -> Optional[int]:
        """
        Determines the next hop node for forwarding a packet towards destination.
        Returns None if no route or next hop is available.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset routing state between simulation episodes."""
        pass

    def route_packet(self, packet: Packet, simulator: Any) -> bool:
        """
        Routes packet hop-by-hop until delivery, drop, or max_hops.
        Returns True if delivered, False otherwise.
        """
        while not packet.delivered and not packet.dropped:
            curr = packet.current_node
            dest = packet.destination

            if curr == dest:
                packet.delivered = True
                return True

            next_hop = self.select_next_hop(curr, dest, simulator, packet)

            if next_hop is None:
                packet.dropped = True
                packet.drop_reason = f"No route found at node {curr}"
                simulator.packets_dropped.append(packet)
                return False

            success, metrics = simulator.transmit_hop(packet, next_hop)
            if not success:
                return False

        return packet.delivered
