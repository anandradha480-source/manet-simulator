"""
Node representation for MANET.
Maintains state, mobility, battery, and packet queue.
"""

from typing import Tuple, List, Optional
import numpy as np
from manet.mobility import RandomWaypointMobility
from manet.packet import Packet


class Node:
    """Represents an ad-hoc mobile node in the MANET network."""

    def __init__(
        self,
        node_id: int,
        area_size: Tuple[float, float] = (1000.0, 1000.0),
        speed_range: Tuple[float, float] = (20.0, 50.0),
        initial_battery_joules: float = 100.0,
        queue_capacity: int = 50,
        rng: Optional[np.random.RandomState] = None,
        initial_position: Optional[Tuple[float, float]] = None,
    ):
        self.node_id = node_id
        self.mobility = RandomWaypointMobility(
            node_id=node_id,
            area_size=area_size,
            speed_range=speed_range,
            rng=rng,
            initial_position=initial_position,
        )
        self.max_battery = initial_battery_joules
        self.battery = initial_battery_joules
        self.queue_capacity = queue_capacity
        self.queue: List[Packet] = []

        # Statistics
        self.packets_originated = 0
        self.packets_delivered = 0
        self.packets_forwarded = 0
        self.packets_dropped = 0
        self.energy_consumed = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        return self.mobility.position

    @property
    def x(self) -> float:
        return self.mobility.x

    @property
    def y(self) -> float:
        return self.mobility.y

    @property
    def velocity(self) -> Tuple[float, float]:
        return self.mobility.velocity

    @property
    def speed(self) -> float:
        return self.mobility.speed

    @property
    def battery_ratio(self) -> float:
        """Normalized battery level between [0.0, 1.0]."""
        if self.max_battery <= 0:
            return 0.0
        return float(np.clip(self.battery / self.max_battery, 0.0, 1.0))

    def is_alive(self) -> bool:
        return self.battery > 0.0

    def consume_energy(self, amount: float):
        """Consume energy in Joules."""
        consumed = min(self.battery, amount)
        self.battery -= consumed
        self.energy_consumed += consumed

    def step_mobility(self, dt: float):
        """Update mobility position for time step dt."""
        if self.is_alive():
            self.mobility.step(dt)

    def enqueue_packet(self, packet: Packet) -> bool:
        """Enqueue packet if queue has capacity."""
        if len(self.queue) < self.queue_capacity:
            self.queue.append(packet)
            return True
        else:
            self.packets_dropped += 1
            packet.dropped = True
            packet.drop_reason = "Queue overflow"
            return False

    def dequeue_packet(self) -> Optional[Packet]:
        """Dequeue the head packet."""
        if self.queue:
            return self.queue.pop(0)
        return None

    def reset_stats(self):
        """Reset operational statistics."""
        self.battery = self.max_battery
        self.packets_originated = 0
        self.packets_delivered = 0
        self.packets_forwarded = 0
        self.packets_dropped = 0
        self.energy_consumed = 0.0
        self.queue.clear()
