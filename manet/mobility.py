"""
Random Waypoint Mobility Model for MANET.
Nodes move at velocities between speed_min (20 m/s) and speed_max (50 m/s).
"""

import numpy as np
from typing import Tuple, Optional


class RandomWaypointMobility:
    """
    Random Waypoint (RWP) mobility model.
    In RWP:
    - A node picks a random destination within the simulation area.
    - Selects a speed uniformly between [speed_min, speed_max].
    - Moves in a straight line toward the waypoint.
    - Upon arrival, pauses for pause_time seconds.
    - Picks a new destination and speed, repeating the cycle.
    """

    def __init__(
        self,
        node_id: int,
        area_size: Tuple[float, float] = (1000.0, 1000.0),
        speed_range: Tuple[float, float] = (20.0, 50.0),
        pause_range: Tuple[float, float] = (0.0, 1.0),
        rng: Optional[np.random.RandomState] = None,
        initial_position: Optional[Tuple[float, float]] = None,
    ):
        self.node_id = node_id
        self.area_width, self.area_height = area_size
        self.speed_min, self.speed_max = speed_range
        self.pause_min, self.pause_max = pause_range
        self.rng = rng if rng is not None else np.random.RandomState()

        # Initial position
        if initial_position is not None:
            self.x = float(initial_position[0])
            self.y = float(initial_position[1])
        else:
            self.x = float(self.rng.uniform(0.0, self.area_width))
            self.y = float(self.rng.uniform(0.0, self.area_height))

        self.waypoint_x: float = self.x
        self.waypoint_y: float = self.y
        self.speed: float = 0.0
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.pause_time_remaining: float = 0.0

        # Pick initial destination and speed
        self._pick_new_waypoint()

    def _pick_new_waypoint(self):
        """Pick a new random destination within the area bounds and a random speed."""
        self.waypoint_x = float(self.rng.uniform(0.0, self.area_width))
        self.waypoint_y = float(self.rng.uniform(0.0, self.area_height))
        self.speed = float(self.rng.uniform(self.speed_min, self.speed_max))

        dx = self.waypoint_x - self.x
        dy = self.waypoint_y - self.y
        dist = np.hypot(dx, dy)

        if dist > 1e-5:
            self.vx = float((dx / dist) * self.speed)
            self.vy = float((dy / dist) * self.speed)
        else:
            self.vx = 0.0
            self.vy = 0.0

    def step(self, dt: float):
        """Update node position over time step dt (in seconds)."""
        if self.pause_time_remaining > 0:
            self.pause_time_remaining -= dt
            if self.pause_time_remaining <= 0:
                self.pause_time_remaining = 0.0
                self._pick_new_waypoint()
            return

        dx = self.waypoint_x - self.x
        dy = self.waypoint_y - self.y
        dist = np.hypot(dx, dy)
        step_distance = self.speed * dt

        if dist <= step_distance:
            # Reached waypoint
            self.x = self.waypoint_x
            self.y = self.waypoint_y
            self.pause_time_remaining = float(self.rng.uniform(self.pause_min, self.pause_max))
            if self.pause_time_remaining <= 0:
                self._pick_new_waypoint()
            else:
                self.vx = 0.0
                self.vy = 0.0
        else:
            # Move towards waypoint
            self.x += self.vx * dt
            self.y += self.vy * dt

        # Safety boundary clamp
        self.x = np.clip(self.x, 0.0, self.area_width)
        self.y = np.clip(self.y, 0.0, self.area_height)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        return (self.vx, self.vy)
