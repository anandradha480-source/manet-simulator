"""
Wireless channel, energy model, and link metrics for MANET.
Radio range is 250 m.
"""

import numpy as np
from typing import Tuple


class WirelessChannel:
    """
    Simulates the wireless propagation channel, transmission delay,
    energy consumption, and link quality for nodes separated by distance d.
    """

    def __init__(
        self,
        radio_range: float = 250.0,
        bandwidth_bps: float = 2e6,       # 2 Mbps channel
        tx_power_watt: float = 0.2818,     # ~24.5 dBm
        rx_power_watt: float = 0.140,      # ~21.5 dBm
        e_elec: float = 50e-9,             # 50 nJ/bit
        e_amp: float = 100e-12,            # 100 pJ/bit/m^2
        path_loss_exponent: float = 2.0,
    ):
        self.radio_range = radio_range
        self.bandwidth_bps = bandwidth_bps
        self.tx_power_watt = tx_power_watt
        self.rx_power_watt = rx_power_watt
        self.e_elec = e_elec
        self.e_amp = e_amp
        self.path_loss_exponent = path_loss_exponent
        self.speed_of_light = 3e8  # m/s

    def is_in_range(self, dist: float) -> bool:
        """Check if two nodes are within the 250 m radio range."""
        return dist <= self.radio_range

    def get_link_quality(self, dist: float) -> float:
        """
        Calculates normalized link quality [0.0, 1.0].
        At d=0, LQ=1.0; at d=250m, LQ approaches 0.0.
        Uses quadratic distance decay + signal margin.
        """
        if dist > self.radio_range:
            return 0.0
        # Quadratic decay based on radio range
        ratio = dist / self.radio_range
        lq = max(0.0, 1.0 - (ratio ** self.path_loss_exponent))
        return float(np.clip(lq, 0.0, 1.0))

    def calculate_transmission_energy(self, dist: float, packet_size_bytes: int) -> float:
        """
        First-order radio energy model:
        E_tx = l * E_elec + l * E_amp * (d ^ 2)
        Returns energy consumed in Joules.
        """
        bits = packet_size_bytes * 8
        e_tx = bits * self.e_elec + bits * self.e_amp * (dist ** 2)
        return float(e_tx)

    def calculate_reception_energy(self, packet_size_bytes: int) -> float:
        """E_rx = l * E_elec in Joules."""
        bits = packet_size_bytes * 8
        return float(bits * self.e_elec)

    def calculate_hop_delay(self, dist: float, packet_size_bytes: int, queuing_delay: float = 0.0) -> float:
        """
        Calculates total delay for a single hop:
        t_total = t_tx + t_prop + t_proc + t_queue
        Returns delay in seconds.
        """
        bits = packet_size_bytes * 8
        t_tx = bits / self.bandwidth_bps
        t_prop = dist / self.speed_of_light
        t_proc = 5e-5  # 50 microseconds processing time
        return float(t_tx + t_prop + t_proc + queuing_delay)

    @staticmethod
    def calculate_link_expiration_time(
        pos_i: Tuple[float, float],
        vel_i: Tuple[float, float],
        pos_j: Tuple[float, float],
        vel_j: Tuple[float, float],
        radio_range: float = 250.0,
    ) -> float:
        """
        Computes Link Expiration Time (LET) in seconds using standard kinematic formula.
        Predicts how long node i and node j remain connected within radio_range.
        """
        x_i, y_i = pos_i
        v_ix, v_iy = vel_i
        x_j, y_j = pos_j
        v_jx, v_jy = vel_j

        a = v_ix - v_jx
        b = x_i - x_j
        c = v_iy - v_jy
        d = y_i - y_j

        dist_sq = b * b + d * d
        if dist_sq > radio_range * radio_range:
            return 0.0

        denom = a * a + c * c
        if denom < 1e-6:
            # Nodes moving at almost identical velocities
            return 60.0  # Stable link

        discriminant = (a * a + c * c) * (radio_range * radio_range) - (a * d - b * c) ** 2
        if discriminant < 0:
            return 0.0

        let = (-(a * b + c * d) + np.sqrt(discriminant)) / denom
        return float(max(0.0, let))
