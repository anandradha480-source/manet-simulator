"""
Packet representations for MANET simulation.
Includes DataPacket and ControlPacket for routing protocols.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Dict


class PacketType(Enum):
    DATA = "DATA"
    RREQ = "RREQ"       # Route Request (AODV)
    RREP = "RREP"       # Route Reply (AODV)
    RERR = "RERR"       # Route Error (AODV)
    BEACON = "BEACON"   # Periodic Hello/Beacon (GPSR/Neighborhood)


@dataclass
class Packet:
    packet_id: int
    packet_type: PacketType
    source: int
    destination: int
    created_at: float
    current_node: int
    size_bytes: int = 512
    hop_count: int = 0
    max_hops: int = 32
    path_history: List[int] = field(default_factory=list)
    
    delivered: bool = False
    dropped: bool = False
    drop_reason: Optional[str] = None
    delivery_time: Optional[float] = None
    total_energy_consumed: float = 0.0
    
    # Optional payload or protocol-specific metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.path_history:
            self.path_history = [self.source]

    @property
    def delay(self) -> Optional[float]:
        if self.delivered and self.delivery_time is not None:
            return self.delivery_time - self.created_at
        return None


@dataclass
class AODVRouteRequest:
    rreq_id: int
    source: int
    dest: int
    source_seq: int
    dest_seq: int
    hop_count: int
    path: List[int] = field(default_factory=list)


@dataclass
class AODVRouteReply:
    source: int
    dest: int
    dest_seq: int
    hop_count: int
    lifetime: float
    path: List[int] = field(default_factory=list)
