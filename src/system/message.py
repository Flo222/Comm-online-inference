# src/system/message.py
from dataclasses import dataclass, field
from typing import Any, Optional, Dict


@dataclass
class Message:
    sender: str
    receiver: str
    msg_type: str
    payload: Any

    # old fields
    size: int
    slot_id: int
    timestamp: Optional[float] = None

    # new fields
    bit_size: int = 0
    send_slot: int = -1
    recv_slot: int = -1
    delivered: bool = False
    dropped: bool = False
    drop_reason: Optional[str] = None
    delay_slots: int = 0
    link_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> int:
        if self.recv_slot < 0 or self.send_slot < 0:
            return 0
        return self.recv_slot - self.send_slot