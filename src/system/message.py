from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Message:
    sender: str
    receiver: str
    msg_type: str
    payload: Any
    size: int
    slot_id: int
    timestamp: Optional[float] = None