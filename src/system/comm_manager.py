from collections import defaultdict
from typing import Dict, List

from src.system.message import Message


class CommManager:
    """
    v1 in-memory communication manager.
    No real network, only simulates message passing and counts communication cost.
    """

    def __init__(self):
        self.mailboxes: Dict[str, List[Message]] = defaultdict(list)
        self.slot_comm_cost = 0
        self.slot_num_messages = 0

    def reset_slot(self):
        self.mailboxes.clear()
        self.slot_comm_cost = 0
        self.slot_num_messages = 0

    def send(self, msg: Message):
        self.mailboxes[msg.receiver].append(msg)
        self.slot_comm_cost += msg.size
        self.slot_num_messages += 1

    def collect_for(self, receiver: str) -> List[Message]:
        return list(self.mailboxes.get(receiver, []))

    def get_slot_stats(self) -> Dict:
        return {
            "comm_cost": self.slot_comm_cost,
            "num_messages": self.slot_num_messages,
        }