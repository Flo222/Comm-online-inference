# src/system/comm_manager.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.system.message import Message

from src.system.channel import ChannelModel


class CommManager:
    """
    Channel-aware communication manager.

    Supports:
    - budget
    - drop
    - delay
    - pending queue
    """

    def __init__(self, channel: ChannelModel):
        self.channel = channel

        # delivered messages available in the current slot
        self.mailboxes: Dict[str, List[Message]] = defaultdict(list)

        # delayed messages waiting for future delivery
        self.pending: Dict[int, List[Message]] = defaultdict(list)

        self.reset_slot_stats()

    def reset_slot_stats(self):
        self.slot_comm_cost_bits = 0
        self.slot_num_messages = 0
        self.slot_delivered_messages = 0
        self.slot_dropped_messages = 0
        self.slot_total_delay = 0
        # new packet-level stats
        self.slot_total_packets = 0
        self.slot_lost_packets = 0
        self.slot_partial_messages = 0

    def clear_mailboxes(self):
        self.mailboxes.clear()

    def deliver_pending(self, current_slot: int):
        due = self.pending.pop(current_slot, [])
        for msg in due:
            self.mailboxes[msg.receiver].append(msg)

    def send(self, msg: Message):
        delivered, recv_slot, out_msg, meta = self.channel.apply(msg)

        self.slot_num_messages += 1
        self.slot_comm_cost_bits += int(out_msg.bit_size)

        total_packets = int(meta.get("total_packets", 0) or 0)
        lost_packets = int(meta.get("lost_packets", 0) or 0)

        self.slot_total_packets += total_packets
        self.slot_lost_packets += lost_packets

        if total_packets > 0 and 0 < lost_packets < total_packets:
            self.slot_partial_messages += 1

        if not delivered:
            self.slot_dropped_messages += 1
            return

        if recv_slot <= out_msg.slot_id:
            self.mailboxes[out_msg.receiver].append(out_msg)
        else:
            self.pending[recv_slot].append(out_msg)

        self.slot_delivered_messages += 1
        self.slot_total_delay += int(out_msg.delay_slots)

    def collect_for(self, receiver: str) -> List[Message]:
        return list(self.mailboxes.get(receiver, []))

    def get_slot_stats(self) -> Dict:
        avg_delay = 0.0
        if self.slot_delivered_messages > 0:
            avg_delay = self.slot_total_delay / self.slot_delivered_messages

        packet_loss_rate = 0.0
        if self.slot_total_packets > 0:
            packet_loss_rate = self.slot_lost_packets / self.slot_total_packets

        return {
            "comm_cost_bits": self.slot_comm_cost_bits,
            "comm_cost_mb": self.slot_comm_cost_bits / 8.0 / 1e6,
            "num_messages": self.slot_num_messages,
            "delivered_messages": self.slot_delivered_messages,
            "dropped_messages": self.slot_dropped_messages,
            "avg_delay_slots": avg_delay,

            # new packet-level stats
            "total_packets": self.slot_total_packets,
            "lost_packets": self.slot_lost_packets,
            "packet_loss_rate": packet_loss_rate,
            "partial_messages": self.slot_partial_messages,
        }