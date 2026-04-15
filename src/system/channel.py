# src/system/channel.py
from __future__ import annotations

import random
import torch


class ChannelModel:
    """
    Simple channel model aligned with current main_base.py args:
    - comm_budget_mb
    - comm_delay_ms
    - comm_drop_prob
    - slot_ms
    - comm_bits_per_value
    """

    def __init__(
        self,
        budget_mb: float,
        delay_ms: int,
        drop_prob: float,
        slot_ms: int = 100,
        bits_per_value: int = 16,
        budget_policy: str = "prefix",
        rng_seed: int | None = None,
    ):
        self.budget_mb = float(budget_mb)
        self.delay_ms = int(delay_ms)
        self.drop_prob = float(drop_prob)
        self.slot_ms = int(slot_ms)
        self.bits_per_value = int(bits_per_value)
        self.budget_policy = budget_policy
        self.rng = random.Random(rng_seed)

        self.delay_slots = max(self.delay_ms // self.slot_ms, 0)

    def apply(self, msg):
        """
        Apply budget + drop + delay to one message.

        Returns:
            delivered: bool
            recv_slot: int
            out_msg: Message
            meta: dict
        """
        # 1) budget check
        msg_mb = msg.bit_size / 8.0 / 1e6
        if msg_mb > self.budget_mb:
            msg.dropped = True
            msg.drop_reason = "budget_exceeded"
            return False, msg.slot_id, msg, {"delivered": False}

        # 2) drop
        if self.drop_prob > 0 and self.rng.random() < self.drop_prob:
            msg.dropped = True
            msg.drop_reason = "random_drop"
            return False, msg.slot_id, msg, {"delivered": False}

        # 3) delay
        msg.recv_slot = msg.slot_id + self.delay_slots
        msg.delay_slots = self.delay_slots
        msg.delivered = True

        return True, msg.recv_slot, msg, {
            "delivered": True,
            "delay_slots": self.delay_slots,
            "budget_mb": self.budget_mb,
            "drop_prob": self.drop_prob,
        }