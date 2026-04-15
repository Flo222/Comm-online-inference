# src/system/node.py
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F

from src.system.message import Message


class Node:
    def __init__(
        self,
        node_id: str,
        cam_id: int,
        compress_mode: str = "none",
        quant_bits: int = 16,
        topk_ratio: float = 1.0,
    ):
        self.node_id = node_id
        self.cam_id = cam_id
        self.compress_mode = compress_mode
        self.quant_bits = quant_bits
        self.topk_ratio = topk_ratio

        self.current_slot: Optional[Dict] = None
        self.local_obs: Optional[Dict] = None
        self.local_output: Optional[Dict] = None
        self.local_world_feat: Optional[torch.Tensor] = None
        self.inbox: List[Message] = []

    def reset_slot(self):
        self.current_slot = None
        self.local_obs = None
        self.local_output = None
        self.local_world_feat = None
        self.inbox = []

    def observe(self, slot_sample: Dict):
        self.current_slot = slot_sample
        self.local_obs = {
            "cam_id": self.cam_id,
            "frame_id": slot_sample.get("frame_id", None),
        }

    def set_local_world_feat(self, world_feat: torch.Tensor):
        self.local_world_feat = world_feat
        self.local_output = {
            "node_id": self.node_id,
            "cam_id": self.cam_id,
            "frame_id": self.current_slot.get("frame_id", None) if self.current_slot else None,
            "status": "local_world_feat_ready",
        }

    def _compress_world_feat(self, world_feat: torch.Tensor) -> torch.Tensor:
        if self.compress_mode == "none":
            return world_feat

        if self.compress_mode == "fp16":
            return world_feat.half()

        if self.compress_mode == "avgpool":
            if world_feat.dim() == 4:
                return F.avg_pool2d(world_feat, kernel_size=2, stride=2)
            return world_feat

        if self.compress_mode == "topk":
            flat = world_feat.flatten()
            k = max(1, int(flat.numel() * self.topk_ratio))
            _, idx = torch.topk(flat.abs(), k)
            sparse = torch.zeros_like(flat)
            sparse[idx] = flat[idx]
            return sparse.view_as(world_feat)

        return world_feat

    def build_message(self, receiver: str = "server", msg_type: str = "world_feat") -> Message:
        compressed_feat = self._compress_world_feat(self.local_world_feat)

        payload = {
            "node_id": self.node_id,
            "cam_id": self.cam_id,
            "world_feat": compressed_feat,
        }

        elem_count = int(compressed_feat.numel()) if torch.is_tensor(compressed_feat) else 1
        bit_size = elem_count * int(self.quant_bits)

        slot_id = self.current_slot["slot_id"] if self.current_slot else -1

        return Message(
            sender=self.node_id,
            receiver=receiver,
            msg_type=msg_type,
            payload=payload,
            size=elem_count,          # old field for compatibility
            bit_size=bit_size,
            slot_id=slot_id,
            send_slot=slot_id,
            meta={
                "compress_mode": self.compress_mode,
                "quant_bits": self.quant_bits,
            },
        )

    def receive_message(self, msg: Message):
        self.inbox.append(msg)