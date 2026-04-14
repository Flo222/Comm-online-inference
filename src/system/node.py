from typing import Dict, List, Optional

import torch

from src.system.message import Message


class Node:
    """
    Collaborative inference v1:
    - one node corresponds to one camera/view
    - each node performs local inference on its own view only
    - the local world feature is packed into the message payload
    """

    def __init__(self, node_id: str, cam_id: int):
        self.node_id = node_id
        self.cam_id = cam_id
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

    def build_message(self, msg_type: str = "world_feat") -> Message:
        payload = {
            "node_id": self.node_id,
            "cam_id": self.cam_id,
            "world_feat": self.local_world_feat,
        }
        size = self._estimate_message_size(payload)

        return Message(
            sender=self.node_id,
            receiver="server",
            msg_type=msg_type,
            payload=payload,
            size=size,
            slot_id=self.current_slot["slot_id"] if self.current_slot else -1,
        )

    def receive_message(self, msg: Message):
        self.inbox.append(msg)

    def _estimate_message_size(self, payload) -> int:
        world_feat = payload.get("world_feat", None)
        if torch.is_tensor(world_feat):
            return int(world_feat.numel())
        return 1