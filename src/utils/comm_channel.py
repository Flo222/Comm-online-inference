from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import torch


class BasicCommChannel:
    """
    First-step communication model for multiview online inference.

    It applies three effects to per-view features:
    1) communication budget B
    2) fixed delay d
    3) packet/message drop p_drop

    Input:
        feat:      [B, N, C, H, W]
        keep_cams: [B, N]  (data-side available cameras)
    Output:
        recv_feat: [B, N, C, H, W]
        recv_mask: [B, N]  (message-side available cameras after comm)
        stats:     dict
    """

    def __init__(
        self,
        budget_mb: float,
        delay_ms: int,
        drop_prob: float,
        slot_ms: int = 100,
        bits_per_value: int = 16,
        budget_policy: str = "prefix",
    ):
        self.budget_mb = float(budget_mb)
        self.delay_ms = int(delay_ms)
        self.drop_prob = float(drop_prob)
        self.slot_ms = int(slot_ms)
        self.bits_per_value = int(bits_per_value)
        self.budget_policy = budget_policy

        self.delay_slots = max(self.delay_ms // self.slot_ms, 0)

        # per-camera delay buffer
        self.buffers: Dict[int, deque] = {}

    def reset(self):
        self.buffers = {}

    def _ensure_buffers(self, num_cam: int):
        for cam in range(num_cam):
            if cam not in self.buffers:
                self.buffers[cam] = deque(maxlen=self.delay_slots + 1)

    def _message_size_mb(self, feat: torch.Tensor) -> float:
        """
        feat: [B, C, H, W] for one camera, but message size is per-sample.
        So estimate using one sample's [C, H, W].
        """
        per_sample_numel = feat[0].numel()
        return per_sample_numel * self.bits_per_value / 8.0 / 1e6

    def _apply_delay(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feat: [B, N, C, H, W]
        Return:
            delayed_feat: [B, N, C, H, W]
            delay_mask:   [B, N]  whether delayed message exists
        """
        B, N = feat.shape[:2]
        device = feat.device
        dtype = feat.dtype

        self._ensure_buffers(N)

        delayed_feat = torch.zeros_like(feat)
        delay_mask = torch.zeros((B, N), dtype=torch.bool, device=device)

        # push current features into each camera buffer
        for cam in range(N):
            # store detached historical snapshot
            self.buffers[cam].append(feat[:, cam].detach().clone())

            if self.delay_slots == 0:
                delayed_feat[:, cam] = feat[:, cam]
                delay_mask[:, cam] = True
            else:
                # only available when enough history exists
                if len(self.buffers[cam]) >= self.delay_slots + 1:
                    delayed_feat[:, cam] = self.buffers[cam][0].to(device=device, dtype=dtype)
                    delay_mask[:, cam] = True

        return delayed_feat, delay_mask

    def _apply_drop(
        self, recv_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        recv_mask: [B, N] bool
        """
        if self.drop_prob <= 0:
            return recv_mask

        keep = torch.rand_like(recv_mask.float()) > self.drop_prob
        return recv_mask & keep.bool()

    def _apply_budget(
        self,
        recv_mask: torch.Tensor,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        recv_mask: [B, N] bool
        feat: [B, N, C, H, W]
        """
        B, N = recv_mask.shape
        budget_keep = torch.zeros_like(recv_mask)

        # all cameras have same per-sample message size if feature shape is same
        msg_sizes = [self._message_size_mb(feat[:, cam]) for cam in range(N)]

        for b in range(B):
            if self.budget_policy == "random":
                order = torch.randperm(N).tolist()
            else:
                order = list(range(N))

            used = 0.0
            for cam in order:
                if not recv_mask[b, cam]:
                    continue
                msg_mb = msg_sizes[cam]
                if used + msg_mb <= self.budget_mb + 1e-12:
                    budget_keep[b, cam] = True
                    used += msg_mb

        return budget_keep

    def apply(
        self,
        feat: torch.Tensor,
        keep_cams: Optional[torch.Tensor] = None,
        frame_ids=None,
    ):
        """
        feat: [B, N, C, H, W]
        keep_cams: [B, N] or None
        """
        B, N = feat.shape[:2]
        device = feat.device

        if keep_cams is None:
            base_mask = torch.ones((B, N), dtype=torch.bool, device=device)
        else:
            base_mask = keep_cams.bool().to(device)

        # 1) delay
        delayed_feat, delay_mask = self._apply_delay(feat)

        recv_mask = base_mask & delay_mask

        # 2) packet drop
        recv_mask = self._apply_drop(recv_mask)

        # 3) budget
        budget_mask = self._apply_budget(recv_mask, delayed_feat)
        recv_mask = recv_mask & budget_mask

        # zero out unavailable messages
        recv_feat = delayed_feat * recv_mask[:, :, None, None, None].to(delayed_feat.dtype)

        msg_size_mb = self._message_size_mb(feat[:, 0]) if N > 0 else 0.0
        effective_msgs = recv_mask.sum().item()
        total_msgs = base_mask.sum().item()

        stats = {
            "budget_mb": self.budget_mb,
            "delay_ms": self.delay_ms,
            "delay_slots": self.delay_slots,
            "drop_prob": self.drop_prob,
            "msg_size_mb_per_cam": msg_size_mb,
            "effective_msgs": int(effective_msgs),
            "total_msgs": int(total_msgs),
            "effective_ratio": float(effective_msgs / max(total_msgs, 1)),
            "realized_comm_mb": float(effective_msgs * msg_size_mb),
        }

        return recv_feat, recv_mask, stats