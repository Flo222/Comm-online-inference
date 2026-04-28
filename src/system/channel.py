# src/system/channel.py
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import torch


class GilbertElliottLink:
    """
    Gilbert-Elliott two-state burst packet loss model.

    Convention:
      p = P(Good -> Bad)
      r = P(Bad -> Good)
      k = success probability in Good state
      h = success probability in Bad state

    Therefore:
      Good-state loss prob = 1 - k
      Bad-state loss prob  = 1 - h
    """

    def __init__(
        self,
        p: float,
        r: float,
        h: float,
        k: float,
        rng: random.Random,
        init_state: str = "G",
    ):
        self.p = float(p)
        self.r = float(r)
        self.h = float(h)
        self.k = float(k)
        self.rng = rng

        if init_state not in ("G", "B"):
            raise ValueError(f"init_state must be 'G' or 'B', got {init_state}")

        self.state = init_state

    def sample_loss(self) -> Tuple[bool, str, float]:
        """
        Returns:
            lost: whether current packet is lost
            state: current GE state after transition
            loss_prob: packet loss probability under current state
        """

        # First update Markov state.
        if self.state == "G":
            if self.rng.random() < self.p:
                self.state = "B"
        else:
            if self.rng.random() < self.r:
                self.state = "G"

        # Then sample state-dependent packet loss.
        if self.state == "G":
            loss_prob = 1.0 - self.k
        else:
            loss_prob = 1.0 - self.h

        lost = self.rng.random() < loss_prob
        return lost, self.state, loss_prob


class ChannelModel:
    """
    Channel model for online collaborative inference.

    Supported:
      1. budget check
      2. message-level Bernoulli loss
      3. channel-packet-level Bernoulli loss
      4. message-level Gilbert-Elliott burst loss
      5. channel-packet-level Gilbert-Elliott burst loss
      6. slot delay

    For channel-packet-level loss:
      - payload['world_feat'] is packetized along the channel dimension
      - packetization can be fixed-size or size-aware
      - lost packets are zeroed
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

        # new channel model args
        channel_model: str = "bernoulli",
        loss_granularity: str = "message",
        packetize_mode: str = "fixed",
        channel_group_size: int = 16,
        packet_max_bytes: int = 6400,

        # Gilbert-Elliott args
        ge_p: float = 0.378563411896744,
        ge_r: float = 0.883314627759071,
        ge_h: float = 0.810,
        ge_k: float = 0.938571428571429,
        ge_scope: str = "link",
        ge_init_state: str = "G",
    ):
        self.budget_mb = float(budget_mb)
        self.delay_ms = int(delay_ms)
        self.drop_prob = float(drop_prob)
        self.slot_ms = int(slot_ms)
        self.bits_per_value = int(bits_per_value)
        self.budget_policy = budget_policy

        self.channel_model = str(channel_model)
        self.loss_granularity = str(loss_granularity)
        self.packetize_mode = str(packetize_mode)
        self.channel_group_size = int(channel_group_size)
        self.packet_max_bytes = int(packet_max_bytes)

        self.ge_p = float(ge_p)
        self.ge_r = float(ge_r)
        self.ge_h = float(ge_h)
        self.ge_k = float(ge_k)
        self.ge_scope = str(ge_scope)
        self.ge_init_state = str(ge_init_state)

        self.rng = random.Random(rng_seed)
        self.delay_slots = max(self.delay_ms // self.slot_ms, 0)

        self._ge_links: Dict[str, GilbertElliottLink] = {}

        if self.channel_model not in ("bernoulli", "gilbert_elliott"):
            raise ValueError(
                f"Unknown channel_model={self.channel_model}. "
                "Use 'bernoulli' or 'gilbert_elliott'."
            )

        if self.loss_granularity not in ("message", "channel"):
            raise ValueError(
                f"Unknown loss_granularity={self.loss_granularity}. "
                "Use 'message' or 'channel'."
            )

        if self.packetize_mode not in ("fixed", "size"):
            raise ValueError(
                f"Unknown packetize_mode={self.packetize_mode}. "
                "Use 'fixed' or 'size'."
            )

        if self.ge_scope not in ("global", "link"):
            raise ValueError("ge_scope must be 'global' or 'link'.")

        if self.ge_scope == "global":
            self._ge_links["global"] = self._new_ge_link()

    def _new_ge_link(self) -> GilbertElliottLink:
        return GilbertElliottLink(
            p=self.ge_p,
            r=self.ge_r,
            h=self.ge_h,
            k=self.ge_k,
            rng=self.rng,
            init_state=self.ge_init_state,
        )

    def _get_link_key(self, msg) -> str:
        if self.ge_scope == "global":
            return "global"
        if getattr(msg, "link_id", None):
            return str(msg.link_id)
        return f"{msg.sender}->{msg.receiver}"

    def _get_ge_link(self, msg) -> Tuple[GilbertElliottLink, str]:
        key = self._get_link_key(msg)
        if key not in self._ge_links:
            self._ge_links[key] = self._new_ge_link()
        return self._ge_links[key], key

    def _extract_world_feat(self, msg) -> torch.Tensor | None:
        """
        Current Node.build_message() uses:
            msg.payload = {'node_id': ..., 'cam_id': ..., 'world_feat': tensor}
        """
        payload = getattr(msg, "payload", None)
        if not isinstance(payload, dict):
            return None

        feat = payload.get("world_feat", None)
        if torch.is_tensor(feat):
            return feat

        return None

    def _infer_feature_shape(self, feat: torch.Tensor) -> Tuple[int, int, int, int]:
        """
        Returns:
            channel_dim, C, H, W

        Supported shapes:
            [C, H, W]
            [B, C, H, W]
        """
        if feat.dim() == 3:
            C, H, W = int(feat.shape[0]), int(feat.shape[1]), int(feat.shape[2])
            return 0, C, H, W

        if feat.dim() == 4:
            C, H, W = int(feat.shape[1]), int(feat.shape[2]), int(feat.shape[3])
            return 1, C, H, W

        raise ValueError(
            f"Only [C,H,W] or [B,C,H,W] feature maps are supported, got shape={tuple(feat.shape)}"
        )

    def _build_channel_packets(
        self,
        C: int,
        H: int,
        W: int,
        bits_per_value: int,
    ) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
        """
        Build channel packet ranges.

        fixed mode:
            every channel_group_size channels are one packet.

        size mode:
            estimate one channel size by H * W * bits_per_value / 8.
            Then pack as many channels as possible into packet_max_bytes.
        """
        packets: List[Tuple[int, int]] = []

        bytes_per_channel = float(H * W * bits_per_value / 8.0)

        if self.packetize_mode == "fixed":
            channels_per_packet = max(1, self.channel_group_size)

        else:
            # size-aware packetization
            if self.packet_max_bytes <= 0:
                raise ValueError("packet_max_bytes must be positive when packetize_mode='size'.")

            if bytes_per_channel <= 0:
                channels_per_packet = 1
            else:
                channels_per_packet = max(1, int(self.packet_max_bytes // bytes_per_channel))

        for c0 in range(0, C, channels_per_packet):
            c1 = min(c0 + channels_per_packet, C)
            packets.append((c0, c1))

        # meta = {
        #     "packetize_mode": self.packetize_mode,
        #     "packet_max_bytes": self.packet_max_bytes,
        #     "bytes_per_channel_est": bytes_per_channel,
        #     "channels_per_packet": channels_per_packet,
        #     "total_packets": len(packets),
        #     "total_channels": C,
        # }

        meta = {
            "packetize_mode": self.packetize_mode,
            "packet_max_bytes": self.packet_max_bytes,
            "bytes_per_channel_est": bytes_per_channel,
            "channels_per_packet": channels_per_packet,
            "total_packets": len(packets),
            "total_channels": C,
        }

        # 只打印一次，避免每个 message 都刷屏
        if not hasattr(self, "_printed_packetize_debug"):
            print(
                "[DEBUG packetize]",
                "mode=", self.packetize_mode,
                "C=", C,
                "H=", H,
                "W=", W,
                "bits_per_value=", bits_per_value,
                "bytes_per_channel=", bytes_per_channel,
                "packet_max_bytes=", self.packet_max_bytes,
                "channels_per_packet=", channels_per_packet,
                "total_packets=", len(packets),
            )
            self._printed_packetize_debug = True

        return packets, meta

    def _sample_message_loss(self, msg) -> Tuple[bool, Dict[str, Any]]:
        """
        Sample whether the whole message is lost.
        """
        if self.channel_model == "bernoulli":
            lost = self.drop_prob > 0 and self.rng.random() < self.drop_prob
            return lost, {
                "channel_model": "bernoulli",
                "loss_granularity": "message",
                "drop_prob": self.drop_prob,
                "total_packets": 1,
                "lost_packets": 1 if lost else 0,
                "packet_loss_rate": 1.0 if lost else 0.0,
            }

        ge_link, link_key = self._get_ge_link(msg)
        lost, state, loss_prob = ge_link.sample_loss()
        return lost, {
            "channel_model": "gilbert_elliott",
            "loss_granularity": "message",
            "ge_link": link_key,
            "ge_state": state,
            "ge_loss_prob": loss_prob,
            "ge_p": self.ge_p,
            "ge_r": self.ge_r,
            "ge_h": self.ge_h,
            "ge_k": self.ge_k,
            "total_packets": 1,
            "lost_packets": 1 if lost else 0,
            "packet_loss_rate": 1.0 if lost else 0.0,
        }

    def _sample_packet_loss(self, msg) -> Tuple[bool, Dict[str, Any]]:
        """
        Sample loss for one channel packet.
        """
        if self.channel_model == "bernoulli":
            lost = self.drop_prob > 0 and self.rng.random() < self.drop_prob
            return lost, {
                "packet_state": "I",
                "packet_loss_prob": self.drop_prob,
            }

        ge_link, _ = self._get_ge_link(msg)
        lost, state, loss_prob = ge_link.sample_loss()
        return lost, {
            "packet_state": state,
            "packet_loss_prob": loss_prob,
        }

    def _apply_channel_packet_loss(self, msg) -> Tuple[bool, Dict[str, Any]]:
        """
        Packetize payload['world_feat'] along channel dimension.
        Lost channel packets are zeroed.

        If all packets are lost, we treat the whole message as dropped.
        This avoids adding an all-zero corrupted feature into max fusion.
        """
        feat = self._extract_world_feat(msg)

        if feat is None:
            # If this is a logits late-fusion message, packetize logits instead
            # of falling back to message-level loss.
            heatmap, offset = self._extract_logits_payload(msg)
            if heatmap is not None and offset is not None:
                return self._apply_logits_packet_loss(msg)

            # Fallback for unexpected payloads.
            lost, meta = self._sample_message_loss(msg)
            return (not lost), meta
        try:
            channel_dim, C, H, W = self._infer_feature_shape(feat)
        except ValueError:
            lost, meta = self._sample_message_loss(msg)
            return (not lost), meta

        bits_per_value = int(msg.meta.get("quant_bits", self.bits_per_value))
        packets, pkt_meta = self._build_channel_packets(
            C=C,
            H=H,
            W=W,
            bits_per_value=bits_per_value,
        )

        # Important:
        # If compress_mode='none', Node.build_message() may pass the same tensor object
        # as node.local_world_feat. Clone it before zeroing, otherwise the sender local
        # feature can be corrupted.
        out_feat = feat.clone()

        packet_mask: List[int] = []
        packet_states: List[str] = []
        packet_loss_probs: List[float] = []
        lost_packets = 0

        channel_valid_mask = torch.ones(C, device=feat.device, dtype=torch.bool)

        for c0, c1 in packets:
            lost, packet_meta = self._sample_packet_loss(msg)
            packet_mask.append(0 if lost else 1)
            packet_states.append(str(packet_meta.get("packet_state", "")))
            packet_loss_probs.append(float(packet_meta.get("packet_loss_prob", 0.0)))

            if not lost:
                continue

            if lost:
                lost_packets += 1
                channel_valid_mask[c0:c1] = False

                if channel_dim == 0:
                    out_feat[c0:c1, :, :] = 0
                else:
                    out_feat[:, c0:c1, :, :] = 0

        msg.payload["world_feat"] = out_feat
        msg.payload["channel_valid_mask"] = channel_valid_mask

        total_packets = len(packets)
        packet_loss_rate = lost_packets / max(total_packets, 1)

        ge_link_key = self._get_link_key(msg) if self.channel_model == "gilbert_elliott" else ""

        meta = {
            "channel_model": self.channel_model,
            "loss_granularity": "channel",
            "lost_packets": lost_packets,
            "total_packets": total_packets,
            "packet_loss_rate": packet_loss_rate,
            "packet_mask": packet_mask,
            "packet_states": packet_states,
            "packet_loss_probs": packet_loss_probs,
            "ge_link": ge_link_key,
            **pkt_meta,
        }

        msg.meta.update(meta)

        if lost_packets == total_packets:
            msg.dropped = True
            msg.drop_reason = "all_channel_packets_lost"
            return False, meta

        return True, meta

    def apply(self, msg):
        """
        Apply budget + drop + delay to one message.

        Returns:
            delivered: bool
            recv_slot: int
            out_msg: Message
            meta: dict
        """

        # 1) budget check: communication attempt cost is still the full message.
        msg_mb = msg.bit_size / 8.0 / 1e6

        if msg_mb > self.budget_mb:
            msg.dropped = True
            msg.drop_reason = "budget_exceeded"
            meta = {
                "delivered": False,
                "drop_reason": "budget_exceeded",
                "msg_mb": msg_mb,
                "budget_mb": self.budget_mb,
                "channel_model": self.channel_model,
                "loss_granularity": self.loss_granularity,
                "lost_packets": 0,
                "total_packets": 0,
                "packet_loss_rate": 0.0,
            }
            msg.meta.update(meta)
            return False, msg.slot_id, msg, meta

        # 2) loss
        if self.loss_granularity == "message":
            lost, loss_meta = self._sample_message_loss(msg)
            msg.meta.update(loss_meta)

            if lost:
                msg.dropped = True
                msg.drop_reason = f"{self.channel_model}_message_lost"
                meta = {
                    "delivered": False,
                    "drop_reason": msg.drop_reason,
                    "msg_mb": msg_mb,
                    "budget_mb": self.budget_mb,
                    **loss_meta,
                }
                msg.meta.update(meta)
                return False, msg.slot_id, msg, meta

        else:
            delivered, loss_meta = self._apply_channel_packet_loss(msg)

            if not delivered:
                meta = {
                    "delivered": False,
                    "drop_reason": msg.drop_reason or "channel_packet_loss",
                    "msg_mb": msg_mb,
                    "budget_mb": self.budget_mb,
                    **loss_meta,
                }
                msg.meta.update(meta)
                return False, msg.slot_id, msg, meta

        # 3) delay
        msg.recv_slot = msg.slot_id + self.delay_slots
        msg.delay_slots = self.delay_slots
        msg.delivered = True

        meta = {
            "delivered": True,
            "delay_slots": self.delay_slots,
            "msg_mb": msg_mb,
            "budget_mb": self.budget_mb,
            "drop_prob": self.drop_prob,
            "channel_model": self.channel_model,
            "loss_granularity": self.loss_granularity,
        }
        meta.update(msg.meta)

        return True, msg.recv_slot, msg, meta

    def _extract_logits_payload(self, msg):
        """
        For logits late fusion messages:
            payload['world_heatmap']
            payload['world_offset']
        """
        payload = getattr(msg, "payload", None)
        if not isinstance(payload, dict):
            return None, None

        hm = payload.get("world_heatmap", None)
        off = payload.get("world_offset", None)

        if torch.is_tensor(hm) and torch.is_tensor(off):
            return hm, off

        return None, None

    def _infer_logits_shape(self, heatmap: torch.Tensor, offset: torch.Tensor):
        """
        Return H, W, total_channels for logits payload.

        Expected common shapes:
            heatmap: [B, 1, H, W]
            offset : [B, 2, H, W]

        Also supports:
            heatmap: [1, H, W]
            offset : [2, H, W]
        """

        if heatmap.dim() == 4:
            hm_c = int(heatmap.shape[1])
            H = int(heatmap.shape[2])
            W = int(heatmap.shape[3])
        elif heatmap.dim() == 3:
            hm_c = int(heatmap.shape[0])
            H = int(heatmap.shape[1])
            W = int(heatmap.shape[2])
        elif heatmap.dim() == 2:
            hm_c = 1
            H = int(heatmap.shape[0])
            W = int(heatmap.shape[1])
        else:
            raise ValueError(f"Unsupported heatmap shape: {tuple(heatmap.shape)}")

        if offset.dim() == 4:
            off_c = int(offset.shape[1])
            off_H = int(offset.shape[2])
            off_W = int(offset.shape[3])
        elif offset.dim() == 3:
            off_c = int(offset.shape[0])
            off_H = int(offset.shape[1])
            off_W = int(offset.shape[2])
        else:
            raise ValueError(f"Unsupported offset shape: {tuple(offset.shape)}")

        if H != off_H or W != off_W:
            raise ValueError(
                f"heatmap and offset spatial size mismatch: "
                f"heatmap=({H},{W}), offset=({off_H},{off_W})"
            )

        total_channels = hm_c + off_c
        return H, W, total_channels

    def _build_logits_spatial_packets(
        self,
        H: int,
        W: int,
        total_channels: int,
        bits_per_value: int,
    ):
        """
        Size-aware packetization for logits.

        Each packet contains a contiguous row block of:
            heatmap + offset

        Packet size estimate:
            rows * W * total_channels * bits_per_value / 8
        """

        bytes_per_row = float(W * total_channels * bits_per_value / 8.0)

        if self.packetize_mode == "fixed":
            # Reuse channel_group_size as rows_per_packet for logits fixed mode.
            rows_per_packet = max(1, int(self.channel_group_size))
        else:
            if self.packet_max_bytes <= 0:
                raise ValueError("packet_max_bytes must be positive when packetize_mode='size'.")

            rows_per_packet = max(1, int(self.packet_max_bytes // bytes_per_row))

        packets = []
        for r0 in range(0, H, rows_per_packet):
            r1 = min(r0 + rows_per_packet, H)
            packets.append((r0, r1))

        meta = {
            "packetize_mode": self.packetize_mode,
            "logits_packetize_axis": "spatial_rows",
            "packet_max_bytes": self.packet_max_bytes,
            "bytes_per_logits_row_est": bytes_per_row,
            "rows_per_packet": rows_per_packet,
            "total_packets": len(packets),
            "logits_H": H,
            "logits_W": W,
            "logits_channels": total_channels,
        }

        if not hasattr(self, "_printed_logits_packetize_debug"):
            print(
                "[DEBUG logits packetize]",
                "mode=", self.packetize_mode,
                "H=", H,
                "W=", W,
                "total_channels=", total_channels,
                "bits_per_value=", bits_per_value,
                "bytes_per_row=", bytes_per_row,
                "packet_max_bytes=", self.packet_max_bytes,
                "rows_per_packet=", rows_per_packet,
                "total_packets=", len(packets),
            )
            self._printed_logits_packetize_debug = True

        return packets, meta

    def _zero_logits_rows(
        self,
        heatmap: torch.Tensor,
        offset: torch.Tensor,
        r0: int,
        r1: int,
    ):
        """
        For max late fusion:
        - lost heatmap rows are set to a very small value
            so they will not win max fusion.
        - lost offset rows are set to 0.
        """

        # heatmap
        if heatmap.dim() == 4:
            heatmap[:, :, r0:r1, :] = -1e4
        elif heatmap.dim() == 3:
            heatmap[:, r0:r1, :] = -1e4
        elif heatmap.dim() == 2:
            heatmap[r0:r1, :] = -1e4

        # offset
        if offset.dim() == 4:
            offset[:, :, r0:r1, :] = 0
        elif offset.dim() == 3:
            offset[:, r0:r1, :] = 0


    def _apply_logits_packet_loss(self, msg) -> Tuple[bool, Dict[str, Any]]:
        """
        Packetize logits payload spatially.

        payload:
            world_heatmap
            world_offset

        A lost packet means the corresponding spatial rows of both
        heatmap and offset are unavailable.
        """

        heatmap, offset = self._extract_logits_payload(msg)

        if heatmap is None or offset is None:
            lost, meta = self._sample_message_loss(msg)
            return (not lost), meta

        try:
            H, W, total_channels = self._infer_logits_shape(heatmap, offset)
        except ValueError:
            lost, meta = self._sample_message_loss(msg)
            return (not lost), meta

        bits_per_value = int(msg.meta.get("quant_bits", self.bits_per_value))

        packets, pkt_meta = self._build_logits_spatial_packets(
            H=H,
            W=W,
            total_channels=total_channels,
            bits_per_value=bits_per_value,
        )

        out_heatmap = heatmap.clone()
        out_offset = offset.clone()

        packet_mask: List[int] = []
        packet_states: List[str] = []
        packet_loss_probs: List[float] = []
        lost_packets = 0

        logits_valid_mask = torch.ones(
            H,
            W,
            device=heatmap.device,
            dtype=torch.bool,
        )

        for r0, r1 in packets:
            lost, packet_meta = self._sample_packet_loss(msg)

            packet_mask.append(0 if lost else 1)
            packet_states.append(str(packet_meta.get("packet_state", "")))
            packet_loss_probs.append(float(packet_meta.get("packet_loss_prob", 0.0)))

            if not lost:
                continue

            lost_packets += 1
            logits_valid_mask[r0:r1, :] = False

            self._zero_logits_rows(
                heatmap=out_heatmap,
                offset=out_offset,
                r0=r0,
                r1=r1,
            )

        msg.payload["world_heatmap"] = out_heatmap
        msg.payload["world_offset"] = out_offset
        msg.payload["logits_valid_mask"] = logits_valid_mask

        total_packets = len(packets)
        packet_loss_rate = lost_packets / max(total_packets, 1)

        ge_link_key = self._get_link_key(msg) if self.channel_model == "gilbert_elliott" else ""

        meta = {
            "channel_model": self.channel_model,
            "loss_granularity": "channel",
            "payload_type": "logits",
            "lost_packets": lost_packets,
            "total_packets": total_packets,
            "packet_loss_rate": packet_loss_rate,
            "packet_mask": packet_mask,
            "packet_states": packet_states,
            "packet_loss_probs": packet_loss_probs,
            "ge_link": ge_link_key,
            **pkt_meta,
        }

        msg.meta.update(meta)

        if lost_packets == total_packets:
            msg.dropped = True
            msg.drop_reason = "all_logits_packets_lost"
            return False, meta

        return True, meta