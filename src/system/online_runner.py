import os
import numpy as np
import torch

from src.stream.online_stream import OnlineStream
from src.system.node import Node
from src.system.comm_manager import CommManager
from src.system.decision import DecisionPolicy
from src.system.logger import OnlineLogger
from src.system.channel import ChannelModel

from src.loss import focal_loss, regL1loss
from src.evaluation.evaluate import evaluate
from src.utils.decode import mvdet_decode
from src.utils.nms import nms
from src.models.aggregation import aggregate_feat


class OnlineRunner:
    """
    Collaborative Inference with channel-aware communication.

    This version is aligned with the current args in main_base.py:
    - comm_budget_mb
    - comm_delay_ms
    - comm_drop_prob
    - slot_ms
    - comm_bits_per_value
    - comm_budget_policy

    Current pipeline:
    1) run original multi-view get_feat ONCE
    2) treat each per-view world feature feat_all[:, i] as one node's local message
    3) send messages through CommManager + ChannelModel
    4) server fuses only delivered messages
    5) detection head runs on fused world feature
    """

    def __init__(self, model, train_dataset, test_dataset, logdir, args, optimizer=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.logdir = logdir
        self.args = args
        self.optimizer = optimizer

        self.train_stream = OnlineStream(train_dataset, sort_by_time=True)
        self.test_stream = OnlineStream(test_dataset, sort_by_time=True)

        self.decision_policy = DecisionPolicy(
            mode=getattr(args, "comm_topology", "all_to_server"),
            max_active=getattr(args, "max_active_senders", None),
            self_loop=getattr(args, "comm_self_loop", False),
            bidirectional=getattr(args, "comm_bidirectional", True),
            custom_edges=getattr(args, "comm_edges", ""),
        )

        ref_dataset = train_dataset
        num_cam = ref_dataset.num_cam if hasattr(ref_dataset, "num_cam") else ref_dataset.base.num_cam

        # Be robust to either old or new Node signature
        self.nodes = []
        for i in range(num_cam):
            try:
                node = Node(
                    node_id=f"node_{i}",
                    cam_id=i,
                    compress_mode=getattr(args, "compress_mode", "none"),
                    quant_bits=getattr(args, "quant_bits", 16),
                    topk_ratio=getattr(args, "topk_ratio", 1.0),
                )
            except TypeError:
                node = Node(node_id=f"node_{i}", cam_id=i)
            self.nodes.append(node)

        self.node_ids = [n.node_id for n in self.nodes]
        self.current_edges = []

        channel = ChannelModel(
            budget_mb=getattr(args, "comm_budget_mb", 128.0),
            delay_ms=getattr(args, "comm_delay_ms", 0),
            drop_prob=getattr(args, "comm_drop_prob", 0.0),
            slot_ms=getattr(args, "slot_ms", 100),
            bits_per_value=getattr(args, "comm_bits_per_value", 16),
            budget_policy=getattr(args, "comm_budget_policy", "prefix"),
        )

        self.comm_manager = CommManager(channel=channel)

        # Be robust to either old or new DecisionPolicy signature
        # try:
        #     self.decision_policy = DecisionPolicy(
        #         mode=getattr(args, "comm_graph", "all_to_server"),
        #         max_active=getattr(args, "max_active_senders", None),
        #     )
        # except TypeError:
        #     self.decision_policy = DecisionPolicy(mode="all_active")

        self.logger = OnlineLogger(logdir=logdir)

    def _ensure_gt_batch_dim(self, gt_dict):
        out = {}
        for k, v in gt_dict.items():
            if torch.is_tensor(v):
                out[k] = v.unsqueeze(0)
            else:
                out[k] = v
        return out

    def _normalize_slot_tensors(self, slot_sample):
        imgs = slot_sample["images"]
        affine_mats = slot_sample["affine_mats"]
        keep_cams = slot_sample["keep_cams"]
        world_gt = slot_sample["world_gt"]
        imgs_gt = slot_sample["imgs_gt"]
        frame = slot_sample["frame_id"]

        if torch.is_tensor(imgs) and imgs.dim() == 4:
            imgs = imgs.unsqueeze(0)
        if torch.is_tensor(affine_mats) and affine_mats.dim() == 3:
            affine_mats = affine_mats.unsqueeze(0)
        if torch.is_tensor(keep_cams) and keep_cams.dim() == 1:
            keep_cams = keep_cams.unsqueeze(0)

        world_gt = self._ensure_gt_batch_dim(world_gt)

        out_imgs_gt = {}
        for k, v in imgs_gt.items():
            if torch.is_tensor(v):
                out_imgs_gt[k] = v.unsqueeze(0)
            else:
                out_imgs_gt[k] = v

        if torch.is_tensor(frame):
            if frame.dim() == 0:
                frame = frame.unsqueeze(0)
        else:
            frame = torch.tensor([frame])

        return imgs, affine_mats, keep_cams, world_gt, out_imgs_gt, frame

    def _prepare_imgs_gt_for_loss(self, imgs_gt):
        out = {}
        for k, v in imgs_gt.items():
            if torch.is_tensor(v) and v.dim() >= 2:
                out[k] = v.flatten(0, 1)
            else:
                out[k] = v
        return out

    def _fuse_world_features(self, world_feat_list):
        """
        world_feat_list: list of [B, C, H, W]
        """
        if len(world_feat_list) == 0:
            raise RuntimeError("No world features received for fusion.")

        fused = torch.stack(world_feat_list, dim=1)  # [B, N_active, C, H, W]
        if self.model.aggregation == "max":
            fused = fused.max(dim=1)[0]
        elif self.model.aggregation == "mean":
            fused = fused.mean(dim=1)
        else:
            raise ValueError(f"Unsupported aggregation: {self.model.aggregation}")
        return fused

    def _slot_comm_and_nodes(self, slot_sample):
        slot_id = slot_sample["slot_id"]

        # reset current-slot stats
        if hasattr(self.comm_manager, "reset_slot_stats"):
            self.comm_manager.reset_slot_stats()
        elif hasattr(self.comm_manager, "reset_slot"):
            self.comm_manager.reset_slot()

        # clear mailbox for current slot
        if hasattr(self.comm_manager, "clear_mailboxes"):
            self.comm_manager.clear_mailboxes()

        # deliver delayed messages scheduled for this slot
        if hasattr(self.comm_manager, "deliver_pending"):
            self.comm_manager.deliver_pending(slot_id)

        # reset node state and let every node observe current slot
        for node in self.nodes:
            node.reset_slot()
            node.observe(slot_sample)

        # NEW: decide communication edges instead of only active nodes
        if hasattr(self.decision_policy, "decide_edges"):
            edges = self.decision_policy.decide_edges(
                self.node_ids,
                server_id="server",
            )
            active_nodes = sorted(list({sender for sender, _ in edges}))
            decision = {
                "edges": edges,
                "active_nodes": active_nodes,
            }
        else:
            # backward-compatible fallback
            decision = self.decision_policy.decide(slot_sample, self.node_ids)
            active_nodes = decision["active_nodes"]
            edges = [(nid, "server") for nid in active_nodes]
            decision["edges"] = edges

        self.current_edges = edges
        return decision, set(active_nodes)

    def _forward_and_loss_collab(self, dataset, imgs, affine_mats, keep_cams, world_gt, imgs_gt, slot_id):
        B, N = imgs.shape[:2]
        imgs_gt = self._prepare_imgs_gt_for_loss(imgs_gt)

        # 1) 原始多视角 backbone 只跑一次
        feat_all, (imgs_heatmap, imgs_offset, imgs_wh) = self.model.get_feat(
            imgs, affine_mats, self.args.down
        )
        # feat_all: [B, N, C, H, W]

        # 2) 每个 node 保存自己的本地 world_feat
        for i, node in enumerate(self.nodes):
            node.set_local_world_feat(feat_all[:, i])

        node_map = {node.node_id: node for node in self.nodes}

        # 3) 按当前边集发送消息
        for sender_id, receiver_id in self.current_edges:
            msg = node_map[sender_id].build_message(
                receiver=receiver_id,
                msg_type="world_feat",
            )
            self.comm_manager.send(msg)

        # 4) 每个 agent 单独收消息 -> 单独融合 -> 单独输出
        agent_outputs = {}
        agent_losses = []

        for receiver_id in self.node_ids:
            recv_msgs = self.comm_manager.collect_for(receiver_id)

            # 先把自己的本地特征放进去
            feat_list = [node_map[receiver_id].local_world_feat]

            # 再加收到的其他人特征
            recv_from = []
            for msg in recv_msgs:
                wf = msg.payload.get("world_feat", None)
                sender = msg.payload.get("node_id", None)
                if wf is None:
                    continue

                # 若消息经过 fp16 压缩，转回本地 dtype
                wf = wf.to(feat_all.dtype)

                # 避免 self_loop=True 时把自己重复加两次
                if sender == receiver_id:
                    continue

                feat_list.append(wf)
                recv_from.append(sender)

            fused_feat = self._fuse_world_features(feat_list)
            world_heatmap_i, world_offset_i = self.model.get_output(fused_feat)

            # 记录
            agent_outputs[receiver_id] = {
                "fused_feat": fused_feat,
                "world_heatmap": world_heatmap_i,
                "world_offset": world_offset_i,
                "recv_from": recv_from,
            }

            # 每个 agent 各算一份 world loss
            loss_w_hm_i = focal_loss(world_heatmap_i, world_gt["heatmap"])
            loss_w_off_i = regL1loss(
                world_offset_i, world_gt["reg_mask"], world_gt["idx"], world_gt["offset"]
            )
            agent_losses.append(loss_w_hm_i + loss_w_off_i)

        # 5) image 辅助损失保留原来的共享写法
        loss_img_hm = focal_loss(
            imgs_heatmap,
            imgs_gt["heatmap"],
            keep_cams.view(B * N, 1, 1, 1),
        )
        loss_img_off = regL1loss(
            imgs_offset, imgs_gt["reg_mask"], imgs_gt["idx"], imgs_gt["offset"]
        )
        loss_img_wh = regL1loss(
            imgs_wh, imgs_gt["reg_mask"], imgs_gt["idx"], imgs_gt["wh"]
        )

        world_loss = torch.stack(agent_losses).mean()
        img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1
        loss = world_loss + img_loss / N * self.args.alpha

        if self.args.use_mse:
            mse_losses = []
            for receiver_id in self.node_ids:
                world_heatmap_i = agent_outputs[receiver_id]["world_heatmap"]
                mse_losses.append(
                    torch.nn.functional.mse_loss(
                        world_heatmap_i, world_gt["heatmap"].to(world_heatmap_i.device)
                    )
                )
            loss = torch.stack(mse_losses).mean() + \
                self.args.alpha * torch.nn.functional.mse_loss(
                    imgs_heatmap, imgs_gt["heatmap"].to(imgs_heatmap.device)
                ) / N

        return loss, agent_outputs

    def _decode_and_collect(self, dataset, world_heatmap, world_offset, frame, res_list):
        xys = mvdet_decode(
            torch.sigmoid(world_heatmap),
            world_offset,
            reduce=dataset.world_reduce,
        ).cpu()

        grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
        if dataset.base.indexing == "xy":
            positions = grid_xy
        else:
            positions = grid_xy[:, :, [1, 0]]

        B = world_heatmap.shape[0]
        for b in range(B):
            ids = scores[b].squeeze() > self.args.cls_thres
            pos, s = positions[b, ids], scores[b, ids, 0]
            ids, count = nms(pos, s, 20, np.inf)
            cur_frame = frame[b].item() if torch.is_tensor(frame) else frame
            if count > 0:
                res = torch.cat(
                    [torch.ones([count, 1]) * cur_frame, pos[ids[:count]]],
                    dim=1,
                )
                res_list.append(res)

    def run_train(self, max_slots=None):
        if self.optimizer is None:
            raise ValueError("online train mode requires optimizer")

        self.model.train()
        losses = 0.0
        num_steps = 0

        for i, slot_sample in enumerate(self.train_stream):
            if max_slots is not None and i >= max_slots:
                break

            slot_id = slot_sample["slot_id"]
            decision, active_nodes = self._slot_comm_and_nodes(slot_sample)

            imgs, affine_mats, keep_cams, world_gt, imgs_gt, frame = self._normalize_slot_tensors(slot_sample)
            imgs = imgs.cuda()
            affine_mats = affine_mats.cuda()

            # loss, world_heatmap, world_offset = self._forward_and_loss_collab(
            #     self.train_dataset, imgs, affine_mats, keep_cams, world_gt, imgs_gt, slot_id
            # )

            loss, agent_outputs = self._forward_and_loss_collab(
                self.train_dataset, imgs, affine_mats, keep_cams, world_gt, imgs_gt, slot_id
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses += loss.item()
            num_steps += 1

            comm_stats = self.comm_manager.get_slot_stats()

            metrics["num_agent_outputs"] = len(agent_outputs)
            metrics = {
                "slot_loss": float(loss.item()),
                "updated": 1,
            }
            self.logger.log_slot(slot_id, metrics, decision, comm_stats)

            print(
                f"[collab-train slot {slot_id}] "
                f"active_nodes={len(active_nodes)} "
                f"messages={comm_stats.get('num_messages', 0)} "
                f"delivered={comm_stats.get('delivered_messages', 0)} "
                f"dropped={comm_stats.get('dropped_messages', 0)} "
                f"comm_mb={comm_stats.get('comm_cost_mb', 0.0):.3f} "
                f"avg_delay_slots={comm_stats.get('avg_delay_slots', 0.0):.2f} "
                f"loss={loss.item():.6f}"
            )

        avg_loss = losses / max(1, num_steps)
        save_path = os.path.join(self.logdir, self.args.online_save_name)
        torch.save(self.model.state_dict(), save_path)
        self.logger.save_csv("online_train_log.csv")

        print("========== COLLAB TRAIN SUMMARY ==========")
        print(f"num_slots: {num_steps}")
        print(f"avg_slot_loss: {avg_loss:.6f}")
        print(f"saved_model: {save_path}")
        print("=========================================")

        return {
            "num_slots": num_steps,
            "avg_slot_loss": avg_loss,
            "saved_model": save_path,
        }

    def run_infer(self, max_slots=None):
        self.model.eval()

        res_dict = {node_id: [] for node_id in self.node_ids}
        losses = 0.0
        num_steps = 0

        with torch.no_grad():
            for i, slot_sample in enumerate(self.test_stream):
                if max_slots is not None and i >= max_slots:
                    break

                slot_id = slot_sample["slot_id"]
                decision, active_nodes = self._slot_comm_and_nodes(slot_sample)
                imgs, affine_mats, keep_cams, world_gt, imgs_gt, frame = self._normalize_slot_tensors(slot_sample)

                imgs = imgs.cuda()
                affine_mats = affine_mats.cuda()

                loss, agent_outputs = self._forward_and_loss_collab(
                    self.test_dataset, imgs, affine_mats, keep_cams, world_gt, imgs_gt, slot_id
                )

                losses += loss.item()
                num_steps += 1

                for node_id, out in agent_outputs.items():
                    self._decode_and_collect(
                        self.test_dataset,
                        out["world_heatmap"],
                        out["world_offset"],
                        frame,
                        res_dict[node_id],
                    )

                comm_stats = self.comm_manager.get_slot_stats()
                metrics = {
                    "slot_loss": float(loss.item()),
                    "updated": 0,
                    "num_agent_outputs": len(agent_outputs),
                }
                self.logger.log_slot(slot_id, metrics, decision, comm_stats)

                print(
                    f"[collab-infer slot {slot_id}] "
                    f"active_nodes={len(active_nodes)} "
                    f"messages={comm_stats.get('num_messages', 0)} "
                    f"delivered={comm_stats.get('delivered_messages', 0)} "
                    f"dropped={comm_stats.get('dropped_messages', 0)} "
                    f"comm_mb={comm_stats.get('comm_cost_mb', 0.0):.3f} "
                    f"avg_delay_slots={comm_stats.get('avg_delay_slots', 0.0):.2f} "
                    f"loss={loss.item():.6f}"
                )

        agent_metrics = {}
        for node_id, res_list in res_dict.items():
            res = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            out_path = os.path.join(self.logdir, f"online_test_{node_id}.txt")
            np.savetxt(out_path, res, "%d")

            moda, modp, precision, recall = evaluate(
                out_path,
                f"{self.test_dataset.gt_fname}.txt",
                self.test_dataset.base.__name__,
                self.test_dataset.frames,
            )
            f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

            agent_metrics[node_id] = {
                "moda": moda,
                "modp": modp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        self.logger.save_csv("online_infer_log.csv")
        summary = self.logger.summarize()
        avg_loss = losses / max(1, num_steps)

        avg_moda = np.mean([m["moda"] for m in agent_metrics.values()])
        avg_modp = np.mean([m["modp"] for m in agent_metrics.values()])
        avg_precision = np.mean([m["precision"] for m in agent_metrics.values()])
        avg_recall = np.mean([m["recall"] for m in agent_metrics.values()])
        avg_f1 = np.mean([m["f1"] for m in agent_metrics.values()])

        print("========== MULTI-AGENT INFER SUMMARY ==========")
        for node_id, m in agent_metrics.items():
            print(
                f"{node_id}: "
                f"moda={m['moda']:.1f}% "
                f"modp={m['modp']:.1f}% "
                f"prec={m['precision']:.1f}% "
                f"recall={m['recall']:.1f}% "
                f"f1={m['f1']:.1f}%"
            )
        print("----------------------------------------------")
        print(f"avg_agent_moda: {avg_moda:.1f}%")
        print(f"avg_agent_modp: {avg_modp:.1f}%")
        print(f"avg_agent_prec: {avg_precision:.1f}%")
        print(f"avg_agent_recall: {avg_recall:.1f}%")
        print(f"avg_agent_f1: {avg_f1:.1f}%")
        print(f"avg_slot_loss: {avg_loss:.6f}")
        print("==============================================")

        return {
            "agent_metrics": agent_metrics,
            "avg_moda": avg_moda,
            "avg_modp": avg_modp,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_slot_loss": avg_loss,
            **summary,
        }

    def run_train_then_infer(self, train_slots=None, infer_slots=None):
        print("========== PHASE 1: COLLAB TRAIN ==========")
        train_summary = self.run_train(max_slots=train_slots)

        if hasattr(self.logger, "records"):
            self.logger.records = []

        print("========== PHASE 2: COLLAB INFER ==========")
        infer_summary = self.run_infer(max_slots=infer_slots)

        return {
            "train_summary": train_summary,
            "infer_summary": infer_summary,
        }

    def run(self):
        if self.args.online_mode == "train":
            return self.run_train(max_slots=self.args.max_slots)
        elif self.args.online_mode == "infer":
            return self.run_infer(max_slots=self.args.online_infer_slots)
        elif self.args.online_mode == "train_then_infer":
            return self.run_train_then_infer(
                train_slots=self.args.online_train_slots,
                infer_slots=self.args.online_infer_slots,
            )
        else:
            raise ValueError(f"Unknown online_mode: {self.args.online_mode}")