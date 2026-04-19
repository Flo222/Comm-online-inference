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


class SimpleUCB:
    """
    最简单多臂 UCB：
      - 每个 arm 先至少试一次
      - 之后按 value + c * sqrt(log(t)/n) 选
    """

    def __init__(self, arms, c=1.0):
        self.arms = list(arms)
        self.c = float(c)
        self.counts = {a: 0 for a in self.arms}
        self.values = {a: 0.0 for a in self.arms}
        self.total_steps = 0

    def select_arm(self):
        for a in self.arms:
            if self.counts[a] == 0:
                return a

        t = max(self.total_steps, 1)
        scores = {}
        for a in self.arms:
            bonus = self.c * np.sqrt(np.log(t) / self.counts[a])
            scores[a] = self.values[a] + bonus
        return max(scores, key=scores.get)

    def update(self, arm, reward):
        self.total_steps += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        old = self.values[arm]
        self.values[arm] = old + (reward - old) / n

    def summary(self):
        return {
            arm: {
                "count": self.counts[arm],
                "value": self.values[arm],
            }
            for arm in self.arms
        }


class OnlineRunner:
    """
    Collaborative Inference with channel-aware communication.

    Supports:
      - fixed static policies:
          * no_comm
          * full_comm (fully_connected + feature fusion)
          * late_fusion_logits (fully_connected + output-level fusion)

      - subset bandit on top of feature-map communication:
          * 每个 slot 选择一个 sender 子集
          * 选中的 sender 向所有节点广播 feature map
          * receiver 始终是全部节点
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
        self.logger = OnlineLogger(logdir=logdir)

        self.subset_bandit = None
        self.subset_arms = []

        # 用于 count_penalty 的运行统计
        self.running_det_count_mean = None
        self.running_det_count_momentum = getattr(args, "subset_reward_count_momentum", 0.9)

        if getattr(args, "use_subset_bandit", False):
            self.subset_arms = self._parse_subset_arms(
                getattr(args, "subset_arms", ""),
                len(self.node_ids)
            )
            if len(self.subset_arms) == 0:
                raise ValueError("use_subset_bandit=True but no valid subset_arms provided")

            arm_names = [self._subset_to_name(s) for s in self.subset_arms]
            self.subset_bandit = SimpleUCB(
                arms=arm_names,
                c=getattr(args, "subset_ucb_c", 1.0),
            )

    def _subset_to_name(self, subset):
        if len(subset) == len(self.node_ids):
            return "all"
        return ",".join(str(x) for x in subset)

    def _name_to_subset(self, name):
        if name.lower() in ["all", "full", "full_comm"]:
            return list(range(len(self.node_ids)))
        if name.strip() == "":
            return []
        return [int(x.strip()) for x in name.split(",") if x.strip() != ""]

    def _parse_subset_arms(self, text, num_nodes):
        arms = []
        text = text.strip()
        if not text:
            return arms

        items = [x.strip() for x in text.split(";") if x.strip()]
        for item in items:
            if item.lower() in ["all", "full", "full_comm"]:
                subset = list(range(num_nodes))
            else:
                subset = [int(x.strip()) for x in item.split(",") if x.strip()]
                subset = sorted(list(dict.fromkeys(subset)))

            if len(subset) == 0:
                continue
            if min(subset) < 0 or max(subset) >= num_nodes:
                raise ValueError(f"Invalid subset arm: {item}")
            arms.append(subset)

        uniq = []
        seen = set()
        for s in arms:
            key = tuple(s)
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq

    def _apply_runtime_policy(self, policy_name):
        if policy_name == "no_comm":
            self.args.static_policy = "no_comm"
            self.args.comm_topology = "no_comm"
            self.args.fusion_stage = "feature"

        elif policy_name == "full_comm":
            self.args.static_policy = "full_comm"
            self.args.comm_topology = "fully_connected"
            self.args.fusion_stage = "feature"

        elif policy_name == "late_fusion_logits":
            self.args.static_policy = "late_fusion_logits"
            self.args.comm_topology = "fully_connected"
            self.args.fusion_stage = "late_logits"

        else:
            raise ValueError(f"Unknown runtime policy: {policy_name}")

        self.decision_policy = DecisionPolicy(
            mode=getattr(self.args, "comm_topology", "all_to_server"),
            max_active=getattr(self.args, "max_active_senders", None),
            self_loop=getattr(self.args, "comm_self_loop", False),
            bidirectional=getattr(self.args, "comm_bidirectional", True),
            custom_edges=getattr(self.args, "comm_edges", ""),
        )

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

    def _late_fuse_outputs(self, feat_list):
        heatmap_list = []
        offset_list = []

        for wf in feat_list:
            hm_i, off_i = self.model.get_output(wf)
            heatmap_list.append(hm_i)
            offset_list.append(off_i)

        heatmaps = torch.stack(heatmap_list, dim=1)
        offsets = torch.stack(offset_list, dim=1)

        if self.model.aggregation == "mean":
            world_heatmap = heatmaps.mean(dim=1)
            world_offset = offsets.mean(dim=1)
        elif self.model.aggregation == "max":
            max_vals, winner_idx = heatmaps.max(dim=1)
            world_heatmap = max_vals
            gather_idx = winner_idx.unsqueeze(2).expand(-1, -1, 2, -1, -1)
            world_offset = torch.gather(offsets, dim=1, index=gather_idx).squeeze(1)
        else:
            raise ValueError(f"Unsupported aggregation for late fusion: {self.model.aggregation}")

        return world_heatmap, world_offset

    def _estimate_avg_score_and_count(self, world_heatmap, world_offset, dataset):
        """
        返回:
          avg_score: 保留下来的框平均分数
          det_count: 保留下来的框数量
        """
        xys = mvdet_decode(
            torch.sigmoid(world_heatmap),
            world_offset,
            reduce=dataset.world_reduce,
        ).detach().cpu()

        scores = xys[:, :, 2]
        keep = scores > self.args.cls_thres
        det_count = int(keep.sum().item())

        if det_count == 0:
            return 0.0, 0

        avg_score = float(scores[keep].mean().item())
        return avg_score, det_count

    def _estimate_count_penalty(self, det_count):
        """
        根据历史运行均值估计 count penalty。
        偏离历史均值越大，惩罚越大。
        """
        if self.running_det_count_mean is None:
            self.running_det_count_mean = float(det_count)
            return 0.0

        ref = max(float(self.running_det_count_mean), 1.0)
        penalty = abs(float(det_count) - ref) / ref

        m = self.running_det_count_momentum
        self.running_det_count_mean = m * self.running_det_count_mean + (1.0 - m) * float(det_count)

        return float(penalty)

    def _slot_comm_and_nodes(self, slot_sample, selected_senders=None):
        slot_id = slot_sample["slot_id"]

        if hasattr(self.comm_manager, "reset_slot_stats"):
            self.comm_manager.reset_slot_stats()
        elif hasattr(self.comm_manager, "reset_slot"):
            self.comm_manager.reset_slot()

        if hasattr(self.comm_manager, "clear_mailboxes"):
            self.comm_manager.clear_mailboxes()

        if hasattr(self.comm_manager, "deliver_pending"):
            self.comm_manager.deliver_pending(slot_id)

        for node in self.nodes:
            node.reset_slot()
            node.observe(slot_sample)

        if selected_senders is not None:
            sender_node_ids = [self.node_ids[i] for i in selected_senders]
            edges = []
            for s in sender_node_ids:
                for r in self.node_ids:
                    if s == r:
                        continue
                    edges.append((s, r))
            decision = {
                "edges": edges,
                "active_nodes": sender_node_ids,
                "selected_senders": sender_node_ids,
            }
            self.current_edges = edges
            return decision, set(sender_node_ids)

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
            decision = self.decision_policy.decide(slot_sample, self.node_ids)
            active_nodes = decision["active_nodes"]
            edges = [(nid, "server") for nid in active_nodes]
            decision["edges"] = edges

        self.current_edges = edges
        return decision, set(active_nodes)

    def _forward_and_loss_collab(self, dataset, imgs, affine_mats, keep_cams, world_gt, imgs_gt, slot_id):
        B, N = imgs.shape[:2]
        imgs_gt = self._prepare_imgs_gt_for_loss(imgs_gt)

        feat_all, (imgs_heatmap, imgs_offset, imgs_wh) = self.model.get_feat(
            imgs, affine_mats, self.args.down
        )

        for i, node in enumerate(self.nodes):
            node.set_local_world_feat(feat_all[:, i])

        node_map = {node.node_id: node for node in self.nodes}

        for sender_id, receiver_id in self.current_edges:
            msg = node_map[sender_id].build_message(
                receiver=receiver_id,
                msg_type="world_feat",
            )
            self.comm_manager.send(msg)

        agent_outputs = {}
        agent_losses = []

        for receiver_id in self.node_ids:
            recv_msgs = self.comm_manager.collect_for(receiver_id)

            feat_list = [node_map[receiver_id].local_world_feat]

            recv_from = []
            for msg in recv_msgs:
                wf = msg.payload.get("world_feat", None)
                sender = msg.payload.get("node_id", None)
                if wf is None:
                    continue

                wf = wf.to(feat_all.dtype)

                if sender == receiver_id:
                    continue

                feat_list.append(wf)
                recv_from.append(sender)

            fusion_stage = getattr(self.args, "fusion_stage", "feature")

            if fusion_stage == "feature":
                fused_feat = self._fuse_world_features(feat_list)
                world_heatmap_i, world_offset_i = self.model.get_output(fused_feat)

                agent_outputs[receiver_id] = {
                    "fused_feat": fused_feat,
                    "world_heatmap": world_heatmap_i,
                    "world_offset": world_offset_i,
                    "recv_from": recv_from,
                }

            elif fusion_stage == "late_logits":
                world_heatmap_i, world_offset_i = self._late_fuse_outputs(feat_list)

                agent_outputs[receiver_id] = {
                    "fused_feat": None,
                    "world_heatmap": world_heatmap_i,
                    "world_offset": world_offset_i,
                    "recv_from": recv_from,
                }

            else:
                raise ValueError(f"Unknown fusion_stage: {fusion_stage}")

            loss_w_hm_i = focal_loss(world_heatmap_i, world_gt["heatmap"])
            loss_w_off_i = regL1loss(
                world_offset_i, world_gt["reg_mask"], world_gt["idx"], world_gt["offset"]
            )
            agent_losses.append(loss_w_hm_i + loss_w_off_i)

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

            loss, agent_outputs = self._forward_and_loss_collab(
                self.train_dataset, imgs, affine_mats, keep_cams, world_gt, imgs_gt, slot_id
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses += loss.item()
            num_steps += 1

            comm_stats = self.comm_manager.get_slot_stats()

            metrics = {
                "slot_loss": float(loss.item()),
                "updated": 1,
                "num_agent_outputs": len(agent_outputs),
            }
            self.logger.log_slot(slot_id, metrics, decision, comm_stats)

            print(
                f"[collab-train slot {slot_id}] "
                f"policy={getattr(self.args, 'static_policy', 'unknown')} "
                f"fusion_stage={getattr(self.args, 'fusion_stage', 'feature')} "
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

        actual_total_comm_mb = 0.0
        actual_total_messages = 0

        baseline_full_comm_total_mb = 0.0
        baseline_full_comm_total_messages = 0

        with torch.no_grad():
            for i, slot_sample in enumerate(self.test_stream):
                if max_slots is not None and i >= max_slots:
                    break

                slot_id = slot_sample["slot_id"]
                chosen_subset_name = None
                chosen_subset = None
                reward = None
                avg_score = 0.0
                det_count = 0
                count_penalty = 0.0

                if self.subset_bandit is not None:
                    chosen_subset_name = self.subset_bandit.select_arm()
                    chosen_subset = self._name_to_subset(chosen_subset_name)

                    self.args.fusion_stage = "feature"
                    self.args.static_policy = "subset_bandit"

                    decision, active_nodes = self._slot_comm_and_nodes(
                        slot_sample,
                        selected_senders=chosen_subset
                    )
                else:
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

                actual_total_comm_mb += float(comm_stats.get("comm_cost_mb", 0.0))
                actual_total_messages += int(comm_stats.get("num_messages", 0))

                num_nodes = len(self.node_ids)
                if num_nodes > 1:
                    baseline_full_comm_total_messages += num_nodes * (num_nodes - 1)

                per_msg_mb = 0.0
                first_feat = self.nodes[0].local_world_feat
                if first_feat is not None:
                    compressed_feat = self.nodes[0]._compress_world_feat(first_feat)
                    per_msg_mb = compressed_feat.numel() * int(self.nodes[0].quant_bits) / 8.0 / 1e6
                    baseline_full_comm_total_mb += per_msg_mb * num_nodes * (num_nodes - 1)

                if self.subset_bandit is not None:
                    lambda_comm = getattr(self.args, "subset_reward_lambda", 0.2)
                    beta_score = getattr(self.args, "subset_reward_beta_score", 0.5)
                    beta_count = getattr(self.args, "subset_reward_beta_count", 0.3)

                    full_mb = max(per_msg_mb * num_nodes * (num_nodes - 1), 1e-12)
                    normalized_comm = float(comm_stats.get("comm_cost_mb", 0.0)) / full_mb

                    rep_node_id = self.node_ids[0]
                    avg_score, det_count = self._estimate_avg_score_and_count(
                        agent_outputs[rep_node_id]["world_heatmap"],
                        agent_outputs[rep_node_id]["world_offset"],
                        self.test_dataset,
                    )
                    count_penalty = self._estimate_count_penalty(det_count)

                    reward = (
                        -float(loss.item())
                        + beta_score * avg_score
                        - beta_count * count_penalty
                        - lambda_comm * normalized_comm
                    )

                    self.subset_bandit.update(chosen_subset_name, reward)

                metrics = {
                    "slot_loss": float(loss.item()),
                    "updated": 0,
                    "num_agent_outputs": len(agent_outputs),
                    "chosen_subset": chosen_subset_name if chosen_subset_name is not None else "",
                    "avg_score": avg_score,
                    "det_count": det_count,
                    "count_penalty": count_penalty,
                    "reward": reward if reward is not None else 0.0,
                }
                self.logger.log_slot(slot_id, metrics, decision, comm_stats)

                policy_str = chosen_subset_name if chosen_subset_name is not None else getattr(self.args, 'static_policy', 'unknown')

                print(
                    f"[collab-infer slot {slot_id}] "
                    f"policy={policy_str} "
                    f"fusion_stage={getattr(self.args, 'fusion_stage', 'feature')} "
                    f"active_nodes={len(active_nodes)} "
                    f"messages={comm_stats.get('num_messages', 0)} "
                    f"delivered={comm_stats.get('delivered_messages', 0)} "
                    f"dropped={comm_stats.get('dropped_messages', 0)} "
                    f"comm_mb={comm_stats.get('comm_cost_mb', 0.0):.3f} "
                    f"avg_delay_slots={comm_stats.get('avg_delay_slots', 0.0):.2f} "
                    f"loss={loss.item():.6f} "
                    f"avg_score={avg_score:.4f} "
                    f"det_count={det_count} "
                    f"count_penalty={count_penalty:.4f} "
                    f"reward={(reward if reward is not None else 0.0):.6f}"
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

        comm_saving_ratio = 0.0
        if baseline_full_comm_total_mb > 0:
            comm_saving_ratio = 1.0 - actual_total_comm_mb / baseline_full_comm_total_mb

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

        print("========== COMM SAVING SUMMARY ==========")
        print(f"actual_total_comm_mb: {actual_total_comm_mb:.3f}")
        print(f"full_comm_total_comm_mb: {baseline_full_comm_total_mb:.3f}")
        print(f"comm_saving_ratio_vs_full_comm: {comm_saving_ratio * 100:.2f}%")
        print(f"actual_total_messages: {actual_total_messages}")
        print(f"full_comm_total_messages: {baseline_full_comm_total_messages}")
        print("=========================================")

        if self.subset_bandit is not None:
            print("========== SUBSET BANDIT SUMMARY ==========")
            for arm, info in self.subset_bandit.summary().items():
                print(f"{arm}: count={info['count']} avg_reward={info['value']:.6f}")
            print("==========================================")

        print("==============================================")

        return {
            "agent_metrics": agent_metrics,
            "avg_moda": avg_moda,
            "avg_modp": avg_modp,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_slot_loss": avg_loss,
            "actual_total_comm_mb": actual_total_comm_mb,
            "full_comm_total_comm_mb": baseline_full_comm_total_mb,
            "comm_saving_ratio_vs_full_comm": comm_saving_ratio,
            "actual_total_messages": actual_total_messages,
            "full_comm_total_messages": baseline_full_comm_total_messages,
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