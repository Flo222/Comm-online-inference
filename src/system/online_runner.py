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
from src.system.local_slot_loader import LocalSlotLoader


class LinearC2UCBBandit:
    """
    C2UCB-lite:
      - atomic arm: single node/view
      - shared linear parameter theta
      - optimistic score:
          mean + alpha * sqrt(x^T V^{-1} x)
      - semi-bandit update on selected views
    """

    def __init__(self, context_dim: int, lambda_reg: float = 1.0, alpha: float = 1.0):
        self.context_dim = int(context_dim)
        self.lambda_reg = float(lambda_reg)
        self.alpha = float(alpha)

        self.V = np.eye(self.context_dim, dtype=np.float64) * self.lambda_reg
        self.b = np.zeros(self.context_dim, dtype=np.float64)
        self.steps = 0

    def theta_hat(self):
        return np.linalg.solve(self.V, self.b)

    def score(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        theta = self.theta_hat()
        mean = float(theta @ x)
        invV_x = np.linalg.solve(self.V, x)
        bonus = self.alpha * float(np.sqrt(max(x @ invV_x, 0.0)))
        ucb = mean + bonus
        return ucb, mean, bonus

    def score_all(self, context_dict):
        out = {}
        for arm_id, x in context_dict.items():
            ucb, mean, bonus = self.score(x)
            out[arm_id] = {
                "x": np.asarray(x, dtype=np.float64),
                "ucb": float(ucb),
                "mean": float(mean),
                "bonus": float(bonus),
            }
        return out

    def update(self, selected_arm_ids, context_dict, observed_rewards):
        for arm_id in selected_arm_ids:
            x = np.asarray(context_dict[arm_id], dtype=np.float64)
            r = float(observed_rewards[arm_id])
            self.V += np.outer(x, x)
            self.b += r * x
        self.steps += 1


class OnlineRunner:
    """
    Clean online runner:
      - no_comm
      - full_comm (feature-map communication)
      - logits_comm_late_fusion (optional static baseline)
      - c2ucb_feature (main method)

    Removed:
      - old feature-map late fusion baseline
      - old toy subset-arm UCB
    """

    def __init__(self, model, train_dataset, test_dataset, logdir, args, optimizer=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.logdir = logdir
        self.args = args
        self.optimizer = optimizer

        self.train_local_loader = LocalSlotLoader(train_dataset)
        self.test_local_loader = LocalSlotLoader(test_dataset)

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
            rng_seed=getattr(args, "seed", None),

            channel_model=getattr(args, "comm_channel_model", "bernoulli"),
            loss_granularity=getattr(args, "loss_granularity", "message"),
            packetize_mode=getattr(args, "packetize_mode", "fixed"),
            channel_group_size=getattr(args, "channel_group_size", 16),
            packet_max_bytes=getattr(args, "packet_max_bytes", 6400),

            ge_p=getattr(args, "ge_p", 0.378563411896744),
            ge_r=getattr(args, "ge_r", 0.883314627759071),
            ge_h=getattr(args, "ge_h", 0.810),
            ge_k=getattr(args, "ge_k", 0.938571428571429),
            ge_scope=getattr(args, "ge_scope", "link"),
            ge_init_state=getattr(args, "ge_init_state", "G"),
        )

        self.comm_manager = CommManager(channel=channel)
        self.logger = OnlineLogger(logdir=logdir)

        # global running stats
        self.running_det_count_mean = None
        self.running_det_count_momentum = getattr(args, "c2ucb_count_momentum", 0.9)

        # per-view running stats
        self.prev_selected_mask = {i: 0.0 for i in range(len(self.node_ids))}
        self.prev_det_count_per_view = {i: 0.0 for i in range(len(self.node_ids))}
        self.running_det_count_mean_per_view = {i: None for i in range(len(self.node_ids))}

        # C2UCB-lite
        self.c2ucb_bandit = None
        if getattr(args, "static_policy", "full_comm") == "c2ucb_feature":
            context_dim = getattr(args, "c2ucb_context_dim", 10)
            self.c2ucb_bandit = LinearC2UCBBandit(
                context_dim=context_dim,
                lambda_reg=getattr(args, "c2ucb_lambda_reg", 1.0),
                alpha=getattr(args, "c2ucb_alpha", 1.0),
            )

    def _apply_runtime_policy(self, policy_name):
        if policy_name == "no_comm":
            self.args.static_policy = "no_comm"
            self.args.comm_topology = "no_comm"
            self.args.fusion_stage = "feature"

        elif policy_name == "full_comm":
            self.args.static_policy = "full_comm"
            self.args.comm_topology = "fully_connected"
            self.args.fusion_stage = "feature"

        elif policy_name == "logits_comm_late_fusion":
            self.args.static_policy = "logits_comm_late_fusion"
            self.args.comm_topology = "fully_connected"
            self.args.fusion_stage = "comm_logits_late"

        elif policy_name == "c2ucb_feature":
            self.args.static_policy = "c2ucb_feature"
            self.args.comm_topology = "fully_connected"
            self.args.fusion_stage = "feature"

        else:
            raise ValueError(f"Unknown runtime policy: {policy_name}")

        self.decision_policy = DecisionPolicy(
            mode=getattr(self.args, "comm_topology", "all_to_server"),
            max_active=getattr(self.args, "max_active_senders", None),
            self_loop=getattr(self.args, "comm_self_loop", False),
            bidirectional=getattr(self.args, "comm_bidirectional", True),
            custom_edges=getattr(self.args, "comm_edges", ""),
        )

    def _uses_c2ucb(self):
        return getattr(self.args, "static_policy", "full_comm") == "c2ucb_feature"

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

    def _reset_for_slot(self, slot_id):
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

    def infer_local_feature(
        self,
        slot_id: int,
        cam_idx: int,
        dataset,
        mode: str = "single_view_realistic",
        force_local_prediction: bool = True,
    ):
        """
        真正的节点本地推理。

        每个节点只做：
        1. 加载自己的单路图像
        2. 加载自己的 affine
        3. 调用 get_feat_single_cam_correct
        4. 得到 local_world_feat
        5. 可选：本地 get_output 得到 logits / heatmap / offset
        """
        loader = (
            self.train_local_loader
            if dataset is self.train_dataset
            else self.test_local_loader
        )

        local_slot = loader.load(slot_id, cam_idx)
        node = self.nodes[cam_idx]

        node.observe({
            "slot_id": slot_id,
            "frame_id": local_slot["frame_id"],
            "cam_idx": cam_idx,
            "meta": local_slot.get("meta", {}),
        })

        img = local_slot["image"].unsqueeze(0).cuda(non_blocking=True)
        affine_mat = local_slot["affine_mat"].unsqueeze(0).cuda(non_blocking=True)

        local_world_feat, aux_res = self.model.get_feat_single_cam_correct(
            img,
            affine_mat,
            cam_idx=cam_idx,
            down=self.args.down,
            visualize=False,
        )

        node.set_local_world_feat(local_world_feat)

        # logits_comm 必须本地先 get_output；
        # c2ucb 也需要本地预测来构造 context；
        # no_comm 也可以直接复用本地 prediction。
        need_local_prediction = (
            force_local_prediction
            or mode == "logits_comm"
            or getattr(self.args, "fusion_stage", "feature") == "comm_logits_late"
            or self._uses_c2ucb()
        )

        if need_local_prediction:
            local_world_heatmap, local_world_offset = self.model.get_output(
                local_world_feat
            )
            node.set_local_world_prediction(
                local_world_heatmap,
                local_world_offset,
            )

        imgs_heatmap, imgs_offset, imgs_wh = aux_res

        return {
            "slot_id": slot_id,
            "cam_idx": cam_idx,
            "frame_id": local_slot["frame_id"],
            "local_world_feat": local_world_feat,
            "imgs_heatmap": imgs_heatmap,
            "imgs_offset": imgs_offset,
            "imgs_wh": imgs_wh,
            "img_gt": local_slot["img_gt"],
            "world_gt": local_slot["world_gt"],
            "keep_cam": local_slot["keep_cam"],
            "meta": local_slot.get("meta", {}),
        }

    def _infer_all_nodes_locally(self, slot_id: int, dataset):
        local_records = []

        for cam_idx in range(len(self.nodes)):
            record = self.infer_local_feature(
                slot_id=slot_id,
                cam_idx=cam_idx,
                dataset=dataset,
                mode=getattr(self.args, "local_infer_mode", "single_view_realistic"),
                force_local_prediction=True,
            )
            local_records.append(record)

        # 这里只是为了兼容原有 loss 计算，不用于中心化特征融合
        feat_all = torch.stack(
            [self.nodes[i].local_world_feat for i in range(len(self.nodes))],
            dim=1,
        )

        imgs_heatmap = torch.cat(
            [r["imgs_heatmap"] for r in local_records],
            dim=0,
        )
        imgs_offset = torch.cat(
            [r["imgs_offset"] for r in local_records],
            dim=0,
        )
        imgs_wh = torch.cat(
            [r["imgs_wh"] for r in local_records],
            dim=0,
        )

        keep_cams = torch.stack(
            [r["keep_cam"] for r in local_records],
            dim=0,
        ).unsqueeze(0)

        imgs_gt = {}
        for key in local_records[0]["img_gt"].keys():
            imgs_gt[key] = torch.stack(
                [r["img_gt"][key] for r in local_records],
                dim=0,
            ).unsqueeze(0)

        world_gt = self._ensure_gt_batch_dim(local_records[0]["world_gt"])

        frame = local_records[0]["frame_id"]
        if not torch.is_tensor(frame):
            frame = torch.tensor([frame])
        elif frame.dim() == 0:
            frame = frame.unsqueeze(0)

        return (
            feat_all,
            imgs_heatmap,
            imgs_offset,
            imgs_wh,
            keep_cams,
            world_gt,
            imgs_gt,
            frame,
            local_records,
        )

    def _make_decision(self, selected_senders=None):
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
            edges = self.decision_policy.decide_edges(self.node_ids, server_id="server")
            active_nodes = sorted(list({sender for sender, _ in edges}))
            decision = {"edges": edges, "active_nodes": active_nodes}
        else:
            decision = self.decision_policy.decide({}, self.node_ids)
            active_nodes = decision["active_nodes"]
            edges = [(nid, "server") for nid in active_nodes]
            decision["edges"] = edges

        self.current_edges = edges
        return decision, set(active_nodes)

    def _extract_slot_features(self, imgs, affine_mats):
        feat_all, (imgs_heatmap, imgs_offset, imgs_wh) = self.model.get_feat(
            imgs, affine_mats, self.args.down
        )
        return feat_all, imgs_heatmap, imgs_offset, imgs_wh

    def _prepare_local_node_states(self, feat_all, force_local_predictions=False):
        """
        For c2ucb_feature and logits baseline, we need local prediction maps.
        """
        need_local_pred = force_local_predictions or self._uses_c2ucb() or getattr(self.args, "fusion_stage", "feature") == "comm_logits_late"

        for i, node in enumerate(self.nodes):
            local_feat = feat_all[:, i]
            node.set_local_world_feat(local_feat)

            if need_local_pred:
                world_heatmap_i, world_offset_i = self.model.get_output(local_feat)
                if hasattr(node, "set_local_world_prediction"):
                    node.set_local_world_prediction(world_heatmap_i, world_offset_i)
                else:
                    node.local_world_heatmap = world_heatmap_i
                    node.local_world_offset = world_offset_i

    def _fuse_world_features(self, world_feat_list):
        if len(world_feat_list) == 0:
            raise RuntimeError("No world features received for fusion.")

        fused = torch.stack(world_feat_list, dim=1)
        if self.model.aggregation == "max":
            fused = fused.max(dim=1)[0]
        elif self.model.aggregation == "mean":
            fused = fused.mean(dim=1)
        else:
            raise ValueError(f"Unsupported aggregation: {self.model.aggregation}")
        return fused

    def _fuse_prediction_maps_from_messages(self, heatmap_list, offset_list, mode=None):
        if len(heatmap_list) == 0 or len(offset_list) == 0:
            raise RuntimeError("No prediction maps received for fusion.")

        mode = mode or self.model.aggregation
        heatmaps = torch.stack(heatmap_list, dim=1)
        offsets = torch.stack(offset_list, dim=1)

        if mode == "mean":
            world_heatmap = heatmaps.mean(dim=1)
            world_offset = offsets.mean(dim=1)
        elif mode == "max":
            max_vals, winner_idx = heatmaps.max(dim=1)
            world_heatmap = max_vals
            gather_idx = winner_idx.unsqueeze(2).expand(-1, -1, 2, -1, -1)
            world_offset = torch.gather(offsets, dim=1, index=gather_idx).squeeze(1)
        else:
            raise ValueError(f"Unsupported prediction fusion mode: {mode}")

        return world_heatmap, world_offset

    def _estimate_avg_score_and_count(self, world_heatmap, world_offset, dataset):
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

    def _peek_count_penalty_for_view(self, view_idx, det_count):
        ref = self.running_det_count_mean_per_view[view_idx]
        if ref is None:
            return 0.0
        ref = max(float(ref), 1.0)
        return abs(float(det_count) - ref) / ref

    def _commit_det_count_for_view(self, view_idx, det_count):
        ref = self.running_det_count_mean_per_view[view_idx]
        if ref is None:
            self.running_det_count_mean_per_view[view_idx] = float(det_count)
        else:
            m = self.running_det_count_momentum
            self.running_det_count_mean_per_view[view_idx] = m * ref + (1.0 - m) * float(det_count)

    def _build_view_contexts(self, feat_all, world_gt, dataset):
        """
        Build slot-dependent context x_t(i) for each node i.
        context dim = 10:
          [cam_norm, feat_abs_mean, feat_l2_mean,
           hm_mass, hm_max, det_count_norm, avg_score,
           prev_selected, delta_det_count_norm, bias]
        """
        context_dict = {}
        local_stats = {}

        num_nodes = len(self.node_ids)

        for i, node in enumerate(self.nodes):
            local_feat = node.local_world_feat
            local_hm = node.local_world_heatmap
            local_off = node.local_world_offset

            avg_score, det_count = self._estimate_avg_score_and_count(local_hm, local_off, dataset)

            hm_sig = torch.sigmoid(local_hm)
            hm_mass = float(hm_sig.mean().item())
            hm_max = float(hm_sig.max().item())

            feat_abs_mean = float(local_feat.abs().mean().item())
            feat_l2_mean = float(torch.sqrt((local_feat ** 2).mean()).item())

            local_loss_hm = focal_loss(local_hm, world_gt["heatmap"])
            local_loss_off = regL1loss(
                local_off,
                world_gt["reg_mask"],
                world_gt["idx"],
                world_gt["offset"],
            )
            local_pred_loss = float((local_loss_hm + local_loss_off).item())

            prev_selected = float(self.prev_selected_mask[i])
            prev_det = float(self.prev_det_count_per_view[i])
            delta_det = float(det_count) - prev_det
            count_penalty = self._peek_count_penalty_for_view(i, det_count)

            x_i = np.array([
                float(i) / max(1, num_nodes - 1),
                feat_abs_mean,
                feat_l2_mean,
                hm_mass,
                hm_max,
                float(det_count) / 100.0,
                avg_score,
                prev_selected,
                delta_det / 100.0,
                1.0,
            ], dtype=np.float64)

            context_dict[i] = x_i
            local_stats[i] = {
                "avg_score": avg_score,
                "det_count": det_count,
                "count_penalty": count_penalty,
                "pred_loss": local_pred_loss,
                "hm_mass": hm_mass,
                "hm_max": hm_max,
                "feat_abs_mean": feat_abs_mean,
                "feat_l2_mean": feat_l2_mean,
            }

        return context_dict, local_stats

    def _diversity_bonus(self, selected_ids, context_dict):
        if len(selected_ids) == 0:
            return 0.0

        sigma = float(getattr(self.args, "c2ucb_div_sigma", 1.0))
        sigma2 = max(sigma * sigma, 1e-8)

        # remove bias term from diversity part
        X = np.stack([context_dict[i][:-1] for i in selected_ids], axis=1)  # [d-1, k]
        M = np.eye(len(selected_ids), dtype=np.float64) + (X.T @ X) / sigma2
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            return 0.0
        return float(logdet)

    def _set_utility(self, selected_ids, scored_contexts, context_dict):
        lambda_div = float(getattr(self.args, "c2ucb_lambda_div", 0.2))
        lambda_comm = float(getattr(self.args, "c2ucb_lambda_comm", 0.1))

        relevance = sum(scored_contexts[i]["ucb"] for i in selected_ids)
        diversity = self._diversity_bonus(selected_ids, context_dict)
        comm_cost = float(len(selected_ids))

        return float(relevance + lambda_div * diversity - lambda_comm * comm_cost)

    def _oracle_greedy_select(self, scored_contexts, context_dict):
        """
        Greedy combinatorial oracle:
          maximize set utility = sum(UCB) + lambda_div * logdet - lambda_comm * |S|
        """
        max_active = int(getattr(self.args, "c2ucb_max_active", 4))
        selected = []

        candidates = list(scored_contexts.keys())
        current_utility = 0.0

        for _ in range(max_active):
            best_i = None
            best_util = None

            for i in candidates:
                if i in selected:
                    continue
                cand = selected + [i]
                util = self._set_utility(cand, scored_contexts, context_dict)
                if best_util is None or util > best_util:
                    best_util = util
                    best_i = i

            if best_i is None:
                break

            if len(selected) > 0 and best_util <= current_utility:
                break

            selected.append(best_i)
            current_utility = best_util

        if len(selected) == 0:
            # fallback: choose the single highest-UCB view
            best_i = max(scored_contexts.keys(), key=lambda k: scored_contexts[k]["ucb"])
            selected = [best_i]
            current_utility = self._set_utility(selected, scored_contexts, context_dict)

        oracle_info = {
            "oracle_utility": float(current_utility),
            "selected_size": len(selected),
            "selected_ucb_mean": float(np.mean([scored_contexts[i]["ucb"] for i in selected])),
            "selected_mean_mean": float(np.mean([scored_contexts[i]["mean"] for i in selected])),
            "selected_bonus_mean": float(np.mean([scored_contexts[i]["bonus"] for i in selected])),
        }
        return selected, oracle_info

    def _compute_selected_view_feedback(self, selected_indices, local_stats):
        """
        Semi-bandit feedback for selected views.
        reward_i = w_score * avg_score - w_count * count_penalty - w_loss * pred_loss
        """
        w_score = float(getattr(self.args, "c2ucb_feedback_w_score", 1.0))
        w_count = float(getattr(self.args, "c2ucb_feedback_w_count", 0.5))
        w_loss = float(getattr(self.args, "c2ucb_feedback_w_loss", 0.5))

        reward_dict = {}
        for i in selected_indices:
            st = local_stats[i]
            r = (
                w_score * st["avg_score"]
                - w_count * st["count_penalty"]
                - w_loss * st["pred_loss"]
            )
            reward_dict[i] = float(r)
        return reward_dict

    def _commit_round_history(self, local_stats, selected_indices):
        selected_set = set(selected_indices)
        for i in range(len(self.node_ids)):
            self.prev_selected_mask[i] = 1.0 if i in selected_set else 0.0
            self.prev_det_count_per_view[i] = float(local_stats[i]["det_count"])
            self._commit_det_count_for_view(i, local_stats[i]["det_count"])

    def _forward_and_loss_collab_precomputed(
        self,
        dataset,
        feat_all,
        imgs_heatmap,
        imgs_offset,
        imgs_wh,
        keep_cams,
        world_gt,
        imgs_gt,
    ):
        B, N = keep_cams.shape[:2]
        imgs_gt = self._prepare_imgs_gt_for_loss(imgs_gt)

        fusion_stage = getattr(self.args, "fusion_stage", "feature")
        node_map = {node.node_id: node for node in self.nodes}

        for sender_id, receiver_id in self.current_edges:
            if fusion_stage == "comm_logits_late":
                msg = node_map[sender_id].build_logits_message(
                    receiver=receiver_id,
                    msg_type="world_prediction",
                )
            else:
                msg = node_map[sender_id].build_message(
                    receiver=receiver_id,
                    msg_type="world_feat",
                )
            self.comm_manager.send(msg)

        agent_outputs = {}
        agent_losses = []

        for receiver_id in self.node_ids:
            recv_msgs = self.comm_manager.collect_for(receiver_id)
            recv_from = []

            if fusion_stage == "feature":
                # feat_list = [node_map[receiver_id].local_world_feat]

                # for msg in recv_msgs:
                #     wf = msg.payload.get("world_feat", None)
                #     sender = msg.payload.get("node_id", None)
                #     if wf is None:
                #         continue

                #     wf = wf.to(feat_all.dtype)
                #     if sender == receiver_id:
                #         continue

                #     feat_list.append(wf)
                #     recv_from.append(sender)

                # fused_feat = self._fuse_world_features(feat_list)
                feat_mask_list = [
                    (node_map[receiver_id].local_world_feat, None)
                ]

                for msg in recv_msgs:
                    wf = msg.payload.get("world_feat", None)
                    sender = msg.payload.get("node_id", None)

                    if wf is None:
                        continue

                    if sender == receiver_id:
                        continue

                    wf = wf.to(feat_all.dtype)

                    channel_valid_mask = msg.payload.get("channel_valid_mask", None)
                    if channel_valid_mask is not None:
                        channel_valid_mask = channel_valid_mask.to(wf.device)

                    feat_mask_list.append((wf, channel_valid_mask))
                    recv_from.append(sender)

                fused_feat = self._fuse_world_features_mask_aware(feat_mask_list)
                world_heatmap_i, world_offset_i = self.model.get_output(fused_feat)

                agent_outputs[receiver_id] = {
                    "fused_feat": fused_feat,
                    "world_heatmap": world_heatmap_i,
                    "world_offset": world_offset_i,
                    "recv_from": recv_from,
                }

            elif fusion_stage == "comm_logits_late":
                heatmap_list = [node_map[receiver_id].local_world_heatmap]
                offset_list = [node_map[receiver_id].local_world_offset]

                for msg in recv_msgs:
                    hm = msg.payload.get("world_heatmap", None)
                    off = msg.payload.get("world_offset", None)
                    sender = msg.payload.get("node_id", None)

                    if hm is None or off is None:
                        continue
                    if sender == receiver_id:
                        continue

                    heatmap_list.append(hm.to(feat_all.dtype))
                    offset_list.append(off.to(feat_all.dtype))
                    recv_from.append(sender)

                world_heatmap_i, world_offset_i = self._fuse_prediction_maps_from_messages(
                    heatmap_list,
                    offset_list,
                    mode=self.model.aggregation,
                )

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

    # def _run_one_slot(self, slot_sample, dataset, train_mode=False):
    #     slot_id = slot_sample["slot_id"]
    #     self._reset_for_slot(slot_sample)

    #     imgs, affine_mats, keep_cams, world_gt, imgs_gt, frame = self._normalize_slot_tensors(slot_sample)
    #     imgs = imgs.cuda()
    #     affine_mats = affine_mats.cuda()

    #     feat_all, imgs_heatmap, imgs_offset, imgs_wh = self._extract_slot_features(imgs, affine_mats)
    #     self._prepare_local_node_states(feat_all)

    #     local_stats = None
    #     oracle_info = {}
    #     selected_indices = None

    #     if self._uses_c2ucb():
    #         context_dict, local_stats = self._build_view_contexts(feat_all, world_gt, dataset)
    #         scored_contexts = self.c2ucb_bandit.score_all(context_dict)
    #         selected_indices, oracle_info = self._oracle_greedy_select(scored_contexts, context_dict)
    #         decision, active_nodes = self._make_decision(selected_indices)
    #     else:
    #         decision, active_nodes = self._make_decision(None)

    #     loss, agent_outputs = self._forward_and_loss_collab_precomputed(
    #         dataset=dataset,
    #         feat_all=feat_all,
    #         imgs_heatmap=imgs_heatmap,
    #         imgs_offset=imgs_offset,
    #         imgs_wh=imgs_wh,
    #         keep_cams=keep_cams,
    #         world_gt=world_gt,
    #         imgs_gt=imgs_gt,
    #     )

    #     comm_stats = self.comm_manager.get_slot_stats()

    #     c2ucb_info = {}
    #     if self._uses_c2ucb():
    #         reward_dict = self._compute_selected_view_feedback(selected_indices, local_stats)
    #         self.c2ucb_bandit.update(selected_indices, context_dict, reward_dict)
    #         self._commit_round_history(local_stats, selected_indices)

    #         c2ucb_info = {
    #             "selected_indices": selected_indices,
    #             "selected_str": ",".join(str(i) for i in selected_indices),
    #             "avg_feedback_selected": float(np.mean([reward_dict[i] for i in selected_indices])),
    #             "oracle_utility": oracle_info["oracle_utility"],
    #             "oracle_selected_size": oracle_info["selected_size"],
    #             "oracle_ucb_mean": oracle_info["selected_ucb_mean"],
    #             "oracle_mean_mean": oracle_info["selected_mean_mean"],
    #             "oracle_bonus_mean": oracle_info["selected_bonus_mean"],
    #         }

    #     return {
    #         "slot_id": slot_id,
    #         "loss": loss,
    #         "agent_outputs": agent_outputs,
    #         "frame": frame,
    #         "decision": decision,
    #         "active_nodes": active_nodes,
    #         "comm_stats": comm_stats,
    #         "c2ucb_info": c2ucb_info,
    #     }

    def _run_one_slot(self, slot_id, dataset, train_mode=False):
        self._reset_for_slot(slot_id)

        (
            feat_all,
            imgs_heatmap,
            imgs_offset,
            imgs_wh,
            keep_cams,
            world_gt,
            imgs_gt,
            frame,
            local_records,
        ) = self._infer_all_nodes_locally(slot_id, dataset)

        local_stats = None
        oracle_info = {}
        selected_indices = None

        if self._uses_c2ucb():
            context_dict, local_stats = self._build_view_contexts(
                feat_all,
                world_gt,
                dataset,
            )
            scored_contexts = self.c2ucb_bandit.score_all(context_dict)
            selected_indices, oracle_info = self._oracle_greedy_select(
                scored_contexts,
                context_dict,
            )
            decision, active_nodes = self._make_decision(selected_indices)
        else:
            decision, active_nodes = self._make_decision(None)

        loss, agent_outputs = self._forward_and_loss_collab_precomputed(
            dataset=dataset,
            feat_all=feat_all,
            imgs_heatmap=imgs_heatmap,
            imgs_offset=imgs_offset,
            imgs_wh=imgs_wh,
            keep_cams=keep_cams,
            world_gt=world_gt,
            imgs_gt=imgs_gt,
        )

        comm_stats = self.comm_manager.get_slot_stats()

        c2ucb_info = {}
        if self._uses_c2ucb():
            reward_dict = self._compute_selected_view_feedback(
                selected_indices,
                local_stats,
            )
            self.c2ucb_bandit.update(
                selected_indices,
                context_dict,
                reward_dict,
            )
            self._commit_round_history(local_stats, selected_indices)

            c2ucb_info = {
                "selected_indices": selected_indices,
                "selected_str": ",".join(str(i) for i in selected_indices),
                "avg_feedback_selected": float(
                    np.mean([reward_dict[i] for i in selected_indices])
                ),
                "oracle_utility": oracle_info["oracle_utility"],
                "oracle_selected_size": oracle_info["selected_size"],
                "oracle_ucb_mean": oracle_info["selected_ucb_mean"],
                "oracle_mean_mean": oracle_info["selected_mean_mean"],
                "oracle_bonus_mean": oracle_info["selected_bonus_mean"],
            }

        return {
            "slot_id": slot_id,
            "loss": loss,
            "agent_outputs": agent_outputs,
            "frame": frame,
            "decision": decision,
            "active_nodes": active_nodes,
            "comm_stats": comm_stats,
            "c2ucb_info": c2ucb_info,
        }

    def run_train(self, max_slots=None):
        if self.optimizer is None:
            raise ValueError("online train mode requires optimizer")

        self.model.train()
        losses = 0.0
        num_steps = 0

        for slot_id in range(len(self.train_dataset)):
            if max_slots is not None and slot_id >= max_slots:
                break

            out = self._run_one_slot(
                slot_id,
                dataset=self.train_dataset,
                train_mode=True,
            )
            slot_id = out["slot_id"]
            loss = out["loss"]
            decision = out["decision"]
            active_nodes = out["active_nodes"]
            comm_stats = out["comm_stats"]
            c2ucb_info = out["c2ucb_info"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses += loss.item()
            num_steps += 1

            metrics = {
                "slot_loss": float(loss.item()),
                "updated": 1,
                "num_agent_outputs": len(out["agent_outputs"]),
                "selected_senders": c2ucb_info.get("selected_str", ""),
                "oracle_utility": c2ucb_info.get("oracle_utility", 0.0),
                "oracle_ucb_mean": c2ucb_info.get("oracle_ucb_mean", 0.0),
                "oracle_bonus_mean": c2ucb_info.get("oracle_bonus_mean", 0.0),
            }
            self.logger.log_slot(slot_id, metrics, decision, comm_stats)

            policy_name = getattr(self.args, "static_policy", "unknown")
            extra = ""
            if self._uses_c2ucb():
                extra = (
                    f" selected={c2ucb_info.get('selected_str', '')}"
                    f" oracle_util={c2ucb_info.get('oracle_utility', 0.0):.4f}"
                    f" avg_ucb={c2ucb_info.get('oracle_ucb_mean', 0.0):.4f}"
                    f" avg_bonus={c2ucb_info.get('oracle_bonus_mean', 0.0):.4f}"
                )

            print(
                f"[collab-train slot {slot_id}] "
                f"policy={policy_name} "
                f"fusion_stage={getattr(self.args, 'fusion_stage', 'feature')} "
                f"active_nodes={len(active_nodes)} "
                f"messages={comm_stats.get('num_messages', 0)} "
                f"delivered={comm_stats.get('delivered_messages', 0)} "
                f"dropped={comm_stats.get('dropped_messages', 0)} "
                f"comm_mb={comm_stats.get('comm_cost_mb', 0.0):.3f} "
                f"avg_delay_slots={comm_stats.get('avg_delay_slots', 0.0):.2f} "
                f"loss={loss.item():.6f}"
                f"{extra}"
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

        c2ucb_trace = []

        with torch.no_grad():
            for slot_id in range(len(self.test_dataset)):
                if max_slots is not None and slot_id >= max_slots:
                    break

                out = self._run_one_slot(
                    slot_id,
                    dataset=self.test_dataset,
                    train_mode=False,
                )
                slot_id = out["slot_id"]
                loss = out["loss"]
                agent_outputs = out["agent_outputs"]
                frame = out["frame"]
                decision = out["decision"]
                active_nodes = out["active_nodes"]
                comm_stats = out["comm_stats"]
                c2ucb_info = out["c2ucb_info"]

                losses += loss.item()
                num_steps += 1

                for node_id, node_out in agent_outputs.items():
                    self._decode_and_collect(
                        self.test_dataset,
                        node_out["world_heatmap"],
                        node_out["world_offset"],
                        frame,
                        res_dict[node_id],
                    )

                actual_total_comm_mb += float(comm_stats.get("comm_cost_mb", 0.0))
                actual_total_messages += int(comm_stats.get("num_messages", 0))

                num_nodes = len(self.node_ids)
                if num_nodes > 1:
                    baseline_full_comm_total_messages += num_nodes * (num_nodes - 1)

                fusion_stage = getattr(self.args, "fusion_stage", "feature")
                per_msg_mb = 0.0
                if fusion_stage == "comm_logits_late":
                    hm = getattr(self.nodes[0], "local_world_heatmap", None)
                    off = getattr(self.nodes[0], "local_world_offset", None)
                    if hm is not None and off is not None:
                        elem_count = hm.numel() + off.numel()
                        per_msg_mb = elem_count * int(self.nodes[0].quant_bits) / 8.0 / 1e6
                else:
                    first_feat = self.nodes[0].local_world_feat
                    if first_feat is not None:
                        compressed_feat = self.nodes[0]._compress_world_feat(first_feat)
                        per_msg_mb = compressed_feat.numel() * int(self.nodes[0].quant_bits) / 8.0 / 1e6

                baseline_full_comm_total_mb += per_msg_mb * num_nodes * (num_nodes - 1)

                metrics = {
                    "slot_loss": float(loss.item()),
                    "updated": 0,
                    "num_agent_outputs": len(agent_outputs),
                    "selected_senders": c2ucb_info.get("selected_str", ""),
                    "oracle_utility": c2ucb_info.get("oracle_utility", 0.0),
                    "oracle_ucb_mean": c2ucb_info.get("oracle_ucb_mean", 0.0),
                    "oracle_bonus_mean": c2ucb_info.get("oracle_bonus_mean", 0.0),
                    "avg_feedback_selected": c2ucb_info.get("avg_feedback_selected", 0.0),
                }
                self.logger.log_slot(slot_id, metrics, decision, comm_stats)

                policy_name = getattr(self.args, "static_policy", "unknown")
                extra = ""
                if self._uses_c2ucb():
                    extra = (
                        f" selected={c2ucb_info.get('selected_str', '')}"
                        f" oracle_util={c2ucb_info.get('oracle_utility', 0.0):.4f}"
                        f" avg_ucb={c2ucb_info.get('oracle_ucb_mean', 0.0):.4f}"
                        f" avg_bonus={c2ucb_info.get('oracle_bonus_mean', 0.0):.4f}"
                        f" avg_feedback={c2ucb_info.get('avg_feedback_selected', 0.0):.4f}"
                    )
                    c2ucb_trace.append(c2ucb_info)

                print(
                    f"[collab-infer slot {slot_id}] "
                    f"policy={policy_name} "
                    f"fusion_stage={fusion_stage} "
                    f"active_nodes={len(active_nodes)} "
                    f"messages={comm_stats.get('num_messages', 0)} "
                    f"delivered={comm_stats.get('delivered_messages', 0)} "
                    f"dropped={comm_stats.get('dropped_messages', 0)} "
                    f"packets={comm_stats.get('total_packets', 0)} "
                    f"lost_packets={comm_stats.get('lost_packets', 0)} "
                    f"pkt_loss={comm_stats.get('packet_loss_rate', 0.0):.3f} "
                    f"partial_msgs={comm_stats.get('partial_messages', 0)} "
                    f"comm_mb={comm_stats.get('comm_cost_mb', 0.0):.3f} "
                    f"avg_delay_slots={comm_stats.get('avg_delay_slots', 0.0):.2f} "
                    f"loss={loss.item():.6f}"
                    f"{extra}"
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

        if self._uses_c2ucb() and self.c2ucb_bandit is not None:
            theta = self.c2ucb_bandit.theta_hat()
            print("========== C2UCB-LITE SUMMARY ==========")
            print(f"context_dim: {self.c2ucb_bandit.context_dim}")
            print(f"c2ucb_steps: {self.c2ucb_bandit.steps}")
            print(f"theta_hat: {np.array2string(theta, precision=4, suppress_small=True)}")
            print("========================================")

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

    def _fuse_world_features_mask_aware(self, feat_mask_list):
        """
        feat_mask_list: list of (feat, channel_valid_mask)

        feat:
        [B, C, H, W] or [C, H, W]

        channel_valid_mask:
        [C], bool
        True = valid
        False = missing
        """

        feats = []
        masks = []

        for feat, mask in feat_mask_list:
            if mask is None:
                if feat.dim() == 4:
                    C = feat.shape[1]
                else:
                    C = feat.shape[0]
                mask = torch.ones(C, device=feat.device, dtype=torch.bool)

            mask = mask.to(device=feat.device)

            if feat.dim() == 4:
                mask_view = mask.view(1, -1, 1, 1)
            else:
                mask_view = mask.view(-1, 1, 1)

            feats.append(feat)
            masks.append(mask_view)

        if self.model.aggregation == "max":
            masked_feats = []
            for feat, mask_view in zip(feats, masks):
                masked_feats.append(
                    torch.where(
                        mask_view,
                        feat,
                        torch.full_like(feat, -1e4),
                    )
                )

            fused = torch.stack(masked_feats, dim=1).max(dim=1)[0]
            return fused

        elif self.model.aggregation == "mean":
            weighted_sum = None
            valid_count = None

            for feat, mask_view in zip(feats, masks):
                valid = mask_view.to(dtype=feat.dtype)
                cur = feat * valid

                if weighted_sum is None:
                    weighted_sum = cur
                    valid_count = valid
                else:
                    weighted_sum = weighted_sum + cur
                    valid_count = valid_count + valid

            fused = weighted_sum / valid_count.clamp_min(1.0)
            return fused

        else:
            raise ValueError(f"Unsupported aggregation: {self.model.aggregation}")