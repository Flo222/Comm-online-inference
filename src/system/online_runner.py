import os
import numpy as np
import torch

from src.stream.online_stream import OnlineStream
from src.system.node import Node
from src.system.comm_manager import CommManager
from src.system.decision import DecisionPolicy
from src.system.logger import OnlineLogger
from src.loss import focal_loss, regL1loss
from src.evaluation.evaluate import evaluate
from src.utils.decode import mvdet_decode
from src.utils.nms import nms


class OnlineRunner:
    """
    Collaborative Inference v1.1 (stable):
    - run original multi-view get_feat ONCE
    - treat each per-view world feature feat_all[:, i] as one node's local message
    - server fuses received world features by max/mean
    - detection head runs on fused world feature

    Compared with the previous aggressive version:
    - DO NOT run one extra local get_feat() per node
    - keep the feature path much closer to the original baseline
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

        ref_dataset = train_dataset
        num_cam = ref_dataset.num_cam if hasattr(ref_dataset, "num_cam") else ref_dataset.base.num_cam
        self.nodes = [Node(node_id=f"node_{i}", cam_id=i) for i in range(num_cam)]
        self.node_ids = [n.node_id for n in self.nodes]

        self.comm_manager = CommManager()
        self.decision_policy = DecisionPolicy(mode="all_active")
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
        self.comm_manager.reset_slot()

        for node in self.nodes:
            node.reset_slot()

        decision = self.decision_policy.decide(slot_sample, self.node_ids)
        active_nodes = set(decision["active_nodes"])

        for node in self.nodes:
            if node.node_id not in active_nodes:
                continue
            node.observe(slot_sample)

        return decision, active_nodes

    def _forward_and_loss_collab(self, dataset, imgs, affine_mats, keep_cams, world_gt, imgs_gt):
        """
        Stable collaborative inference/training:

        1. Run original baseline feature extraction ONCE:
           feat_all, (imgs_heatmap, imgs_offset, imgs_wh) = self.model.get_feat(...)
        2. Treat feat_all[:, i] as node i's local world feature
        3. Send these local features as messages
        4. Fuse received world features at server
        5. Run original detection head on fused world feature

        This makes messages truly affect the final result, while keeping the
        feature path close to the original baseline.
        """
        B, N = imgs.shape[:2]
        imgs_gt = self._prepare_imgs_gt_for_loss(imgs_gt)

        # === original baseline multi-view feature extraction (ONE forward only) ===
        feat_all, (imgs_heatmap, imgs_offset, imgs_wh) = self.model.get_feat(
            imgs, affine_mats, self.args.down
        )
        # feat_all: [B, N, C, H, W]

        # === collaborative part: node messages are per-view world features ===
        active_node_ids = set(self.node_ids)

        for node in self.nodes:
            if node.node_id not in active_node_ids:
                continue

            local_world_feat = feat_all[:, node.cam_id]   # [B, C, H, W]
            node.set_local_world_feat(local_world_feat)

            msg = node.build_message(msg_type="world_feat")
            self.comm_manager.send(msg)

        server_msgs = self.comm_manager.collect_for("server")
        recv_world_feats = [msg.payload["world_feat"] for msg in server_msgs]

        fused_world_feat = self._fuse_world_features(recv_world_feats)
        world_heatmap, world_offset = self.model.get_output(fused_world_feat)

        # === keep original loss structure as much as possible ===
        loss_w_hm = focal_loss(world_heatmap, world_gt["heatmap"])
        loss_w_off = regL1loss(world_offset, world_gt["reg_mask"], world_gt["idx"], world_gt["offset"])

        loss_img_hm = focal_loss(
            imgs_heatmap,
            imgs_gt["heatmap"],
            keep_cams.view(B * N, 1, 1, 1)
        )
        loss_img_off = regL1loss(imgs_offset, imgs_gt["reg_mask"], imgs_gt["idx"], imgs_gt["offset"])
        loss_img_wh = regL1loss(imgs_wh, imgs_gt["reg_mask"], imgs_gt["idx"], imgs_gt["wh"])

        w_loss = loss_w_hm + loss_w_off
        img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1
        loss = w_loss + img_loss / N * self.args.alpha

        if self.args.use_mse:
            loss = (
                torch.nn.functional.mse_loss(world_heatmap, world_gt["heatmap"].to(world_heatmap.device))
                + self.args.alpha * torch.nn.functional.mse_loss(
                    imgs_heatmap, imgs_gt["heatmap"].to(imgs_heatmap.device)
                ) / N
            )

        return loss, world_heatmap, world_offset

    def _decode_and_collect(self, dataset, world_heatmap, world_offset, frame, res_list):
        xys = mvdet_decode(
            torch.sigmoid(world_heatmap),
            world_offset,
            reduce=dataset.world_reduce
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
                res = torch.cat([torch.ones([count, 1]) * cur_frame, pos[ids[:count]]], dim=1)
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

            loss, world_heatmap, world_offset = self._forward_and_loss_collab(
                self.train_dataset, imgs, affine_mats, keep_cams, world_gt, imgs_gt
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
            }
            self.logger.log_slot(slot_id, metrics, decision, comm_stats)

            print(
                f"[collab-train slot {slot_id}] "
                f"active_nodes={len(active_nodes)} "
                f"messages={comm_stats['num_messages']} "
                f"comm_cost={comm_stats['comm_cost']} "
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
        res_list = []
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

                loss, world_heatmap, world_offset = self._forward_and_loss_collab(
                    self.test_dataset, imgs, affine_mats, keep_cams, world_gt, imgs_gt
                )

                losses += loss.item()
                num_steps += 1

                self._decode_and_collect(self.test_dataset, world_heatmap, world_offset, frame, res_list)

                comm_stats = self.comm_manager.get_slot_stats()
                metrics = {
                    "slot_loss": float(loss.item()),
                    "updated": 0,
                }
                self.logger.log_slot(slot_id, metrics, decision, comm_stats)

                print(
                    f"[collab-infer slot {slot_id}] "
                    f"active_nodes={len(active_nodes)} "
                    f"messages={comm_stats['num_messages']} "
                    f"comm_cost={comm_stats['comm_cost']} "
                    f"loss={loss.item():.6f}"
                )

        res = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
        np.savetxt(os.path.join(self.logdir, "online_test.txt"), res, "%d")

        moda, modp, precision, recall = evaluate(
            os.path.join(self.logdir, "online_test.txt"),
            f"{self.test_dataset.gt_fname}.txt",
            self.test_dataset.base.__name__,
            self.test_dataset.frames,
        )
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

        self.logger.save_csv("online_infer_log.csv")
        summary = self.logger.summarize()

        avg_loss = losses / max(1, num_steps)

        print("========== COLLAB INFER SUMMARY ==========")
        print(f"num_slots: {summary.get('num_slots', 0)}")
        print(f"avg_slot_loss: {avg_loss:.6f}")
        print(f"avg_comm_cost: {summary.get('avg_comm_cost', 0):.4f}")
        print(f"avg_num_messages: {summary.get('avg_num_messages', 0):.4f}")
        print(
            f"collab metrics -> "
            f"moda: {moda:.1f}%, "
            f"modp: {modp:.1f}%, "
            f"prec: {precision:.1f}%, "
            f"recall: {recall:.1f}%, "
            f"f1: {f1:.1f}%"
        )
        print("=========================================")

        return {
            "moda": moda,
            "modp": modp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_slot_loss": avg_loss,
            **summary,
        }

    def run_train_then_infer(self, train_slots=None, infer_slots=None):
        print("========== PHASE 1: COLLAB TRAIN ==========")
        train_summary = self.run_train(max_slots=train_slots)

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
            return self.run_infer(max_slots=self.args.max_slots)
        elif self.args.online_mode == "train_then_infer":
            return self.run_train_then_infer(
                train_slots=self.args.online_train_slots,
                infer_slots=self.args.online_infer_slots,
            )
        else:
            raise ValueError(f"Unknown online_mode: {self.args.online_mode}")