import time
import numpy as np
import torch
import torch.nn.functional as F

from src.loss import focal_loss, regL1loss
from src.evaluation.evaluate import evaluate
from src.utils.decode import mvdet_decode
from src.utils.nms import nms
from src.models.aggregation import aggregate_feat


class PerspectiveTrainer(object):
    def __init__(self, model, logdir, args):
        self.model = model
        self.args = args
        self.logdir = logdir

    def train(self, epoch, dataloader, optimizer, scheduler=None, log_interval=100):
        self.model.train()
        if self.args.base_lr_ratio == 0:
            self.model.base.eval()

        losses = 0.0
        t0 = time.time()

        for batch_idx, (imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams) in enumerate(dataloader):
            B, N = imgs.shape[:2]

            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].flatten(0, 1)

            imgs = imgs.cuda()
            affine_mats = affine_mats.cuda()

            feat, (imgs_heatmap, imgs_offset, imgs_wh) = self.model.get_feat(
                imgs, affine_mats, self.args.down
            )
            overall_feat = aggregate_feat(feat, keep_cams, self.model.aggregation)
            world_heatmap, world_offset = self.model.get_output(overall_feat)

            loss_w_hm = focal_loss(world_heatmap, world_gt['heatmap'])
            loss_w_off = regL1loss(world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset'])

            loss_img_hm = focal_loss(
                imgs_heatmap,
                imgs_gt['heatmap'],
                keep_cams.view(B * N, 1, 1, 1)
            )
            loss_img_off = regL1loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
            loss_img_wh = regL1loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])

            w_loss = loss_w_hm + loss_w_off
            img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1
            loss = w_loss + img_loss / N * self.args.alpha

            if self.args.use_mse:
                loss = F.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device)) + \
                       self.args.alpha * F.mse_loss(imgs_heatmap, imgs_gt['heatmap'].to(imgs_heatmap.device)) / N

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                                            torch.optim.lr_scheduler.LambdaLR)):
                    scheduler.step(epoch - 1 + batch_idx / len(dataloader))

            if (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == len(dataloader):
                t1 = time.time()
                print(
                    f'Train epoch: {epoch}, batch:{batch_idx + 1}, '
                    f'loss: {losses / (batch_idx + 1):.3f}, time: {t1 - t0:.1f}'
                )

        return losses / len(dataloader), None

    def test(self, dataloader):
        t0 = time.time()
        self.model.eval()

        losses = 0.0
        res_list = []

        with torch.no_grad():
            for batch_idx, (imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams) in enumerate(dataloader):
                B, N = imgs.shape[:2]

                imgs = imgs.cuda()
                affine_mats = affine_mats.cuda()

                (world_heatmap, world_offset), _ = self.model(
                    imgs, affine_mats, self.args.down, keep_cams=keep_cams
                )

                loss = focal_loss(world_heatmap, world_gt['heatmap'])
                if self.args.use_mse:
                    loss = F.mse_loss(world_heatmap, world_gt['heatmap'].cuda())

                losses += loss.item()

                xys = mvdet_decode(
                    torch.sigmoid(world_heatmap),
                    world_offset,
                    reduce=dataloader.dataset.world_reduce
                ).cpu()

                grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
                if dataloader.dataset.base.indexing == 'xy':
                    positions = grid_xy
                else:
                    positions = grid_xy[:, :, [1, 0]]

                for b in range(B):
                    ids = scores[b].squeeze() > self.args.cls_thres
                    pos, s = positions[b, ids], scores[b, ids, 0]
                    ids, count = nms(pos, s, 20, np.inf)
                    res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
                    res_list.append(res)

        res = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
        np.savetxt(f'{self.logdir}/test.txt', res, '%d')

        # evaluate returns: moda, modp, precision, recall
        moda, modp, precision, recall = evaluate(
            f'{self.logdir}/test.txt',
            f'{dataloader.dataset.gt_fname}.txt',
            dataloader.dataset.base.__name__,
            dataloader.dataset.frames
        )
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

        print(
            f'Test, loss: {losses / len(dataloader):.6f}, '
            f'moda: {moda:.1f}%, modp: {modp:.1f}%, '
            f'prec: {precision:.1f}%, recall: {recall:.1f}%, '
            f'f1: {f1:.1f}%, time: {time.time() - t0:.1f}s'
        )

        return losses / len(dataloader), [moda, modp, precision, recall, f1]