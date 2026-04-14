import time
import torch
import torch.nn.functional as F

from src.models.aggregation import aggregate_feat


class ClassifierTrainer(object):
    def __init__(self, model, logdir, args):
        self.model = model
        self.args = args
        self.logdir = logdir

    def train(self, epoch, dataloader, optimizer, scheduler=None, log_interval=200):
        self.model.train()
        if self.args.base_lr_ratio == 0:
            self.model.base.eval()

        losses, correct, miss = 0.0, 0.0, 1e-8
        t0 = time.time()

        for batch_idx, (imgs, tgt, keep_cams) in enumerate(dataloader):
            imgs, tgt = imgs.cuda(), tgt.cuda()

            feat, _ = self.model.get_feat(imgs, None, self.args.down)
            overall_feat = aggregate_feat(feat, keep_cams, self.model.aggregation)
            output = self.model.get_output(overall_feat)

            loss = F.cross_entropy(output, tgt)

            pred = torch.argmax(output, 1)
            correct += (pred == tgt).sum().item()
            miss += imgs.shape[0] - (pred == tgt).sum().item()

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
                    f'loss: {losses / (batch_idx + 1):.3f}, '
                    f'acc: {correct / (correct + miss) * 100:.2f}%, '
                    f'time: {t1 - t0:.1f}'
                )

        return losses / len(dataloader), correct / (correct + miss) * 100.0

    def test(self, dataloader):
        t0 = time.time()
        self.model.eval()

        losses, correct, miss = 0.0, 0.0, 1e-8

        with torch.no_grad():
            for batch_idx, (imgs, tgt, keep_cams) in enumerate(dataloader):
                imgs, tgt = imgs.cuda(), tgt.cuda()

                output, _ = self.model(imgs, None, self.args.down, keep_cams=keep_cams)
                loss = F.cross_entropy(output, tgt)

                losses += loss.item()

                pred = torch.argmax(output, 1)
                correct += (pred == tgt).sum().item()
                miss += imgs.shape[0] - (pred == tgt).sum().item()

        acc = correct / (correct + miss) * 100.0
        print(
            f'Test, loss: {losses / len(dataloader):.3f}, '
            f'acc: {acc:.2f}%, time: {time.time() - t0:.1f}s'
        )

        return losses / len(dataloader), [acc]