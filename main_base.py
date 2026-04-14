import os
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import sys
import shutil
import random
import datetime
import tqdm
import numpy as np
import torch

from torch import optim
from torch.utils.data import DataLoader
from distutils.dir_util import copy_tree

from src.datasets import *
from src.models.mvdet import MVDet
from src.models.mvcnn import MVCNN
from src.utils.logger import Logger
from src.utils.draw_curve import draw_curve
from src.utils.str2bool import str2bool
from src.trainer import PerspectiveTrainer
from src.system.online_runner import OnlineRunner


def build_datasets(args):
    if 'modelnet' in args.dataset:
        if args.dataset == 'modelnet40_12':
            fpath = os.path.expanduser('~/Data/modelnet/modelnet40_images_new_12x')
            num_cam = 12
        elif args.dataset == 'modelnet40_20':
            fpath = os.path.expanduser('~/Data/modelnet/modelnet40v2png_ori4')
            num_cam = 20
        else:
            raise ValueError(f'Unknown dataset: {args.dataset}')

        args.task = 'mvcnn'
        args.lr = 5e-5 if args.lr is None else args.lr
        args.batch_size = 8 if args.batch_size is None else args.batch_size

        train_set = ModelNet40(fpath, num_cam, split='train')
        val_set = ModelNet40(fpath, num_cam, split='train', per_cls_instances=25)
        test_set = ModelNet40(fpath, num_cam, split='test')

    elif args.dataset == 'scanobjectnn':
        fpath = os.path.expanduser('~/Data/ScanObjectNN')

        args.task = 'mvcnn'
        args.lr = 5e-5 if args.lr is None else args.lr
        args.batch_size = 8 if args.batch_size is None else args.batch_size

        train_set = ScanObjectNN(fpath, split='train')
        val_set = ScanObjectNN(fpath, split='train', per_cls_instances=25)
        test_set = ScanObjectNN(fpath, split='test')

    else:
        if args.dataset == 'wildtrack':
            base = Wildtrack(os.path.expanduser('~/Data/Wildtrack'))
        elif args.dataset == 'multiviewx':
            base = MultiviewX(os.path.expanduser('~/Data/MultiviewX'))
        else:
            raise ValueError('dataset must be one of [wildtrack, multiviewx, modelnet40_12, modelnet40_20, scanobjectnn]')

        args.task = 'mvdet'
        args.lr = 5e-4 if args.lr is None else args.lr
        args.batch_size = 1 if args.batch_size is None else args.batch_size

        train_set = frameDataset(
            base, split='trainval',
            world_reduce=args.world_reduce,
            img_reduce=args.img_reduce,
            world_kernel_size=args.world_kernel_size,
            img_kernel_size=args.img_kernel_size,
            dropout=args.dropcam,
            augmentation=args.augmentation
        )
        val_set = frameDataset(
            base, split='val',
            world_reduce=args.world_reduce,
            img_reduce=args.img_reduce,
            world_kernel_size=args.world_kernel_size,
            img_kernel_size=args.img_kernel_size
        )
        test_set = frameDataset(
            base, split='test',
            world_reduce=args.world_reduce,
            img_reduce=args.img_reduce,
            world_kernel_size=args.world_kernel_size,
            img_kernel_size=args.img_kernel_size
        )

    return train_set, val_set, test_set


def build_loaders(args, train_set, val_set, test_set):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    return train_loader, val_loader, test_loader


def build_logdir(args, is_debug):
    if not args.eval and not args.online:
        logdir = (
            f'logs/{args.dataset}/'
            f'{"DEBUG_" if is_debug else ""}'
            f'BASELINE_{args.arch}_{args.aggregation}_'
            f'lr{args.lr}_b{args.batch_size}_e{args.epochs}_'
            f'dropcam{args.dropcam}_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
        )
    elif args.online:
        if args.online_mode == "train_then_infer":
            logdir = (
                f'logs/{args.dataset}/'
                f'{"DEBUG_" if is_debug else ""}'
                f'ONLINE_train_then_infer_{args.arch}_{args.aggregation}_'
                f'trainslots{args.online_train_slots}_inferslots{args.online_infer_slots}_'
                f'dropcam{args.dropcam}_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
            )
        else:
            logdir = (
                f'logs/{args.dataset}/'
                f'{"DEBUG_" if is_debug else ""}'
                f'ONLINE_{args.online_mode}_{args.arch}_{args.aggregation}_'
                f'maxslots{args.max_slots}_'
                f'dropcam{args.dropcam}_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
            )
    else:
        if args.resume is None and args.resume_path is None:
            raise ValueError('--eval requires --resume or --resume_path')
        logdir = f'logs/{args.dataset}/EVAL_BASELINE_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
    return logdir


def copy_scripts(logdir):
    os.makedirs(logdir, exist_ok=True)
    copy_tree('src', os.path.join(logdir, 'scripts', 'src'))
    for script in os.listdir('.'):
        if script.endswith('.py'):
            dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def build_model_and_runner(args, train_set, logdir):
    if args.task == 'mvcnn':
        model = MVCNN(train_set, args.arch, args.aggregation).cuda()
        from src.trainer_mvcnn import ClassifierTrainer
        trainer = ClassifierTrainer(model, logdir, args)
    else:
        model = MVDet(
            train_set,
            args.arch,
            args.aggregation,
            args.use_bottleneck,
            args.hidden_dim,
            args.outfeat_dim
        ).cuda()
        trainer = PerspectiveTrainer(model, logdir, args)
    return model, trainer


def load_resume_if_needed(args, model):
    """
    Priority:
    1) --resume_path
    2) --resume  -> logs/{dataset}/{resume}/model.pth
    3) default checkpoint path (if use_default_ckpt=True)
    4) random init
    """
    ckpt_path = None

    if args.resume_path is not None:
        ckpt_path = args.resume_path
    elif args.resume:
        ckpt_path = f'logs/{args.dataset}/{args.resume}/model.pth'
    elif args.use_default_ckpt:
        default_ckpt_map = {
            # 这里改成你已经验证能跑到 90.1 的那份模型
            'wildtrack': '/home/server2/online_inference/logs/wildtrack/resnet18_max_down1_lr0.0005_b1_e10_dropcam0.0_2026-04-06_21-38-52/model.pth',
            'multiviewx': '/home/server2/online_inference/logs/multiviewx/resnet18_max_down1_lr0.0005_b1_e10_dropcam0.0_2026-04-08_13-58-58/model.pth',
        }
        ckpt_path = default_ckpt_map.get(args.dataset, None)

    if ckpt_path is None:
        print('No checkpoint specified. Using random initialization.')
        return

    if not os.path.isfile(ckpt_path):
        print(f'Checkpoint not found: {ckpt_path}')
        print('Using random initialization instead.')
        return

    print(f'loading checkpoint: {ckpt_path}')
    pretrained_dict = torch.load(ckpt_path, map_location='cpu')
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('Checkpoint loaded successfully.')


def build_optimizer_scheduler(args, model):
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if 'base' not in n and p.requires_grad],
            "lr": args.lr * args.other_lr_ratio,
        },
        {
            "params": [p for n, p in model.named_parameters() if 'base' in n and p.requires_grad],
            "lr": args.lr * args.base_lr_ratio,
        },
    ]
    optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    def warmup_lr_scheduler(epoch, warmup_epochs=0.1 * args.epochs):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return (np.cos((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * np.pi) + 1) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)
    return optimizer, scheduler


def run_baseline(args, trainer, model, logdir, train_loader, test_loader):
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []

    best_primary = -1.0
    best_epoch = -1
    best_metrics = None

    if not args.eval:
        optimizer, scheduler = build_optimizer_scheduler(args, model)

        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, scheduler)

            if epoch % max(args.epochs // 10, 1) == 0:
                print('Testing...')
                test_loss, test_prec = trainer.test(test_loader)

                x_epoch.append(epoch)
                train_loss_s.append(train_loss)
                train_prec_s.append(train_prec)
                test_loss_s.append(test_loss)

                if isinstance(test_prec, (list, tuple)):
                    test_prec_s.append(test_prec[0])
                    current_primary = float(test_prec[0])
                else:
                    test_prec_s.append(test_prec)
                    current_primary = float(test_prec)

                if current_primary > best_primary:
                    best_primary = current_primary
                    best_epoch = epoch
                    best_metrics = list(test_prec) if isinstance(test_prec, (list, tuple)) else [float(test_prec)]
                    torch.save(model.state_dict(), os.path.join(logdir, 'model_best.pth'))

                draw_curve(
                    os.path.join(logdir, 'learning_curve.jpg'),
                    x_epoch,
                    train_loss_s,
                    test_loss_s,
                    train_prec_s,
                    test_prec_s
                )
                torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))

        print('========== FINAL SUMMARY ==========')
        print(f'Last epoch: {args.epochs}')
        print(f'Best epoch: {best_epoch}')
        if args.task == 'mvdet' and best_metrics is not None and len(best_metrics) >= 5:
            print(
                f'Best metrics -> '
                f'moda: {best_metrics[0]:.1f}%, '
                f'modp: {best_metrics[1]:.1f}%, '
                f'prec: {best_metrics[2]:.1f}%, '
                f'recall: {best_metrics[3]:.1f}%, '
                f'f1: {best_metrics[4]:.1f}%'
            )
        elif args.task == 'mvcnn' and best_metrics is not None:
            print(f'Best acc -> {best_metrics[0]:.2f}%')
        print('===================================')
    else:
        print('Testing loaded model...')
        trainer.test(test_loader)


def run_online(args, model, train_set, test_set, logdir):
    if args.task != 'mvdet':
        raise ValueError('online mode currently only supports detection datasets [wildtrack, multiviewx]')

    optimizer = None
    if args.online_mode in ["train", "train_then_infer"]:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.online_lr,
            weight_decay=args.weight_decay,
        )

    runner = OnlineRunner(
        model=model,
        train_dataset=train_set,
        test_dataset=test_set,
        logdir=logdir,
        args=args,
        optimizer=optimizer,
    )
    runner.run()


def main(args):
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
        torch.autograd.set_detect_anomaly(True)
    else:
        print('No sys.gettrace')
        is_debug = False

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_set, val_set, test_set = build_datasets(args)
    train_loader, val_loader, test_loader = build_loaders(args, train_set, val_set, test_set)

    logdir = build_logdir(args, is_debug)
    copy_scripts(logdir)

    sys.stdout = Logger(os.path.join(logdir, 'log.txt'))
    print(logdir)
    print('Settings:')
    print(vars(args))

    model, trainer = build_model_and_runner(args, train_set, logdir)
    load_resume_if_needed(args, model)

    if args.online:
        run_online(args, model, train_set, test_set, logdir)
    else:
        run_baseline(args, trainer, model, logdir, train_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='baseline + collaborative online inference')

    parser.add_argument('--eval', action='store_true', help='evaluation only')
    parser.add_argument('--online', action='store_true', help='run online mode')
    parser.add_argument('--online_mode', type=str, default='infer',
                        choices=['train', 'infer', 'train_then_infer'],
                        help='online mode: train / infer / train_then_infer')
    parser.add_argument('--max_slots', type=int, default=20, help='maximum online slots to run')
    parser.add_argument('--online_train_slots', type=int, default=100,
                        help='number of slots used in online training phase')
    parser.add_argument('--online_infer_slots', type=int, default=40,
                        help='number of slots used in online inference phase')
    parser.add_argument('--online_lr', type=float, default=1e-5,
                        help='learning rate for online training')
    parser.add_argument('--online_save_name', type=str, default='online_model.pth',
                        help='filename to save online-trained model')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='direct path to checkpoint file, higher priority than --resume')
    parser.add_argument('--use_default_ckpt', type=str2bool, default=True,
                        help='auto-load default checkpoint if no resume is given')

    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--aggregation', type=str, default='max', choices=['mean', 'max'])
    parser.add_argument(
        '-d', '--dataset', type=str, default='wildtrack',
        choices=['wildtrack', 'multiviewx', 'modelnet40_12', 'modelnet40_20', 'scanobjectnn']
    )
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=None, help='input batch size for training')
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--base_lr_ratio', type=float, default=1.0)
    parser.add_argument('--other_lr_ratio', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--down', type=int, default=1, help='down sample the image to 1/N size')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', type=str2bool, default=False)

    parser.add_argument('--eval_init_cam', type=str2bool, default=False)
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--augmentation', type=str2bool, default=True)
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--cls_thres', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per-view loss')
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)

    args = parser.parse_args()
    main(args)