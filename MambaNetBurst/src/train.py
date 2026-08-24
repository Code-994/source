"""
MambaNetBurst 端到端监督训练脚本。


典型用法：
  # 单卡
  python train.py \\
      --data_path /data/ISCXVPN2016/dataset_sampled \\
      --nb_classes 7 \\
      --model mambanetburst_classifier \\
      --output_dir ./output/iscxvpn2016

  # 多卡 DDP（4 GPU）
  torchrun --nproc_per_node=4 train.py \\
      --data_path /data/ISCXVPN2016/dataset_sampled \\
      --nb_classes 7 --batch_size 128
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)          # MambaNetBurst/
_DATASET_DIR = os.path.join(_ROOT_DIR, "dataset")

for _p in [_SRC_DIR, _ROOT_DIR, _DATASET_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 项目模块 ─────────────────────────────────────────────────────────────── #
import models_net_mamba2 as model_zoo
from engine import (
    MetricLogger,
    adjust_lr,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    speed_benchmark,
    train_one_epoch,
)
from dataset_burst import build_dataloader, get_dataset


# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #

def get_args_parser():
    p = argparse.ArgumentParser(
        "MambaNetBurst supervised training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 数据 ──────────────────────────────────────────────────────────────── #
    p.add_argument("--data_path", required=True,
                   help="根目录，包含 train/ val/ test/ 子目录")
    p.add_argument("--fmt", default="npy", choices=["npy", "png"],
                   help="数据格式：npy（默认）或 png（NetMamba 旧格式）")
    p.add_argument("--nb_classes", default=7, type=int,
                   help="分类类别数")
    p.add_argument("--byte_length", default=1600, type=int,
                   help="每个样本的字节序列长度")

    # ── 模型 ──────────────────────────────────────────────────────────────── #
    p.add_argument("--model", default="mambanetburst_classifier",
                   choices=list(vars(model_zoo).keys()),
                   help="模型工厂函数名")
    p.add_argument("--drop_rate", default=0.1, type=float,
                   help="Dropout rate（位置编码后）")

    # ── 训练超参数 ─────────────────────────────────────────────────────────── #
    p.add_argument("--epochs", default=120, type=int)
    p.add_argument("--batch_size", default=128, type=int,
                   help="每 GPU 的 batch size")
    p.add_argument("--accum_iter", default=1, type=int,
                   help="梯度累积步数（等效增大 batch）")

    # ── 优化器 ────────────────────────────────────────────────────────────── #
    p.add_argument("--lr", default=None, type=float,
                   help="绝对学习率（设置后忽略 --blr）")
    p.add_argument("--blr", default=1e-3, type=float,
                   help="基础学习率，实际 lr = blr × total_batch / 256")
    p.add_argument("--min_lr", default=1e-6, type=float,
                   help="cosine 衰减下界")
    p.add_argument("--warmup_epochs", default=10, type=int,
                   help="线性 warmup epoch 数")
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--clip_grad", default=1.0, type=float,
                   help="梯度裁剪范数（0 = 不裁剪）")

    # ── 损失函数 ──────────────────────────────────────────────────────────── #
    p.add_argument("--smoothing", default=0.1, type=float,
                   help="Label smoothing（0 = 标准 CE）")
    p.add_argument("--class_balance", action="store_true",
                   help="类平衡损失权重（按有效样本数计算）")
    p.add_argument("--class_balance_beta", default=0.999, type=float)

    # ── 检查点 ────────────────────────────────────────────────────────────── #
    p.add_argument("--output_dir", default="./output/train")
    p.add_argument("--resume", default="",
                   help="从检查点恢复训练")
    p.add_argument("--start_epoch", default=0, type=int)
    p.add_argument("--save_freq", default=10, type=int,
                   help="每隔多少 epoch 保存一次（除 best 之外）")

    # ── 运行模式 ──────────────────────────────────────────────────────────── #
    p.add_argument("--eval_only", action="store_true",
                   help="仅评测（需要 --resume 指定检查点）")
    p.add_argument("--speed_test", action="store_true",
                   help="训练结束后额外跑吞吐量测试")

    # ── 系统 ──────────────────────────────────────────────────────────────── #
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--pin_mem", action="store_true", default=True)
    p.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")

    # ── AMP ───────────────────────────────────────────────────────────────── #
    p.add_argument("--amp", action="store_true", default=True,
                   help="启用混合精度训练（AMP）")
    p.add_argument("--no_amp", action="store_false", dest="amp")

    # ── 分布式 ────────────────────────────────────────────────────────────── #
    p.add_argument("--dist_url", default="env://")
    p.add_argument("--dist_backend", default="nccl")

    return p


# --------------------------------------------------------------------------- #
# 分布式初始化
# --------------------------------------------------------------------------- #

def init_distributed(args) -> bool:
    """初始化 DDP，返回是否为分布式模式。"""
    if "RANK" not in os.environ:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.distributed = False
        return False

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    args.distributed = True
    return True


def is_main_process(args) -> bool:
    return getattr(args, "rank", 0) == 0


# --------------------------------------------------------------------------- #
# 类平衡损失权重计算
# --------------------------------------------------------------------------- #

def compute_class_weights(dataset, nb_classes: int, beta: float, device) -> torch.Tensor:
    """
    Class-balanced re-weighting (Cui et al., CVPR 2019):
        weight_c = (1 - beta) / (1 - beta^{n_c})，再归一化。
    """
    counts = torch.zeros(nb_classes)
    for _, label in dataset:
        counts[label] += 1
    eff = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / eff
    weights = weights / weights.sum() * nb_classes
    return weights.to(device)


# --------------------------------------------------------------------------- #
# 主函数
# --------------------------------------------------------------------------- #

def main(args):
    # ── 分布式初始化 ───────────────────────────────────────────────────────── #
    init_distributed(args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed + getattr(args, "rank", 0))
    np.random.seed(args.seed + getattr(args, "rank", 0))
    cudnn.benchmark = True

    if is_main_process(args):
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 60)
        print("MambaNetBurst training")
        print(json.dumps(vars(args), indent=2))
        print("=" * 60)

    # ── 数据集 ────────────────────────────────────────────────────────────── #
    dataset_train = get_dataset(args.data_path, split="train",
                                fmt=args.fmt, byte_length=args.byte_length)
    dataset_val   = get_dataset(args.data_path, split="val",
                                fmt=args.fmt, byte_length=args.byte_length)
    dataset_test  = get_dataset(args.data_path, split="test",
                                fmt=args.fmt, byte_length=args.byte_length)

    print(f"[data] train={len(dataset_train)}  val={len(dataset_val)}  "
          f"test={len(dataset_test)}  classes={dataset_train.classes}")

    # ── Sampler & DataLoader ──────────────────────────────────────────────── #
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
        sampler_val  = torch.utils.data.DistributedSampler(dataset_val,  shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test  = torch.utils.data.SequentialSampler(dataset_test)

    _loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, drop_last=True, **_loader_kwargs
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, drop_last=False, **_loader_kwargs
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size, drop_last=False, **_loader_kwargs
    )

    # ── 学习率计算 ────────────────────────────────────────────────────────── #
    eff_batch = args.batch_size * args.accum_iter * getattr(args, "world_size", 1)
    if args.lr is None:
        args.lr = args.blr * eff_batch / 256
    print(f"[lr] base_lr={args.blr:.2e}  actual_lr={args.lr:.2e}  "
          f"eff_batch={eff_batch}")

    # ── 模型 ──────────────────────────────────────────────────────────────── #
    factory = getattr(model_zoo, args.model)
    model: torch.nn.Module = factory(
        num_classes=args.nb_classes,
        byte_length=args.byte_length,
        drop_rate=args.drop_rate,
    )
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.model}  params={n_params/1e6:.2f}M")

    model_no_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_no_ddp = model.module

    # ── 优化器 ────────────────────────────────────────────────────────────── #
    # pos_embed / cls_token 不加 weight decay（各向同性参数）
    no_wd_keys = model_no_ddp.no_weight_decay()
    param_groups = [
        {
            "params": [
                p for n, p in model_no_ddp.named_parameters()
                if n not in no_wd_keys and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model_no_ddp.named_parameters()
                if n in no_wd_keys and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    # ── AMP ───────────────────────────────────────────────────────────────── #
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = torch.cuda.amp.GradScaler()
    else:
        from contextlib import suppress
        amp_autocast = suppress
        loss_scaler = "none"

    # ── 损失函数 ──────────────────────────────────────────────────────────── #
    if args.class_balance:
        weights = compute_class_weights(
            dataset_train, args.nb_classes, args.class_balance_beta, device
        )
        criterion = torch.nn.CrossEntropyLoss(
            weight=weights, label_smoothing=args.smoothing
        )
        print(f"[loss] class-balanced CE  weights={weights.tolist()}")
    elif args.smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        print(f"[loss] LabelSmoothingCE  smoothing={args.smoothing}")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("[loss] CrossEntropyLoss")

    # ── 恢复检查点 ────────────────────────────────────────────────────────── #
    if args.resume:
        args.start_epoch = load_checkpoint(
            args.resume, model, optimizer, loss_scaler
        )

    # ── 仅评测模式 ────────────────────────────────────────────────────────── #
    if args.eval_only:
        print("\n[eval-only] running on val split...")
        stats = evaluate(loader_val, model, device)
        print(json.dumps(
            {k: v for k, v in stats.items() if k not in ("cm", "pre_per_class", "rec_per_class", "f1_per_class", "support_per_class")},
            indent=2
        ))
        return

    # ── TensorBoard ───────────────────────────────────────────────────────── #
    log_writer = None
    if is_main_process(args):
        log_dir = os.path.join(args.output_dir, "tensorboard")
        log_writer = SummaryWriter(log_dir=log_dir)

    # ── 训练循环 ──────────────────────────────────────────────────────────── #
    print(f"\n[train] start  epochs={args.epochs}  best_ckpt → {args.output_dir}/checkpoint-best.pth")
    t_start = time.time()
    best_acc = 0.0
    best_f1  = 0.0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            amp_autocast=amp_autocast,
            max_norm=args.clip_grad,
            log_writer=log_writer,
            args=args,
        )

        val_stats = evaluate(loader_val, model, device)

        # ── best checkpoint ────────────────────────────────────────────────── #
        if is_main_process(args) and val_stats["acc"] > best_acc:
            best_acc = val_stats["acc"]
            best_f1  = val_stats["macro_f1"]
            best_epoch = epoch
            save_checkpoint(
                args.output_dir, "best", model, optimizer, epoch, loss_scaler, args
            )

        # ── 定期保存 ────────────────────────────────────────────────────────── #
        if is_main_process(args) and (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                args.output_dir, str(epoch), model, optimizer, epoch, loss_scaler, args
            )

        # ── 日志 ──────────────────────────────────────────────────────────── #
        print(
            f"  epoch {epoch:3d}  "
            f"train_loss={train_stats['loss']:.4f}  "
            f"val_acc={val_stats['acc']:.4f}  "
            f"val_f1={val_stats['macro_f1']:.4f}  "
            f"best_acc={best_acc:.4f} (ep {best_epoch})"
        )

        if is_main_process(args):
            log_row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{
                    f"val_{k}": v for k, v in val_stats.items()
                    if k not in ("cm", "pre_per_class", "rec_per_class", "f1_per_class", "support_per_class")
                },
            }
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_row) + "\n")

            if log_writer is not None:
                log_writer.add_scalar("val/acc", val_stats["acc"], epoch)
                log_writer.add_scalar("val/f1",  val_stats["macro_f1"], epoch)

    # ── 训练结束，用 best checkpoint 跑最终测试 ────────────────────────────── #
    total_time = time.time() - t_start
    print(f"\n[done] total time: {str(datetime.timedelta(seconds=int(total_time)))}")
    print(f"[done] best val acc={best_acc:.4f}  f1={best_f1:.4f}  epoch={best_epoch}")

    best_ckpt_path = os.path.join(args.output_dir, "checkpoint-best.pth")
    if os.path.exists(best_ckpt_path) and is_main_process(args):
        load_checkpoint(best_ckpt_path, model)
        test_stats = evaluate(loader_test, model, device)
        print(
            f"[test] acc={test_stats['acc']:.4f}  "
            f"f1={test_stats['macro_f1']:.4f}  "
            f"pre={test_stats['macro_pre']:.4f}  "
            f"rec={test_stats['macro_rec']:.4f}"
        )
        with open(os.path.join(args.output_dir, "test_stats.json"), "w") as f:
            json.dump(
                {
                    k: v for k, v in test_stats.items()
                    if k not in ("pre_per_class", "rec_per_class", "f1_per_class", "support_per_class")
                },
                f, indent=2,
            )

        # 保存详细的每类指标
        with open(os.path.join(args.output_dir, "test_per_class.json"), "w") as f:
            json.dump(
                {
                    "classes": dataset_test.classes,
                    "pre_per_class":     test_stats["pre_per_class"],
                    "rec_per_class":     test_stats["rec_per_class"],
                    "f1_per_class":      test_stats["f1_per_class"],
                    "support_per_class": test_stats["support_per_class"],
                    "cm": test_stats["cm"],
                },
                f, indent=2,
            )

    # 保存训练汇总
    if is_main_process(args):
        with open(os.path.join(args.output_dir, "train_summary.json"), "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "dataset": args.data_path,
                    "nb_classes": args.nb_classes,
                    "epochs": args.epochs,
                    "best_val_acc": best_acc,
                    "best_val_f1":  best_f1,
                    "best_epoch":   best_epoch,
                    "total_time_s": int(total_time),
                },
                f, indent=2,
            )

    # ── 吞吐量测试（可选）─────────────────────────────────────────────────── #
    if args.speed_test and is_main_process(args):
        print("\n[speed] running throughput benchmark...")
        speed_benchmark(dataset_train, model, device, args.output_dir,
                        num_workers=args.num_workers)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
