"""
MambaNetBurst 训练/评测引擎。

Exports:
    train_one_epoch  — 单 epoch 监督训练循环
    evaluate         — 完整评测（acc / precision / recall / F1 / 混淆矩阵）
    adjust_lr        — cosine LR with linear warmup
    save_checkpoint  — 保存检查点
    load_checkpoint  — 加载检查点
    MetricLogger     — 迭代级日志
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from contextlib import suppress
from typing import Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from timm.utils import accuracy


# --------------------------------------------------------------------------- #
# 日志工具（自包含，不依赖 NetMamba util/）
# --------------------------------------------------------------------------- #

class SmoothedValue:
    """滑动窗口平均值，用于平滑打印训练指标。"""

    def __init__(self, window_size: int = 20, fmt: str = "{avg:.4f}"):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1):
        self.deque.append(value)
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return sum(self.deque) / max(len(self.deque), 1)

    @property
    def global_avg(self) -> float:
        return self.total / max(self.count, 1)

    def __str__(self) -> str:
        return self.fmt.format(
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.deque[-1] if self.deque else 0,
        )


class MetricLogger:
    """多指标聚合日志，每隔 print_freq 步打印一次。"""

    def __init__(self, delimiter: str = "  "):
        self.meters: dict[str, SmoothedValue] = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)

    def __getattr__(self, name: str):
        if name in self.meters:
            return self.meters[name]
        raise AttributeError(f"MetricLogger has no attribute '{name}'")

    def __str__(self) -> str:
        parts = [f"{k}: {v}" for k, v in self.meters.items()]
        return self.delimiter.join(parts)

    def log_every(self, iterable, print_freq: int, header: str = ""):
        i = 0
        start = time.time()
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0:
                elapsed = time.time() - start
                print(
                    f"{header}  [{i}/{len(iterable)}]  "
                    f"{self}  time/it: {elapsed / i:.3f}s"
                )

    def synchronize_between_processes(self):
        """多进程下汇总各卡的 count / total。"""
        if not (dist.is_available() and dist.is_initialized()):
            return
        for meter in self.meters.values():
            t = torch.tensor(
                [meter.count, meter.total], dtype=torch.float64, device="cuda"
            )
            dist.barrier()
            dist.all_reduce(t)
            meter.count = int(t[0].item())
            meter.total = t[1].item()


# --------------------------------------------------------------------------- #
# 学习率调度：cosine decay with linear warmup
# --------------------------------------------------------------------------- #

def adjust_lr(optimizer: torch.optim.Optimizer, epoch: float, args) -> float:
    """
    每 iteration 调用，返回当前 lr。

    策略：
        epoch < warmup_epochs → 线性从 min_lr 升到 lr
        epoch >= warmup_epochs → cosine decay 从 lr 降到 min_lr
    """
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
    lr = max(lr, args.min_lr)
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


# --------------------------------------------------------------------------- #
# 检查点 I/O
# --------------------------------------------------------------------------- #

def save_checkpoint(
    output_dir: str,
    name: str,
    model,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss_scaler=None,
    args=None,
) -> None:
    """保存训练状态到 <output_dir>/checkpoint-<name>.pth。"""
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "model": (model.module if hasattr(model, "module") else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args) if args is not None else {},
    }
    if loss_scaler is not None and loss_scaler != "none":
        state["scaler"] = loss_scaler.state_dict()
    path = os.path.join(output_dir, f"checkpoint-{name}.pth")
    torch.save(state, path)
    print(f"[checkpoint] saved → {path}")


def load_checkpoint(
    path: str,
    model,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_scaler=None,
) -> int:
    """
    从 path 加载检查点，返回起始 epoch。
    strict=False 允许分类头形状不匹配（更换数据集时很常见）。
    """
    ckpt = torch.load(path, map_location="cpu")
    model_without_ddp = model.module if hasattr(model, "module") else model

    # 分类头尺寸不匹配时自动跳过
    state_dict = ckpt["model"]
    own_state = model_without_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in own_state and state_dict[k].shape != own_state[k].shape:
            print(f"[load_checkpoint] skip mismatched key: {k}")
            del state_dict[k]

    msg = model_without_ddp.load_state_dict(state_dict, strict=False)
    print(f"[load_checkpoint] {path}\n  {msg}")

    start_epoch = ckpt.get("epoch", 0) + 1
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if loss_scaler is not None and loss_scaler != "none" and "scaler" in ckpt:
        loss_scaler.load_state_dict(ckpt["scaler"])
    return start_epoch


# --------------------------------------------------------------------------- #
# 训练循环
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    amp_autocast,
    max_norm: float = 0.0,
    log_writer=None,
    args=None,
) -> dict:
    """
    MambaNetBurst 单 epoch 监督训练。

    输入 batch: (bytes_tensor, targets)
        bytes_tensor : (B, 1600) LongTensor — 原始字节整数
        targets      : (B,) LongTensor — 类别标签

    与 NetMamba 的差异：
        • 无 Mixup（论文未使用；需要时可外部传 mixup_fn 扩展）
        • 无 mask_ratio（无 MAE 阶段）
        • 每 iteration 调整 LR（cosine 调度）
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Train Epoch [{epoch}]"
    accum_iter = getattr(args, "accum_iter", 1)

    optimizer.zero_grad()

    for step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq=20, header=header)
    ):
        # ── 每 iteration 更新 LR ─────────────────────────────────────────── #
        if step % accum_iter == 0:
            cur_epoch = step / len(data_loader) + epoch
            lr = adjust_lr(optimizer, cur_epoch, args)
            metric_logger.update(lr=lr)

        samples = samples.to(device, non_blocking=True)   # (B, 1600) LongTensor
        targets = targets.to(device, non_blocking=True)

        # ── 前向 + 损失 ──────────────────────────────────────────────────── #
        with amp_autocast():
            outputs = model(samples)           # (B, num_classes)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training.")
            raise RuntimeError("NaN/Inf loss detected.")

        # ── 反向传播 ─────────────────────────────────────────────────────── #
        if loss_scaler != "none":
            loss /= accum_iter
            loss_scaler.scale(loss).backward()
            if (step + 1) % accum_iter == 0:
                if max_norm > 0:
                    loss_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                loss_scaler.step(optimizer)
                loss_scaler.update()
                optimizer.zero_grad()
        else:
            loss /= accum_iter
            loss.backward()
            if (step + 1) % accum_iter == 0:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        # ── TensorBoard ──────────────────────────────────────────────────── #
        if log_writer is not None and (step + 1) % accum_iter == 0:
            global_step = int((step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train/loss", loss_value, global_step)
            log_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

    metric_logger.synchronize_between_processes()
    print(f"  [train] {metric_logger}")
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(
    data_loader: Iterable,
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    """
    完整评测：准确率、加权 P/R/F1、每类指标、混淆矩阵。

    Returns:
        dict，键包括:
            loss, acc1, acc5, acc,
            weighted_pre, weighted_rec, weighted_f1,
            pre_per_class, rec_per_class, f1_per_class, support_per_class,
            cm
    """
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")

    model.eval()
    pred_all: list[int] = []
    target_all: list[int] = []

    for samples, targets in metric_logger.log_every(data_loader, print_freq=10, header="Eval:"):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        _, pred = outputs.topk(1, dim=1)
        pred_all.extend(pred.squeeze(1).cpu().tolist())
        target_all.extend(targets.cpu().tolist())

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters.setdefault("acc1", SmoothedValue()).update(acc1.item() / 100, n=batch_size)
        metric_logger.meters.setdefault("acc5", SmoothedValue()).update(acc5.item() / 100, n=batch_size)

    metric_logger.synchronize_between_processes()

    # sklearn 全局指标
    acc = accuracy_score(target_all, pred_all)
    macro = precision_recall_fscore_support(target_all, pred_all, average="macro", zero_division=0)
    per_cls = precision_recall_fscore_support(target_all, pred_all, average=None, zero_division=0)
    cm = confusion_matrix(target_all, pred_all)

    print(
        f"  [eval] Acc@1 {metric_logger.meters['acc1'].global_avg:.4f}  "
        f"Loss {metric_logger.meters['loss'].global_avg:.4f}  "
        f"Pre {macro[0]:.4f}  Rec {macro[1]:.4f}  F1 {macro[2]:.4f}"
    )

    result = {k: m.global_avg for k, m in metric_logger.meters.items()}
    result.update(
        acc=acc,
        macro_pre=float(macro[0]),
        macro_rec=float(macro[1]),
        macro_f1=float(macro[2]),
        pre_per_class=per_cls[0].tolist(),
        rec_per_class=per_cls[1].tolist(),
        f1_per_class=per_cls[2].tolist(),
        support_per_class=per_cls[3].tolist(),
        cm=cm.tolist(),
    )
    return result


# --------------------------------------------------------------------------- #
# 吞吐量基准（可选）
# --------------------------------------------------------------------------- #

@torch.no_grad()
def speed_benchmark(
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: str,
    num_workers: int = 4,
    max_seconds: int = 30,
) -> None:
    """
    测试不同 batch_size 下的推理吞吐量和显存占用，结果写 speed_test.json。
    """
    import gc

    model.eval()
    model_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    results = []

    # 从 batch=8 到 batch=1024（2^3 到 2^10），首轮 batch=1024 作 warm-up
    batch_sizes = [2 ** i for i in range(3, 11)]
    batch_sizes[0] = 1024  # warm-up

    for idx, bs in enumerate(batch_sizes):
        torch.cuda.reset_peak_memory_stats()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
        preds = []
        t0 = time.time()
        for samples, _ in loader:
            if time.time() - t0 > max_seconds:
                break
            samples = samples.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                out = model(samples)
            preds.extend(out.argmax(1).cpu().tolist())
        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.empty_cache()
        gc.collect()

        if idx == 0:  # skip warm-up
            continue

        results.append({
            "batch_size": bs,
            "total_samples": len(preds),
            "time_s": round(elapsed, 3),
            "throughput_samples_per_s": round(len(preds) / elapsed, 1),
            "peak_memory_MB": round(peak_mem, 1),
            "model_memory_MB": round(model_mem, 1),
        })
        print(f"  bs={bs:4d}  {len(preds)/elapsed:7.0f} samples/s  "
              f"peak {peak_mem:.0f} MB")

    out_path = os.path.join(output_dir, "speed_test.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[speed] results → {out_path}")
