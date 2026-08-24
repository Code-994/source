"""
Mamba-1 vs Mamba-2 速度对比基准。

用法:
    python speed_benchmark.py              # 默认 batch=128, seq=1600, N=200
    python speed_benchmark.py --batch 64 --warmup 50 --iters 300
"""

import argparse
import time

import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models_net_mamba2 as zoo


def benchmark(model, x, warmup: int, iters: int, forward_only: bool = False):
    """
    返回 (throughput_samples_per_sec, latency_ms_per_batch)
    """
    model.eval() if forward_only else model.train()
    device = x.device

    # Warmup（让 CUDA JIT / Triton 编译完成）
    for _ in range(warmup):
        with torch.cuda.amp.autocast():
            out = model(x)
        if not forward_only:
            out.sum().backward()
        torch.cuda.synchronize()

    # 正式计时
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.cuda.amp.autocast():
            out = model(x)
        if not forward_only:
            out.sum().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    batch_size = x.shape[0]
    throughput = batch_size * iters / elapsed
    latency_ms = elapsed / iters * 1000
    return throughput, latency_ms


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"Batch  : {args.batch}  SeqLen : {args.seq_len}  "
          f"Warmup : {args.warmup}  Iters : {args.iters}")
    print(f"Mode   : {'forward only' if args.forward_only else 'forward + backward'}")
    print("=" * 60)

    x = torch.randint(0, 256, (args.batch, args.seq_len), device=device)

    results = {}
    for name, factory in [
        ("Mamba-2 (d_state=16)", zoo.mambanetburst_classifier),
        ("Mamba-1 (d_state=16)", zoo.mambanetburst_mamba1_classifier),
    ]:
        model = factory(num_classes=args.nb_classes, byte_length=args.seq_len).to(device)
        n_params = count_params(model)

        thr, lat = benchmark(
            model, x,
            warmup=args.warmup,
            iters=args.iters,
            forward_only=args.forward_only,
        )
        results[name] = (thr, lat, n_params)
        print(f"{name}")
        print(f"  params      : {n_params/1e6:.2f} M")
        print(f"  throughput  : {thr:,.0f} samples/sec")
        print(f"  latency     : {lat:.2f} ms/batch")
        print()

    # 对比
    thr2, lat2, _ = results["Mamba-2 (d_state=16)"]
    thr1, lat1, _ = results["Mamba-1 (d_state=16)"]
    print("=" * 60)
    print(f"Mamba-2 / Mamba-1 throughput ratio : {thr2/thr1:.2f}×")
    print(f"Mamba-2 / Mamba-1 latency ratio    : {lat2/lat1:.2f}×")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch",        type=int, default=128)
    p.add_argument("--seq_len",      type=int, default=1600)
    p.add_argument("--nb_classes",   type=int, default=7)
    p.add_argument("--warmup",       type=int, default=50)
    p.add_argument("--iters",        type=int, default=200)
    p.add_argument("--forward_only", action="store_true",
                   help="只测前向（推理速度），默认前向+反向（训练速度）")
    run(p.parse_args())
