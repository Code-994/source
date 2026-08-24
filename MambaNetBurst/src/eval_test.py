"""
单卡 test set 评估脚本，输出 test_stats.json / test_per_class.json / train_summary.json。

用法:
  python eval_test.py \
      --data_path /path/to/dataset_sampled1 \
      --nb_classes 12 \
      --resume /path/to/checkpoint-best.pth \
      --output_dir /path/to/output_dir
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
_DATASET_DIR = os.path.join(_ROOT_DIR, "dataset")
for _p in [_SRC_DIR, _ROOT_DIR, _DATASET_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import models_net_mamba2 as model_zoo
from engine import evaluate, load_checkpoint
from dataset_burst import get_dataset


def main():
    p = argparse.ArgumentParser("MambaNetBurst test-set evaluation")
    p.add_argument("--data_path", required=True)
    p.add_argument("--nb_classes", default=12, type=int)
    p.add_argument("--resume", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", default=256, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--byte_length", default=1600, type=int)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_test = get_dataset(args.data_path, split="test",
                               fmt="npy", byte_length=args.byte_length)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[data] test={len(dataset_test)}  classes={dataset_test.classes}")

    model = model_zoo.mambanetburst_classifier(num_classes=args.nb_classes)
    model.to(device)

    load_checkpoint(args.resume, model)

    test_stats = evaluate(loader_test, model, device)

    print(
        f"[test] acc={test_stats['acc']:.4f}  "
        f"f1={test_stats['macro_f1']:.4f}  "
        f"pre={test_stats['macro_pre']:.4f}  "
        f"rec={test_stats['macro_rec']:.4f}"
    )

    out = Path(args.output_dir)
    with open(out / "test_stats.json", "w") as f:
        json.dump(
            {k: v for k, v in test_stats.items()
             if k not in ("pre_per_class", "rec_per_class", "f1_per_class", "support_per_class", "cm")},
            f, indent=2,
        )

    with open(out / "test_per_class.json", "w") as f:
        json.dump(
            {
                "classes": dataset_test.classes,
                "pre_per_class":     test_stats["pre_per_class"],
                "rec_per_class":     test_stats["rec_per_class"],
                "f1_per_class":      test_stats["f1_per_class"],
                "support_per_class": test_stats["support_per_class"],
                "cm":                test_stats["cm"],
            },
            f, indent=2,
        )

    with open(out / "train_summary.json", "w") as f:
        json.dump(
            {
                "model": "mambanetburst_classifier",
                "dataset": args.data_path,
                "nb_classes": args.nb_classes,
                "checkpoint": args.resume,
            },
            f, indent=2,
        )

    print(f"[done] saved test_stats.json / test_per_class.json / train_summary.json → {args.output_dir}")


if __name__ == "__main__":
    main()
