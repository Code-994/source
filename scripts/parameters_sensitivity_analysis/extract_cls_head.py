#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 phase1 的 trainer checkpoint（checkpoint-*/model.safetensors）里抽取 cls_head 权重，
导出成 phase2 需要的独立文件：<phase1_dir>/cls_head/{pytorch_model.bin, class_list.json}

背景：phase1 用 connector_only recipe（走 base.save()），只导出 vision_tower/connector，
      不导出独立 cls_head；但 cls_head 权重已训练并保存在 trainer checkpoint 里。
      phase2 的 train_v2.py 通过 MMTRAFFIC_CLS_HEAD_INIT 加载 cls_head/pytorch_model.bin，
      找不到会静默回退随机初始化（正是 v2 要修复的崩溃根因），故必须补出该文件。

class_list 与 dataset.py 完全一致：sorted(set(sample["class"]))，保证 phase1/phase2 行号对齐。

用法：
  python extract_cls_head.py <phase1_dir> <train_jsonl>
例：
  python extract_cls_head.py \
    /root/autodl-tmp/mmTraffic/output/parameter_sensitivity_analysis/1epoch/phase1 \
    /root/autodl-tmp/mmTraffic/data/Crossplatform_ios/nlp_output_noLLMclass_50_3000/train.jsonl
"""
import glob
import json
import os
import re
import sys

import torch
from safetensors import safe_open


def find_latest_checkpoint(phase1_dir):
    cks = glob.glob(os.path.join(phase1_dir, "checkpoint-*"))
    cks = [c for c in cks if os.path.isfile(os.path.join(c, "model.safetensors"))]
    if not cks:
        raise FileNotFoundError(f"未在 {phase1_dir} 找到含 model.safetensors 的 checkpoint-* 目录")
    # 按 step 号取最新
    def step_of(p):
        m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(cks, key=step_of)


def build_class_list(train_jsonl):
    names = set()
    with open(train_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls = json.loads(line).get("class", "")
            if cls:
                names.add(cls)
    return sorted(names)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    phase1_dir, train_jsonl = sys.argv[1], sys.argv[2]

    ckpt = find_latest_checkpoint(phase1_dir)
    sft = os.path.join(ckpt, "model.safetensors")
    print(f"[extract] 源 checkpoint: {sft}")

    state = {}
    with safe_open(sft, framework="pt") as f:
        keys = [k for k in f.keys() if k.startswith("cls_head.")]
        if not keys:
            raise KeyError("model.safetensors 中未找到 cls_head.* 权重")
        for k in keys:
            state[k.replace("cls_head.", "")] = f.get_tensor(k)  # 去前缀，匹配主实验格式
    if "weight" not in state:
        raise KeyError(f"抽出的 cls_head 缺少 weight；实际键：{list(state)}")
    n_rows = state["weight"].shape[0]
    print(f"[extract] cls_head 权重: weight={tuple(state['weight'].shape)} "
          f"bias={tuple(state['bias'].shape) if 'bias' in state else None} dtype={state['weight'].dtype}")

    class_list = build_class_list(train_jsonl)
    print(f"[extract] class_list（sorted）: {len(class_list)} 类")
    if len(class_list) != n_rows:
        raise ValueError(f"类别数({len(class_list)}) 与 cls_head 行数({n_rows}) 不一致，"
                         f"顺序对齐会出错，已中止。")

    out_dir = os.path.join(phase1_dir, "cls_head")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, "pytorch_model.bin"))
    with open(os.path.join(out_dir, "class_list.json"), "w") as f:
        json.dump(class_list, f, ensure_ascii=False, indent=2)
    print(f"[extract] ✓ 已写出: {out_dir}/pytorch_model.bin")
    print(f"[extract] ✓ 已写出: {out_dir}/class_list.json")


if __name__ == "__main__":
    main()
