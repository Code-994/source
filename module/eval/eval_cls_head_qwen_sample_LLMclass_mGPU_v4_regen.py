#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stratified sampling evaluation for TinyLLaVA traffic classification (LLMclass mode).

LLMclass 模式说明：
  - class 字段由 LLM 自行预测并输出到 JSON 中（不由 NetMamba 强制填写）
  - <cls_placeholder> 仍由 NetMamba top-1 embedding 替换，辅助 LLM 推理
  - required_fields 包含 class，evaluate() 不做 pred_cls 拼接

v3 新增（相对 v2）：
  1. cls_head 兜底机制：
     当 LLM 输出的 class 字段缺失或为 null 时，自动用 cls_head argmax 作为最终分类
     报告两套指标：class_exact_match_rate（纯LLM）和 class_exact_match_with_fallback（LLM+兜底）
  2. 必须通过 --cls_head_path 传入 cls_head 目录（含 pytorch_model.bin + class_list.json）才能启用兜底
  3. 子进程结果额外保存 cls_head_pred_idx 字段

Usage:
    # 多卡（4 卡，推荐）
    python eval_cls_head_qwen_sample_LLMclass_mGPU_v3.py \\
        --checkpoint_path  /path/to/checkpoint \\
        --vision_tower_path /path/to/vision_tower/pytorch_model.bin \\
        --cls_head_path    /path/to/checkpoint/cls_head \\
        --eval_data_path   /path/to/test.jsonl \\
        --image_folder     /path/to/npy_dir \\
        --output_dir       /path/to/eval_out \\
        --samples_per_class 20 --batch_size 4 --max_new_tokens 400 \\
        --num_gpus 4
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer

from tinyllava.model import TinyLlavaForConditionalGeneration
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from tinyllava.data.template import TemplateFactory
from tinyllava.data.template.base import Template


# ============================================================
# 工具函数
# ============================================================

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pcap_rel_to_npy_rel(sample_relpath: str) -> str:
    base, ext = os.path.splitext(sample_relpath)
    return sample_relpath if ext.lower() == ".npy" else (base + ".npy")


def check_npy_exists(sample: Dict, image_folder: str) -> bool:
    rel = sample.get("sample_relpath")
    # 🌟 核心：提取语义化类别名称，例如 "UC Browser"
    cls_name = sample.get("class", "")

    if not rel:
        return False

    npy_rel = pcap_rel_to_npy_rel(rel)
    filename = os.path.basename(npy_rel)

    # 🌟 核心：拼装新的真实路径 (image_folder / UC Browser / xxx.npy)
    path = os.path.join(image_folder, cls_name, filename)

    # 容错：如果没找到，尝试退回原始路径寻找
    if not os.path.exists(path):
        path = os.path.join(image_folder, npy_rel)

    return os.path.exists(path)


def strip_think(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think>', '', text)
    return text.strip()


# ============================================================
# 文件验证 & 分层采样
# ============================================================

def filter_valid_samples(data: List[Dict], image_folder: str):
    valid, invalid = [], []
    class_stats = defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0})

    print("\nChecking NPY file existence...")
    for sample in tqdm(data, desc="Validating files"):
        cls = sample.get("class", "unknown")
        class_stats[cls]["total"] += 1
        if check_npy_exists(sample, image_folder):
            valid.append(sample)
            class_stats[cls]["valid"] += 1
        else:
            invalid.append(sample)
            class_stats[cls]["invalid"] += 1

    print("\n" + "=" * 70)
    print("FILE VALIDATION REPORT")
    print("=" * 70)
    print(f"{'Class':<20} {'Total':>8} {'Valid':>8} {'Invalid':>8} {'Valid%':>8}")
    print("-" * 70)
    for cls in sorted(class_stats):
        s = class_stats[cls]
        pct = s["valid"] / s["total"] * 100 if s["total"] else 0
        print(f"{cls:<20} {s['total']:>8} {s['valid']:>8} {s['invalid']:>8} {pct:>7.1f}%")
    print("-" * 70)
    total_pct = len(valid) / len(data) * 100 if data else 0
    print(f"{'TOTAL':<20} {len(data):>8} {len(valid):>8} {len(invalid):>8} {total_pct:>7.1f}%")
    print("=" * 70 + "\n")

    no_valid = [c for c, s in class_stats.items() if s["valid"] == 0]
    if no_valid:
        print("WARNING: Classes with NO valid NPY files:", no_valid)

    return valid, {
        "total":       len(data),
        "valid":       len(valid),
        "invalid":     len(invalid),
        "class_stats": dict(class_stats),
    }


def stratified_sample(data: List[Dict], samples_per_class: int) -> List[Dict]:
    groups = defaultdict(list)
    for item in data:
        cls = item.get("class", "unknown")
        groups[cls].append(item)

    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)
    for cls in sorted(groups):
        print(f"  {cls:<20}: {len(groups[cls]):>6} samples")
    print("=" * 70 + "\n")

    sampled = []
    for cls in sorted(groups):
        items = groups[cls]
        n = min(samples_per_class, len(items))
        sampled.extend(random.sample(items, n))

    random.shuffle(sampled)
    print(f"Sampled {len(sampled)} samples total ({samples_per_class} per class max)\n")
    return sampled


# ============================================================
# Dataset
# ============================================================

class EvalDataset(Dataset):
    def __init__(
        self,
        data:         List[Dict],
        image_folder: str,
        tokenizer:    transformers.PreTrainedTokenizer,
        template:     Template,
        byte_length:  int = 1600,
    ):
        super().__init__()
        self.data         = data
        self.image_folder = image_folder
        self.tokenizer    = tokenizer
        self.template     = template
        self.byte_length  = byte_length

    def __len__(self):
        return len(self.data)

    def _load_npy(self, sample) -> torch.Tensor:
        rel = sample["sample_relpath"]
        # 🌟 核心：同样提取语义化类别名称
        cls_name = sample.get("class", "")

        npy_rel = pcap_rel_to_npy_rel(rel)
        filename = os.path.basename(npy_rel)

        # 🌟 核心：拼装新的真实路径
        path = os.path.join(self.image_folder, cls_name, filename)

        if not os.path.exists(path):
            path = os.path.join(self.image_folder, npy_rel)
            if not os.path.exists(path):
                # 抛出明确的错误，方便 debug
                raise FileNotFoundError(f"Missing NPY for class '{cls_name}': {path}")

        arr = np.load(path)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        else:
            arr = arr.reshape(1, -1)

        L = self.byte_length
        if arr.shape[1] < L:
            arr = np.concatenate([arr, np.zeros((1, L - arr.shape[1]), dtype=arr.dtype)], axis=1)
        else:
            arr = arr[:, :L]

        return torch.from_numpy(arr.astype(np.float32)) / 255.0

    def __getitem__(self, i) -> Dict[str, Any]:
        sample = self.data[i]

        # 从 jsonl 读取 user_text（含 <cls_placeholder>，在 prepare_inputs_labels 里在线替换）
        user_text = sample.get("user_text", "").strip()
        if not user_text:
            # 兜底：旧格式 jsonl，同步为新格式
            user_text = (
                f"{DEFAULT_IMAGE_TOKEN}\n"
                "Above are the raw traffic byte features extracted from a network flow.\n\n"
                "Based on the traffic byte features and the above classification result, "
                "return a single JSON object with the following keys:\n"
                "- class: the traffic category name\n"
                "- traits: an object describing byte-level characteristics\n"
                "- evidence: a list of 2~4 concrete byte-level observations\n"
                "- description: a paragraph of 2~3 sentences summarizing the traffic\n"
                "- notes: a single security-relevant observation or recommendation"
            )

        # v2: 追加 JSON 格式约束，防止模型生成 // 注释或 "..." 占位符
        FORMAT_CONSTRAINT = (
            "\n\nIMPORTANT: Your response must be a single valid JSON object only. "
            "Do NOT include any JavaScript-style comments (// or /* */) inside the JSON. "
            "Do NOT use '...' or any placeholder as a key name or value. "
            "Every required field must have a specific, concrete value. "
            "Do NOT add any text outside the JSON object."
        )
        user_text = user_text + FORMAT_CONSTRAINT

        messages = [
            {"from": "human", "value": user_text},
            {"from": "gpt",   "value": ""},
        ]
        encoded   = self.template.encode(messages, self.tokenizer, mode="eval")
        input_ids = encoded["input_ids"]
        image     = self._load_npy(sample)

        return {
            "input_ids":      input_ids,
            "image":          image,
            "ground_truth":   sample.get("target", ""),
            "gt_class":       sample.get("class", "unknown"),
            "sample_relpath": sample.get("sample_relpath", ""),
        }


# collate_fn 改为可调用类，避免多进程 pickle 失败（lambda 不可序列化）
class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids       = [item["input_ids"]      for item in batch]
        images          = [item["image"]          for item in batch]
        ground_truths   = [item["ground_truth"]   for item in batch]
        sample_relpaths = [item["sample_relpath"] for item in batch]
        gt_classes = [item["gt_class"] for item in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        images         = torch.stack(images, dim=0)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "images":          images,
            "ground_truths":   ground_truths,
            "sample_relpaths": sample_relpaths,
            "gt_classes": gt_classes,
        }


# ============================================================
# 模型加载
# ============================================================

def load_model(
    checkpoint_path:    str,
    device:             str = "cuda",
    vision_tower_path:  str = None,
    cls_head_path:      str = None,
    class_list:         List[str] = None,
) -> TinyLlavaForConditionalGeneration:
    print(f"[{device}] Loading model from: {checkpoint_path}")

    parent_dir  = os.path.dirname(checkpoint_path) if "checkpoint-" in os.path.basename(checkpoint_path) else checkpoint_path
    config_path = checkpoint_path if os.path.exists(os.path.join(checkpoint_path, "config.json")) else parent_dir
    config      = TinyLlavaConfig.from_pretrained(config_path)
    model       = TinyLlavaForConditionalGeneration(config)

    # ── LLM ──
    llm_path = config.llm_model_name_or_path
    if os.path.exists(llm_path):
        model.language_model = model.language_model.from_pretrained(llm_path)
    else:
        weight_path = os.path.join(parent_dir, "language_model", "pytorch_model.bin")
        if os.path.exists(weight_path):
            state   = torch.load(weight_path, map_location="cpu", weights_only=False)
            cleaned = {k.replace(".base_layer.", "."): v for k, v in state.items()}
            model.language_model.load_state_dict(cleaned, strict=False)
            print(f"  [{device}] LLM loaded from saved weights (key cleanup applied)")

    # ── Vision Tower（参数化路径）──
    if vision_tower_path is None:
        raise ValueError("--vision_tower_path 不能为空，请通过参数传入 checkpoint-best.pth 路径")
    model.vision_tower._load_model(
        vision_tower_name=None,
        pretrained_vision_tower_path=vision_tower_path,
    )
    print(f"  [{device}] vision_tower loaded from: {vision_tower_path}")

    # ── Connector ──
    conn_path = os.path.join(parent_dir, "connector", "pytorch_model.bin")
    if not os.path.exists(conn_path):
        # 兼容某些使用 safetensors 的环境
        conn_path = os.path.join(parent_dir, "connector", "model.safetensors")
    if os.path.exists(conn_path):
        state = torch.load(conn_path, map_location="cpu", weights_only=False)
        model.connector.load_state_dict(state, strict=False)
        print(f"  {conn_path} 存在， [{device}] connector loaded")

    # ── LoRA Adapter ──
    adapter_cfg   = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_model = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if os.path.exists(adapter_cfg) and os.path.exists(adapter_model):
        size_mb = os.path.getsize(adapter_model) / 1024 / 1024
        print(f"  [{device}] Adapter size: {size_mb:.1f} MB")
        if size_mb < 0.001:
            print(f"  [{device}] WARNING: Adapter file too small, skipping")
        else:
            model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
            try:
                model = model.merge_and_unload()
                print(f"  [{device}] LoRA merged")
            except Exception as e:
                print(f"  [{device}] LoRA merge failed: {e}")
    else:
        print(f"  [{device}] WARNING: No adapter found, using base weights only")

    # ── Tokenizer（强制用训练输出目录替换，确保含所有 special tokens）──
    # 注意：LoRA merge 后再替换，避免被覆盖
    model = model.to(device)
    # 训练用 bf16，推理时统一转换避免 vision_tower(float32) 与 connector(bf16) dtype 不匹配
    # model = model.to(device)
    model = model.to(torch.bfloat16)
    model.eval()

    ckpt_tokenizer = AutoTokenizer.from_pretrained(parent_dir)
    model.tokenizer = ckpt_tokenizer

    # 二次替换兜底（防止 merge_and_unload 内部重置）
    ckpt_tokenizer = AutoTokenizer.from_pretrained(parent_dir)
    model.tokenizer = ckpt_tokenizer

    print(f"  [{device}] tokenizer replaced, vocab_size={len(ckpt_tokenizer)}")

    # ── cls_head（可选）──
    if cls_head_path is not None and os.path.exists(cls_head_path):
        weight_file = os.path.join(cls_head_path, "pytorch_model.bin") \
            if os.path.isdir(cls_head_path) else cls_head_path
        if os.path.exists(weight_file):
            cls_state = torch.load(weight_file, map_location=device, weights_only=False)
            # 从权重推断 num_classes：weight shape = (num_classes, hidden_size)
            num_classes = cls_state["weight"].shape[0]
            # 获取 base model（如果 PeftModel 已 merge_and_unload 则就是 model 本身）
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            base.init_cls_head(num_classes)
            base.cls_head.load_state_dict(cls_state)
            base.cls_head.to(device)
            base.cls_head.eval()
            print(f"  [{device}] cls_head loaded: num_classes={num_classes}")
        else:
            print(f"  [{device}] WARNING: cls_head weight not found at {weight_file}")

    print(f"[{device}] Model loaded successfully\n")
    return model


# ============================================================
# 前缀强制重生成（方法 A）：LLM 漏 class 时，用 cls_head 兜底类别作 JSON 前缀，
# 让 LLM 续写 traits/evidence/description/notes，使整条报告自洽。仅推理、不重训。
# ============================================================

@torch.no_grad()
def regen_with_forced_class(model, tokenizer, image_1, input_ids_1, fb_class,
                            device, max_new_tokens, eos_token_ids):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    attn = torch.ones_like(input_ids_1, dtype=torch.bool)
    (_, _pos, attn_new, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids_1, position_ids=None, attention_mask=attn,
        past_key_values=None, labels=None, images=image_1, image_sizes=None,
    )
    base._cls_logits_cache = None  # 清掉本次重生成产生的 cache，避免污染后续 batch 的 cls_head 读取
    prefix = '{"class": "' + fb_class + '", '
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    prefix_embeds = model.language_model.get_input_embeddings()(prefix_ids).to(inputs_embeds.dtype)
    full_embeds = torch.cat([inputs_embeds, prefix_embeds], dim=1)
    full_attn = torch.cat(
        [attn_new, torch.ones((1, prefix_ids.shape[1]), dtype=attn_new.dtype, device=device)], dim=1)
    out = model.language_model.generate(
        inputs_embeds=full_embeds, attention_mask=full_attn,
        max_new_tokens=max_new_tokens, use_cache=True, do_sample=False,
        eos_token_id=eos_token_ids, no_repeat_ngram_size=5, repetition_penalty=1.1,
    )
    cont = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    return prefix + cont


# ============================================================
# 推理（单进程内）
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader:     DataLoader,
    device:         str,
    max_new_tokens: int,
    temperature:    float,
    top_p:          float,
    class_list:     List[str] = None,
    regen_on_fallback: bool = False,
) -> tuple:
    tokenizer = model.tokenizer
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    has_cls_head = hasattr(base, "cls_head") and base.cls_head is not None

    im_end_id     = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_ids = list({t for t in [tokenizer.eos_token_id, im_end_id] if t is not None})
    print(f"[{device}] eos_token_ids: {eos_token_ids}  (im_end_id={im_end_id})")
    if has_cls_head:
        print(f"[{device}] cls_head detected, will collect cls_head predictions")

    predictions      = []
    ground_truths    = []
    gt_classes_all   = []
    sample_relpaths  = []
    cls_head_preds   = []   # cls_head 预测的类别索引

    for batch in tqdm(dataloader, desc=f"[{device}] Evaluating"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images         = batch["images"].to(device, dtype=model.dtype)

        # ── prepare_inputs_labels_for_multimodal（推理模式）──
        # 这一步会调用 encode_images，从而填充 _cls_logits_cache
        (
            _,
            position_ids,
            attention_mask_new,
            _,
            inputs_embeds,
            _,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids       = input_ids,
            position_ids    = None,
            attention_mask  = attention_mask,
            past_key_values = None,
            labels          = None,
            images          = images,
            image_sizes     = None,
        )

        # ── 读取 cls_head 预测（hoist batch_preds，供重生成兜底使用）──
        batch_preds = [-1] * images.shape[0]
        if has_cls_head:
            cls_logits = getattr(base, "_cls_logits_cache", None)
            if cls_logits is not None:
                batch_preds = cls_logits.argmax(dim=-1).cpu().tolist()
                base._cls_logits_cache = None
            cls_head_preds.extend(batch_preds)

        gen_kwargs = dict(
            inputs_embeds        = inputs_embeds,
            attention_mask       = attention_mask_new,
            position_ids         = position_ids,
            max_new_tokens       = max_new_tokens,
            use_cache            = True,
            eos_token_id         = eos_token_ids,
            no_repeat_ngram_size = 5,
            repetition_penalty   = 1.1,
        )
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=top_p, do_sample=True)
        else:
            gen_kwargs["do_sample"] = False

        assert inputs_embeds.shape[1] == attention_mask_new.shape[1], \
            f"shape mismatch: embeds={inputs_embeds.shape}, mask={attention_mask_new.shape}"

        outputs         = model.language_model.generate(**gen_kwargs)
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # ── LLMclass：解析 LLM 输出；class 缺失且开启 regen 时，用 cls_head 兜底类别强制重生成报告 ──
        for j, text in enumerate(generated_texts):
            text = strip_think(text)
            obj  = parse_json_output(text)
            llm_class = ((obj.get("class") if isinstance(obj, dict) else None) or "")
            if isinstance(obj, dict):
                final_text = json.dumps(obj, ensure_ascii=False)
            else:
                final_text = json.dumps({"class": None, "raw": text}, ensure_ascii=False)

            regenerated = False
            if (not llm_class) and regen_on_fallback and class_list:
                cls_idx = batch_preds[j] if j < len(batch_preds) else -1
                if 0 <= cls_idx < len(class_list):
                    fb_class   = class_list[cls_idx]
                    regen_text = regen_with_forced_class(
                        model, tokenizer, images[j:j + 1], input_ids[j:j + 1],
                        fb_class, device, max_new_tokens, eos_token_ids,
                    )
                    robj = parse_json_output(strip_think(regen_text))
                    final_text = (json.dumps(robj, ensure_ascii=False) if isinstance(robj, dict)
                                  else json.dumps({"class": fb_class, "raw": regen_text}, ensure_ascii=False))
                    regenerated = True

            predictions.append({"text": final_text, "llm_class_raw": llm_class, "regenerated": regenerated})
        ground_truths.extend(batch["ground_truths"])
        sample_relpaths.extend(batch["sample_relpaths"])
        gt_classes_all.extend(batch["gt_classes"])

    return predictions, ground_truths, gt_classes_all, sample_relpaths, cls_head_preds


# ============================================================
# 子进程入口（多卡模式）
# ============================================================

def worker_process(rank: int, args, sampled_data: List[Dict], tmp_dir: str):
    """每个子进程独占 GPU rank，处理分配给它的数据切片，结果写到临时文件。"""
    # 必须在最开始显式设置 CUDA 设备，否则 Triton/mamba 内核找不到正确的 GPU 上下文
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # 按 rank 切分数据
    chunk_size = math.ceil(len(sampled_data) / args.num_gpus)
    start = rank * chunk_size
    end   = min(start + chunk_size, len(sampled_data))
    chunk = sampled_data[start:end]

    if not chunk:
        print(f"[rank {rank}] 无数据，退出")
        return

    cls_head_path = getattr(args, "cls_head_path", None)
    model     = load_model(args.checkpoint_path, device=device,
                           vision_tower_path=args.vision_tower_path,
                           cls_head_path=cls_head_path)
    tokenizer = model.tokenizer
    template  = TemplateFactory(args.conv_version)()

    dataset = EvalDataset(
        data         = chunk,
        image_folder = args.image_folder,
        tokenizer    = tokenizer,
        template     = template,
        byte_length  = args.byte_length,
    )
    # 子进程内不能嵌套多进程 DataLoader，num_workers 必须为 0
    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = CollateFn(tokenizer),
        num_workers = 0,
        pin_memory  = False,
    )

    # 构建 class_list（从分配到此 rank 的数据中收集所有类别，排序）
    class_list = getattr(args, "_class_list", None)

    predictions, ground_truths, gt_classes_all, sample_relpaths, cls_head_preds = evaluate(
        model          = model,
        dataloader     = dataloader,
        device         = device,
        max_new_tokens = args.max_new_tokens,
        temperature    = args.temperature,
        top_p          = args.top_p,
        class_list     = class_list,
        regen_on_fallback = getattr(args, "regen_on_fallback", False),
    )

    # 写到临时文件，主进程合并
    tmp_file = os.path.join(tmp_dir, f"rank_{rank}.jsonl")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for i, (pred, gt, cls, rel) in enumerate(zip(predictions, ground_truths, gt_classes_all, sample_relpaths)):
            row = {
                "prediction":     pred["text"],
                "ground_truth":   gt,
                "gt_class":       cls,
                "sample_relpath": rel,
                "llm_class_raw":  pred.get("llm_class_raw", ""),
                "regenerated":    pred.get("regenerated", False),
            }
            if cls_head_preds:
                row["cls_head_pred_idx"] = cls_head_preds[i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[rank {rank}] Done. {len(predictions)} samples → {tmp_file}")


# ============================================================
# 指标计算
# ============================================================

def parse_json_output(text: str) -> Optional[Dict]:
    text = strip_think(text)
    for marker in ("```json", "```"):
        if marker in text:
            start = text.find(marker) + len(marker)
            end   = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e])
            except Exception:
                pass
    return None


def calculate_metrics(
    predictions:   List[Dict],
    ground_truths: List[str],
    gt_classes:    List[str],
    cls_head_preds: List[int] = None,
    class_list:    List[str] = None,
) -> Dict:
    # LLMclass 模式：class 字段由 LLM 生成，required_fields 包含 class
    required_fields = {"class", "traits", "evidence", "description", "notes"}
    class_metrics   = defaultdict(lambda: {
        "total": 0, "correct": 0, "valid_json": 0, "correct_with_fallback": 0
    })

    def _normalize(s):
        """宽松化：忽略大小写、多余空格、引号、末尾标点、连字符/下划线差异"""
        if not isinstance(s, str):
            return ""
        import re as _re
        s = s.strip().strip('"\'').strip()
        s = s.lower()
        s = _re.sub(r'[._\-]+', ' ', s)
        s = _re.sub(r'[^\w\s]', '', s)
        s = _re.sub(r'\s+', ' ', s).strip()
        return s

    total = len(predictions)
    valid_json_count = complete_fields_count = 0
    exact_match_count = fuzzy_match_count = contain_match_count = 0
    # v3 新增：cls_head 兜底后的指标
    exact_match_with_fallback = 0
    fallback_used = 0
    fallback_correct = 0

    for i, (pred, gt_str, gt_class) in enumerate(zip(predictions, ground_truths, gt_classes)):
        pred_json = parse_json_output(pred.get("text", ""))
        class_metrics[gt_class]["total"] += 1

        pred_class = ""
        if pred_json is not None and isinstance(pred_json, dict):
            valid_json_count += 1
            class_metrics[gt_class]["valid_json"] += 1
            if required_fields.issubset(pred_json.keys()):
                complete_fields_count += 1
            pred_class = pred_json.get("class") or ""   # None 也归为空（仅 json/字段统计用）

        # ── 纯 LLM 分类指标：用「原始 LLM 输出的 class」，重生成兜底不污染该指标 ──
        # v4 写入 llm_class_raw（= 第一遍 LLM 自己给的 class）；旧数据无此字段时回退到 pred_class
        llm_class = pred.get("llm_class_raw", None)
        if llm_class is None:
            llm_class = pred_class
        llm_class = llm_class or ""
        norm_pred = _normalize(llm_class)
        norm_gt   = _normalize(gt_class)
        if llm_class == gt_class:
            exact_match_count += 1
            class_metrics[gt_class]["correct"] += 1
        if norm_pred == norm_gt:
            fuzzy_match_count += 1
        if norm_pred and norm_gt and (norm_pred in norm_gt or norm_gt in norm_pred):
            contain_match_count += 1

        # ── cls_head 兜底（分类口径，不变）──
        # 当 LLM 未输出有效 class 时，用 cls_head top-1 替代
        effective_class = llm_class
        if (not llm_class) and cls_head_preds and class_list:
            idx = cls_head_preds[i] if i < len(cls_head_preds) else -1
            if 0 <= idx < len(class_list):
                effective_class = class_list[idx]
                fallback_used += 1
                if effective_class == gt_class:
                    fallback_correct += 1
        if effective_class == gt_class:
            exact_match_with_fallback += 1
            class_metrics[gt_class]["correct_with_fallback"] += 1

    per_class = {}
    for cls, cm in class_metrics.items():
        t = cm["total"]
        per_class[cls] = {
            "total_samples":             t,
            "json_parse_rate":           cm["valid_json"]            / t * 100 if t else 0,
            "class_match_rate":          cm["correct"]               / t * 100 if t else 0,
            "class_match_with_fallback": cm["correct_with_fallback"] / t * 100 if t else 0,
        }

    # ── cls_head 独立精度（不依赖 LLM 输出，单独评估 cls_head 在测试集上的表现）──
    cls_head_accuracy = None
    cls_head_correct = 0
    cls_head_total = 0
    if cls_head_preds and class_list:
        for pred_idx, gt_cls in zip(cls_head_preds, gt_classes):
            if pred_idx < 0 or pred_idx >= len(class_list):
                continue
            cls_head_total += 1
            if class_list[pred_idx] == gt_cls:
                cls_head_correct += 1
        if cls_head_total > 0:
            cls_head_accuracy = cls_head_correct / cls_head_total * 100

    result = {
        "total_samples":                    total,
        "valid_json_count":                 valid_json_count,
        "complete_fields_count":            complete_fields_count,
        "json_parse_rate":                  valid_json_count        / total * 100 if total else 0,
        "field_completeness_rate":          complete_fields_count   / total * 100 if total else 0,
        "class_exact_match_rate":           exact_match_count       / total * 100 if total else 0,
        "class_exact_match_with_fallback":  exact_match_with_fallback / total * 100 if total else 0,
        "class_fuzzy_match_rate":           fuzzy_match_count       / total * 100 if total else 0,
        "class_contain_match_rate":         contain_match_count     / total * 100 if total else 0,
        "fallback_used":                    fallback_used,
        "fallback_correct":                 fallback_correct,
        "regenerated_count":                sum(1 for p in predictions if p.get("regenerated")),
        "per_class_metrics":                per_class,
    }
    if cls_head_accuracy is not None:
        result["cls_head_accuracy"] = cls_head_accuracy
        result["cls_head_correct"]  = cls_head_correct
        result["cls_head_total"]    = cls_head_total
        print(f"\n[CLS_HEAD]  standalone test acc:   {cls_head_accuracy:.2f}% ({cls_head_correct}/{cls_head_total})")
    print(f"[LLM only]  class_exact_match:      {exact_match_count/total*100:.2f}% ({exact_match_count}/{total})")
    if cls_head_preds:
        print(f"[FALLBACK]  LLM-empty→cls_head:   used={fallback_used}, correct={fallback_correct}")
        print(f"[COMBINED]  LLM + fallback acc:    {exact_match_with_fallback/total*100:.2f}% ({exact_match_with_fallback}/{total})")

    return result


# ============================================================
# 结果保存
# ============================================================

def save_results(predictions, ground_truths, metrics, output_dir, sample_relpaths=None,
                 cls_head_preds=None, class_list=None):
    os.makedirs(output_dir, exist_ok=True)

    pred_file = os.path.join(output_dir, "predictions.jsonl")
    with open(pred_file, "w", encoding="utf-8") as f:
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # 尝试解析并获取 LLM 输出的 class
            pred_json = parse_json_output(pred.get("text", ""))
            pred_class = ""
            if pred_json and isinstance(pred_json, dict):
                pred_class = pred_json.get("class") or ""

            # v3：若 class 缺失，用 cls_head argmax 填充后回写到 prediction 字段
            if (not pred_class) and cls_head_preds and class_list:
                idx = cls_head_preds[i] if i < len(cls_head_preds) else -1
                if 0 <= idx < len(class_list):
                    fallback_class = class_list[idx]
                    if pred_json and isinstance(pred_json, dict):
                        pred_json["class"] = fallback_class
                        pred_json["_class_source"] = "cls_head_fallback"
                    else:
                        pred_json = {"class": fallback_class, "_class_source": "cls_head_fallback"}
                    pred = {"text": json.dumps(pred_json, ensure_ascii=False)}

            row = {"sample_id": i, "prediction": pred["text"], "ground_truth": gt}
            if sample_relpaths:
                row["sample_relpath"] = sample_relpaths[i]
            if "regenerated" in pred:
                row["regenerated"]   = pred.get("regenerated", False)
                row["llm_class_raw"] = pred.get("llm_class_raw", "")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Predictions → {pred_file}")

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics     → {metrics_file}")

    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    for k, v in metrics.items():
        if k not in ("per_class_metrics",):
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)
    print(f"  {'Class':<22} {'Samples':>8} {'JSON%':>8} {'Match%':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    for cls in sorted(metrics["per_class_metrics"]):
        cm = metrics["per_class_metrics"][cls]
        print(f"  {cls:<22} {cm['total_samples']:>8} "
              f"{cm['json_parse_rate']:>7.1f}% {cm['class_match_rate']:>7.1f}%")
    print("=" * 70 + "\n")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path",   required=True,
                        help="TinyLLaVA checkpoint 目录（含 adapter_config.json）")
    parser.add_argument("--vision_tower_path", required=True,
                        help="NetMamba checkpoint-best.pth 路径")
    parser.add_argument("--eval_data_path",    required=True)
    parser.add_argument("--image_folder",      required=True)
    parser.add_argument("--output_dir",        default="./eval_results")
    parser.add_argument("--samples_per_class", type=int,   default=10)
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--max_new_tokens",    type=int,   default=500)
    parser.add_argument("--temperature",       type=float, default=0.0)
    parser.add_argument("--top_p",             type=float, default=0.9)
    parser.add_argument("--byte_length",       type=int,   default=1600)
    parser.add_argument("--conv_version",      default="qwen3_instruct")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--num_gpus",          type=int,   default=1,
                        help="使用几张GPU并行推理（默认1，多卡时自动切分数据）")
    parser.add_argument("--cls_head_path",     default=None,
                        help="cls_head 权重目录（含 pytorch_model.bin），可选")
    parser.add_argument("--train_data_path",   default=None,
                        help="训练集 jsonl 路径，用于生成 class_list.json（比从测试集构建更可靠）")
    parser.add_argument("--regen_on_fallback", action="store_true",
                        help="LLM 未输出 class 时，用 cls_head 兜底类别作前缀强制重生成整条报告（evidence/description 等随之自洽）")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 数据准备（主进程统一完成）──
    print("Loading dataset...")
    full_data = read_jsonl(args.eval_data_path)
    print(f"Total samples: {len(full_data)}")

    valid_data, val_report = filter_valid_samples(full_data, args.image_folder)
    if not valid_data:
        print("ERROR: No valid samples found!")
        return

    sampled_data = stratified_sample(valid_data, args.samples_per_class)

    # ── 构建 class_list（cls_head idx → class name 映射）──
    # 优先级：① cls_head 目录已有 class_list.json → 直接加载
    #         ② --train_data_path 提供 → 从训练集生成并保存到 cls_head 目录
    #         ③ 兜底 → 从测试集按 sorted() 构建（测试集需覆盖全部类别才准确）
    class_list = None
    if args.cls_head_path:
        cl_json = os.path.join(args.cls_head_path, "class_list.json")
        if os.path.exists(cl_json):
            with open(cl_json, "r", encoding="utf-8") as f:
                class_list = json.load(f)
            print(f"[INFO] class_list loaded from {cl_json}: {len(class_list)} classes")
        elif args.train_data_path and os.path.exists(args.train_data_path):
            train_data = read_jsonl(args.train_data_path)
            class_list = sorted(set(s.get("class", "unknown") for s in train_data))
            os.makedirs(args.cls_head_path, exist_ok=True)
            with open(cl_json, "w", encoding="utf-8") as f:
                json.dump(class_list, f, ensure_ascii=False, indent=2)
            print(f"[INFO] class_list generated from train data → saved to {cl_json}: {len(class_list)} classes")
    if class_list is None:
        source = "train" if args.train_data_path and os.path.exists(args.train_data_path) else "eval"
        src_data = read_jsonl(args.train_data_path) if source == "train" else full_data
        class_list = sorted(set(s.get("class", "unknown") for s in src_data))
        print(f"[INFO] class_list built from {source} data: {len(class_list)} classes (sorted, same as training)")

    # ── 单卡模式 ──
    if args.num_gpus <= 1:
        device    = "cuda:0"
        model     = load_model(args.checkpoint_path, device=device,
                               vision_tower_path=args.vision_tower_path,
                               cls_head_path=args.cls_head_path,
                               class_list=class_list)
        tokenizer = model.tokenizer
        template  = TemplateFactory(args.conv_version)()

        dataset = EvalDataset(
            data         = sampled_data,
            image_folder = args.image_folder,
            tokenizer    = tokenizer,
            template     = template,
            byte_length  = args.byte_length,
        )
        dataloader = DataLoader(
            dataset,
            batch_size  = args.batch_size,
            shuffle     = False,
            collate_fn  = CollateFn(tokenizer),
            num_workers = 4,
            pin_memory  = True,
        )
        predictions, ground_truths, gt_classes_all, sample_relpaths, cls_head_preds = evaluate(
            model          = model,
            dataloader     = dataloader,
            device         = device,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_p          = args.top_p,
            class_list     = class_list,
            regen_on_fallback = getattr(args, "regen_on_fallback", False),
        )

    # ── 多卡模式：spawn 子进程，每张卡处理一个切片 ──
    else:
        import torch.multiprocessing as mp
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
        if num_gpus < args.num_gpus:
            print(f"WARNING: 只检测到 {num_gpus} 张 GPU，忽略 --num_gpus {args.num_gpus}")
        args.num_gpus = num_gpus
        args._class_list = class_list  # 传给子进程
        print(f"[INFO] 多卡推理：{num_gpus} 张 GPU，共 {len(sampled_data)} 样本")

        tmp_dir = tempfile.mkdtemp(prefix="eval_llmcls_tmp_")
        mp.set_start_method("spawn", force=True)

        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=worker_process,
                args=(rank, args, sampled_data, tmp_dir),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 合并所有子进程结果（保持原始顺序：rank0 → rank1 → ...）
        predictions     = []
        ground_truths   = []
        gt_classes_all  = []
        sample_relpaths = []
        cls_head_preds  = []

        for rank in range(num_gpus):
            tmp_file = os.path.join(tmp_dir, f"rank_{rank}.jsonl")
            if not os.path.exists(tmp_file):
                print(f"WARNING: rank {rank} 结果文件不存在，跳过")
                continue
            with open(tmp_file, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    predictions.append({
                        "text":          row["prediction"],
                        "llm_class_raw": row.get("llm_class_raw", None),
                        "regenerated":   row.get("regenerated", False),
                    })
                    ground_truths.append(row["ground_truth"])
                    gt_classes_all.append(row["gt_class"])
                    sample_relpaths.append(row["sample_relpath"])
                    if "cls_head_pred_idx" in row:
                        cls_head_preds.append(row["cls_head_pred_idx"])

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[INFO] 合并完成，共 {len(predictions)} 条结果")

    # ── 统一计算指标 ──
    metrics = calculate_metrics(predictions, ground_truths, gt_classes_all,
                                cls_head_preds=cls_head_preds if cls_head_preds else None,
                                class_list=class_list)
    metrics["eval_info"] = {
        "total_dataset":     len(full_data),
        "valid_samples":     len(valid_data),
        "sampled":           len(sampled_data),
        "samples_per_class": args.samples_per_class,
        "checkpoint":        args.checkpoint_path,
        "vision_tower_path": args.vision_tower_path,
        "num_gpus":          args.num_gpus,
        "seed":              args.seed,
    }
    metrics["validation_report"] = val_report

    save_results(predictions, ground_truths, metrics, args.output_dir, sample_relpaths,
                 cls_head_preds=cls_head_preds if cls_head_preds else None,
                 class_list=class_list)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()