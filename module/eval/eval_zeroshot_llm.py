#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_zeroshot_llm.py — LLM zero-shot baseline for the paper's main table.

Feeds the raw v3 byte matrix (10 packets x 160 bytes) as a hexadecimal text
sequence to the ORIGINAL (untrained) Qwen3 LLM, with the same structured-report
JSON schema used by mmTraffic. No vision tower, no connector, no fine-tuning.
Output predictions.jsonl is compatible with evaluate_predictions.py.

Usage:
    python eval_zeroshot_llm.py \
        --test_jsonl /root/autodl-tmp/mmTraffic/data/USTC-TFC-2016/nlp_output_LLMclass_3000_6000/test.jsonl \
        --npy_root   /root/autodl-tmp/mmTraffic/data/USTC-TFC-2016/USTC-TFC-2016_npy_v3_balacned_3000_6000_train_test_splited/test \
        --model_path /root/autodl-tmp/mmTraffic/model/Qwen3-1.7B \
        --output_dir /root/autodl-tmp/mmTraffic/output/USTC-TFC-2016/zeroshot_qwen3_1.7b \
        --samples_per_class 99999 --batch_size 8 --max_new_tokens 512
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_npy_path(sample_relpath, npy_root):
    base, _ = os.path.splitext(sample_relpath)
    return os.path.join(npy_root, base + ".npy")


def npy_to_hex(npy_path, max_bytes=1600):
    arr = np.load(npy_path)
    arr = np.asarray(arr, dtype=np.uint8).reshape(-1)[:max_bytes]
    return " ".join(f"{b:02x}" for b in arr)


def build_prompt(class_list, hex_str):
    classes_str = ", ".join(class_list)
    return (
        "The following is a raw network traffic capture represented as a "
        "hexadecimal byte sequence (10 packets x 160 bytes each; per packet: "
        "1 protocol-id byte, 63 header bytes, 96 payload bytes):\n\n"
        f"{hex_str}\n\n"
        f"The possible traffic categories are: {classes_str}.\n\n"
        "Based on the byte sequence above, return a single JSON object "
        "with the following keys:\n\n"
        "- class: the predicted traffic category (must be exactly one of the listed categories).\n\n"
        "- traits: an object that objectively describes the byte-level characteristics "
        "of this flow, with exactly these keys:\n"
        "    - has_tls_record: boolean. True if TLS record header pattern is detected "
        "(content-type byte 0x14~0x17 followed by version byte 0x03xx).\n"
        "    - has_http_method: boolean. True if HTTP tokens such as GET, POST, HTTP/1.x, "
        "Host:, or User-Agent: are present in the payload.\n"
        "    - ascii_ratio_bucket: one of 'low', 'mid', 'high'. Indicates the proportion "
        "of printable ASCII bytes (0x20~0x7E) in the non-zero payload.\n"
        "    - entropy_bucket: one of 'low', 'mid', 'high'. Indicates Shannon entropy of "
        "the non-zero payload. High entropy suggests encrypted or compressed data.\n"
        "    - zero_pad_ratio_bucket: one of 'low', 'mid', 'high'. Indicates the proportion "
        "of zero bytes. High ratio means the flow is short relative to the capture window.\n\n"
        "- evidence: a list of 2~4 strings. Each string should describe a concrete "
        "byte-level observation or protocol pattern that supports the classification result. "
        "Focus on what is actually visible in the byte features, such as header patterns, "
        "payload characteristics, or entropy signatures.\n\n"
        "- description: a single paragraph of 2~3 sentences that explains what this traffic "
        "is, what protocol or application it belongs to, and what the byte features reveal "
        "about its behavior. Be specific to the classified category.\n\n"
        "- notes: a single sentence with a security-relevant observation or recommendation "
        "about this traffic category, such as whether it could be misused, what to monitor, "
        "or any anomaly indicators.\n\n"
        "Return ONLY the JSON object, no extra text."
    )


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text):
    """Parse model output into a dict. Returns (pred_dict, ok)."""
    text = THINK_RE.sub("", text).strip()
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1]), True
                    except json.JSONDecodeError:
                        break
    m = re.search(r'"class"\s*:\s*"([^"]+)"', text)
    return {"class": m.group(1) if m else None, "raw": text}, False


def stratified_sample(rows, per_class, seed):
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r)
    rng = random.Random(seed)
    picked = []
    for cls in sorted(by_cls):
        pool = by_cls[cls]
        picked.extend(pool if len(pool) <= per_class else rng.sample(pool, per_class))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--npy_root", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--samples_per_class", type=int, default=99999)
    ap.add_argument("--sample_ids_from", default=None,
                    help="Path to a previous predictions.jsonl; evaluate exactly the same sample_ids (overrides --samples_per_class)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = read_jsonl(args.test_jsonl)
    class_list = sorted({r["class"] for r in rows if r.get("class")})
    print(f"[data] {len(rows)} samples, {len(class_list)} classes")
    rows = [r for r in rows if os.path.exists(resolve_npy_path(r["sample_relpath"], args.npy_root))]
    print(f"[data] {len(rows)} samples with npy present")
    if args.sample_ids_from:
        def norm(relpath):
            return os.path.splitext(relpath)[0]
        ref_ids = {norm(rec["sample_relpath"]) for rec in read_jsonl(args.sample_ids_from)}
        rows = [r for r in rows if norm(r["sample_relpath"]) in ref_ids]
        print(f"[data] {len(rows)} samples matched to reference set "
              f"({len(ref_ids)} relpaths in {args.sample_ids_from})")
        if len(rows) < len(ref_ids):
            print(f"[warn] {len(ref_ids) - len(rows)} reference samples not found in test_jsonl/npy_root")
    else:
        rows = stratified_sample(rows, args.samples_per_class, args.seed)
        print(f"[data] {len(rows)} samples after stratified sampling")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    out_path = os.path.join(args.output_dir, "predictions.jsonl")
    n_correct = n_parsed = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(rows), args.batch_size), desc="zero-shot"):
            batch = rows[i:i + args.batch_size]
            texts = []
            for r in batch:
                hex_str = npy_to_hex(resolve_npy_path(r["sample_relpath"], args.npy_root))
                prompt = build_prompt(class_list, hex_str)
                texts.append(tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                ))
            enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            gen = out[:, enc["input_ids"].shape[1]:]
            for r, g in zip(batch, tokenizer.batch_decode(gen, skip_special_tokens=True)):
                pred, ok = extract_json(g)
                n_parsed += ok
                pred_cls = (pred.get("class") or "").strip()
                gt_cls = r["class"].strip()
                n_correct += pred_cls.lower() == gt_cls.lower()
                fout.write(json.dumps({
                    "sample_id": r["sample_id"],
                    "prediction": json.dumps(pred, ensure_ascii=False),
                    "ground_truth": r["target"],
                    "class": gt_cls,
                    "sample_relpath": r["sample_relpath"],
                }, ensure_ascii=False) + "\n")
                fout.flush()

    n = len(rows)
    summary = {
        "model": args.model_path, "mode": "zero-shot (untrained LLM, hex input)",
        "n": n, "acc": n_correct / n if n else 0.0,
        "json_parse_rate": n_parsed / n if n else 0.0,
        "samples_per_class": args.samples_per_class, "seed": args.seed,
        "sample_ids_from": args.sample_ids_from,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
