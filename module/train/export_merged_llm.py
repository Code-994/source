"""
导出链式(full)模型的合并 LLM 权重，供 eval 正确加载。

背景：链式 full 模型的 language_model/pytorch_model.bin 是 base+validate-LoRA 合并后的权重，
      但 key 带 PEFT 的 `.base_layer.` 前缀；且 eval 脚本会因 config.llm_model_name_or_path
      指向 base Qwen（路径存在）而跳过这份合并权重、错误加载 base → 生成乱码。
      （详见 docs/mmtraffic_ios_ood_diagnosis_and_v2_fix.md §9.3）

本脚本：把合并权重清洗(strip .base_layer.)并存成 safetensors 到 <RUN_DIR>/Qwen3-1.7B-merged/，
        config 用训练保存的 text_config（vocab 已含 <image> token），tokenizer 从 base 拷贝。
        之后把 <RUN_DIR>/config.json 的 llm_model_name_or_path 指向该合并目录，eval 即正确。

用法：
    python export_merged_llm.py <RUN_DIR> [BASE_LLM]
"""
import os, sys, json, shutil, torch
from safetensors.torch import save_file

RUN_DIR = sys.argv[1].rstrip("/")
BASE_LLM = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/mmTraffic/model/Qwen3-1.7B"
OUT = os.path.join(RUN_DIR, "Qwen3-1.7B-merged")  # 目录名含 qwen3 → LLMFactory 子串匹配

lm_bin = os.path.join(RUN_DIR, "language_model", "pytorch_model.bin")
lm_cfg = os.path.join(RUN_DIR, "language_model", "config.json")
assert os.path.exists(lm_bin), f"找不到 {lm_bin}"
assert os.path.exists(lm_cfg), f"找不到 {lm_cfg}（训练保存的 text_config）"

print(f"[merge] 读取 {lm_bin}")
sd = torch.load(lm_bin, map_location="cpu")
n_base = sum("base_layer" in k for k in sd)
# 清洗 PEFT 的 .base_layer. 前缀 → 标准 Qwen3 key
new = {}
for k, v in sd.items():
    nk = k.replace(".base_layer.", ".")
    new[nk] = v.contiguous()
print(f"[merge] 共 {len(sd)} key，其中 {n_base} 个含 .base_layer.（已清洗）")

os.makedirs(OUT, exist_ok=True)
save_file(new, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
print(f"[merge] 已写 {OUT}/model.safetensors")

# config 用训练保存的（vocab 含 <image>，与权重一致），而非 base Qwen config
shutil.copy(lm_cfg, os.path.join(OUT, "config.json"))
with open(os.path.join(OUT, "config.json")) as f:
    print(f"[merge] config vocab_size = {json.load(f).get('vocab_size')}")

# tokenizer / generation_config 从 base 拷贝（eval 加载 LLM 本体不需 tokenizer，但备齐更稳）
for fn in ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
    src = os.path.join(BASE_LLM, fn)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, fn))

# 把 RUN_DIR/config.json 的 llm_model_name_or_path 指向合并目录（备份 .orig）
run_cfg = os.path.join(RUN_DIR, "config.json")
orig = run_cfg + ".orig"
if not os.path.exists(orig):
    shutil.copy(run_cfg, orig)
with open(run_cfg) as f:
    cfg = json.load(f)
cfg["llm_model_name_or_path"] = OUT
with open(run_cfg, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"[merge] 已将 {run_cfg} 的 llm_model_name_or_path 指向合并目录（备份 config.json.orig）")
print(f"[merge] 完成 → {OUT}")
