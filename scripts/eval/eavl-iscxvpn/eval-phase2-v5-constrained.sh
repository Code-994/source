#!/bin/bash
# ==============================================================================
# ISCXVPN2016 双阶段 phase2 评估 —— v5 约束解码（在 v4 基础上）
# 对应训练：scripts/train/iscxvpn/train_mambanetburst_iscxvpn2016-phase2_direct.sh
#
# v5 = v4（cls_head 兜底 + 前缀强制重生成）+ --constrained（约束解码）：
#   生成时用 lm-format-enforcer 强制输出符合报告 schema 的合法 JSON，
#   class 字段约束为有效类别枚举 → 从源头杜绝格式错误（引号/多值/跑偏等）。
#   预期：格式失败→~0，纯 LLM 分类与文本 ROUGE/BERTScore 同时提升。
#   --regen_on_fallback 仍保留作安全网（约束下若因 max_new_tokens 截断致不完整时兜底）。
# 输出独立为 eval_results_constrained/。
# ==============================================================================

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16
export OMP_NUM_THREADS=4

RUN_DIR="/root/autodl-tmp/mmTraffic/output/ISCXVPN/phase2"

CHECKPOINT_PATH="$RUN_DIR"
VT_CKPT="$RUN_DIR/vision_tower/pytorch_model.bin"
CLS_HEAD_PATH="$RUN_DIR/cls_head"
TRAIN_DATA="/root/autodl-tmp/mmTraffic/data/ISCXVPN2016/nlp_output_LLMclass_200_6000/train.jsonl"
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/ISCXVPN2016/nlp_output_LLMclass_200_6000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/ISCXVPN2016/ISCXVPN2016_npy_split_npy_v3_balacned_200_6000"
OUTPUT_DIR="$RUN_DIR/eval_results_constrained"

if [ ! -f "$CHECKPOINT_PATH/adapter_model.safetensors" ] || [ ! -f "$VT_CKPT" ] || [ ! -f "$CLS_HEAD_PATH/pytorch_model.bin" ]; then
    echo "✗ 训练产物不完整：$RUN_DIR"; exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ">>> [ISCXVPN phase2 Eval v5] 约束解码(强制合法JSON) + cls_head兜底 + 重生成安全网"
echo "    Checkpoint : $CHECKPOINT_PATH"
echo "    Output     : $OUTPUT_DIR"

conda run -n mambanetbust python \
    /root/autodl-tmp/mmTraffic/tinyllava/eval/eval_cls_head_qwen_sample_LLMclass_mGPU_v5_constrained.py \
    --checkpoint_path   "$CHECKPOINT_PATH" \
    --vision_tower_path "$VT_CKPT" \
    --cls_head_path     "$CLS_HEAD_PATH" \
    --train_data_path   "$TRAIN_DATA" \
    --eval_data_path    "$EVAL_DATA" \
    --image_folder      "$IMAGE_FOLDER" \
    --output_dir        "$OUTPUT_DIR" \
    --samples_per_class 20 \
    --batch_size        4 \
    --max_new_tokens    400 \
    --temperature       0.0 \
    --conv_version      qwen3_instruct \
    --num_gpus          4 \
    --constrained \
    --regen_on_fallback \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

if [ -f "$OUTPUT_DIR/predictions.jsonl" ]; then
    echo ">>> [文本指标] ROUGE-L / BERTScore"
    conda run -n mambanetbust python \
        /root/autodl-tmp/mmTraffic/tinyllava/eval/evaluate_predictions.py \
        --input  "$OUTPUT_DIR/predictions.jsonl" \
        --output "$OUTPUT_DIR/all_results.json" \
        --text \
        --bert-model-path /root/autodl-tmp/mmTraffic/model/robert-large \
        2>&1 | tee "$OUTPUT_DIR/text_metrics.log"
fi
