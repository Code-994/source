#!/bin/bash
# ==============================================================================
# ISCXVPN2016 单阶段（mambanetburst_lora）评估 —— v4 前缀强制重生成兜底
# 对比对象：双阶段 phase2 的 v4（eval-phase2-v4-regen.sh）
#
# --regen_on_fallback：LLM 未输出 class 时，用 cls_head 兜底类别作 JSON 前缀强制
# 重生成整条报告。纯 LLM 指标读 llm_class_raw（不污染）。
# 注：单阶段模型 cls_head 目录无 class_list.json → 由 --train_data_path 自动生成。
# 输出独立为 eval_results_regen/。
# ==============================================================================

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16

RUN_DIR="/root/autodl-tmp/mmTraffic/output/ISCXVPN/mambanetburst_lora"

CHECKPOINT_PATH="$RUN_DIR"
VT_CKPT="$RUN_DIR/vision_tower/pytorch_model.bin"
CLS_HEAD_PATH="$RUN_DIR/cls_head"
TRAIN_DATA="/root/autodl-tmp/mmTraffic/data/ISCXVPN2016/nlp_output_LLMclass_200_6000/train.jsonl"
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/ISCXVPN2016/nlp_output_LLMclass_200_6000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/ISCXVPN2016/ISCXVPN2016_npy_split_npy_v3_balacned_200_6000"
OUTPUT_DIR="$RUN_DIR/eval_results_regen_full"

if [ ! -f "$CHECKPOINT_PATH/adapter_model.safetensors" ] || [ ! -f "$VT_CKPT" ] || [ ! -f "$CLS_HEAD_PATH/pytorch_model.bin" ]; then
    echo "✗ 训练产物不完整：$RUN_DIR"; exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ">>> [ISCXVPN 单阶段 Eval v4] cls_head 兜底 + 前缀强制重生成报告"
echo "    Checkpoint : $CHECKPOINT_PATH"
echo "    Output     : $OUTPUT_DIR"

conda run -n mambanetbust python \
    /root/autodl-tmp/mmTraffic/tinyllava/eval/eval_cls_head_qwen_sample_LLMclass_mGPU_v4_regen.py \
    --checkpoint_path   "$CHECKPOINT_PATH" \
    --vision_tower_path "$VT_CKPT" \
    --cls_head_path     "$CLS_HEAD_PATH" \
    --train_data_path   "$TRAIN_DATA" \
    --eval_data_path    "$EVAL_DATA" \
    --image_folder      "$IMAGE_FOLDER" \
    --output_dir        "$OUTPUT_DIR" \
    --samples_per_class 2000 \
    --batch_size        4 \
    --max_new_tokens    400 \
    --temperature       0.0 \
    --conv_version      qwen3_instruct \
    --num_gpus          4 \
    --regen_on_fallback \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

if [ -f "$OUTPUT_DIR/predictions.jsonl" ]; then
    echo ">>> [文本指标] ROUGE-L / BERTScore（失败样本用重生成报告）"
    conda run -n mambanetbust python \
        /root/autodl-tmp/mmTraffic/tinyllava/eval/evaluate_predictions.py \
        --input  "$OUTPUT_DIR/predictions.jsonl" \
        --output "$OUTPUT_DIR/all_results.json" \
        --text \
        --bert-model-path /root/autodl-tmp/mmTraffic/model/robert-large \
        2>&1 | tee "$OUTPUT_DIR/text_metrics.log"
fi
