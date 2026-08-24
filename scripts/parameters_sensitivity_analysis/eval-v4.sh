#!/bin/bash
# ==============================================================================
# CrossPlatform-iOS 双阶段 phase2_direct 评估 —— v4 前缀强制重生成兜底
# 对应训练：scripts/train/crossplatform-ios/train_mambanetburst_crossplatform_ios_phase2_direct.sh
#
# 与 v3 的区别（--regen_on_fallback）：LLM 未输出 class 时，用 cls_head 兜底类别作
# JSON 前缀强制重生成整条报告 → evidence/description 随之自洽。
# 纯 LLM 指标读 llm_class_raw（不被污染）；文本指标对失败样本用重生成报告。
# 输出独立为 eval_results_regen/。
# ==============================================================================

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16

# 用法：bash eval-v4.sh <epoch>      例：bash eval-v4.sh 5
#   <epoch> = 对应 phase1 的训练轮数，用于定位 <epoch>epoch/phase2 的产物。
EPOCH="${1:?用法: bash eval-v4.sh <epoch>，例如 1 / 3 / 5 / 7 / 10（= 对应的 phase1 轮数）}"

RUN_DIR="/root/autodl-tmp/mmTraffic/output/parameter_sensitivity_analysis/${EPOCH}epoch/phase2"

CHECKPOINT_PATH="$RUN_DIR"
VT_CKPT="$RUN_DIR/vision_tower/pytorch_model.bin"
CLS_HEAD_PATH="$RUN_DIR/cls_head"
TRAIN_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform_ios/nlp_output_noLLMclass_50_3000/train.jsonl"
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform_ios/nlp_output_noLLMclass_50_3000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform_ios/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
OUTPUT_DIR="$RUN_DIR/eval_results_regen"

if [ ! -f "$CHECKPOINT_PATH/adapter_model.safetensors" ] || [ ! -f "$VT_CKPT" ] || [ ! -f "$CLS_HEAD_PATH/pytorch_model.bin" ]; then
    echo "✗ 训练产物不完整：$RUN_DIR"; exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ">>> [iOS phase2 Eval v4] cls_head 兜底 + 前缀强制重生成报告"
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
    --samples_per_class 20 \
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
