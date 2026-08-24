#!/bin/bash
# ==============================================================================
# ISCX-Tor-2016 双阶段 phase2 评估 —— v4 前缀强制重生成兜底
# 对应训练：scripts/train/iscxtor/train_mambanetburst_tor2016-phase2_direct.sh
#
# 与 v3 的区别（--regen_on_fallback）：
#   LLM 未输出 class（null/缺失）时，用 cls_head 兜底类别作 JSON 前缀
#   {"class":"X", ... 强制重生成整条报告 → evidence/description 随之自洽。
#
# 指标口径（保持诚实）：
#   - class_exact_match_rate（纯 LLM）：仍用「原始 LLM 输出的 class」算，重生成不污染；
#   - class_exact_match_with_fallback：cls_head 兜底分类（不变）；
#   - 文本指标 ROUGE/BERTScore：对失败样本用「重生成后的自洽报告」算；
#   - metrics.json 多一项 regenerated_count（重生成了几条）。
# 输出独立为 eval_results_regen/，不覆盖 v3（eval_results/）。
# ==============================================================================

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16

RUN_DIR="/root/autodl-tmp/mmTraffic/output/ISCX-Tor-2016/phase2"

CHECKPOINT_PATH="$RUN_DIR"
VT_CKPT="$RUN_DIR/vision_tower/pytorch_model.bin"
CLS_HEAD_PATH="$RUN_DIR/cls_head"
TRAIN_DATA="/root/autodl-tmp/mmTraffic/data/ISCX-Tor-2016/nlp_output_LLMclass_3000_10000/train.jsonl"
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/ISCX-Tor-2016/nlp_output_LLMclass_3000_10000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/ISCX-Tor-2016/Tor_split_pcap_merged_npy_v3_balacned_3000_10000"
OUTPUT_DIR="$RUN_DIR/eval_results_regen_full"

if [ ! -f "$CHECKPOINT_PATH/adapter_model.safetensors" ] || [ ! -f "$VT_CKPT" ] || [ ! -f "$CLS_HEAD_PATH/pytorch_model.bin" ]; then
    echo "✗ 训练产物不完整：$RUN_DIR"; exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ">>> [Tor phase2 Eval v4] cls_head 兜底 + 前缀强制重生成报告"
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
