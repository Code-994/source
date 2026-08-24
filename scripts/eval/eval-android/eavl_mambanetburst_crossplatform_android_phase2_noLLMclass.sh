#!/bin/bash
# ==============================================================================
# Android phase2_direct 评估 —— 评测"phase1→直接全解冻 phase2"的输出
# 对应训练：scripts/train/train_mambanetburst_crossplatform_android_phase2_direct.sh
#
# 该模型 = base Qwen + 全新 LoRA（适配全在 adapter，未合并）→ eval 加载 base+adapter
# 正好对，无合并权重坑，直接评估（生成+分类 → 文本指标）。
# ==============================================================================

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16

RUN_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-Android/mambanetburst_lora_phase2_direct"

CHECKPOINT_PATH="$RUN_DIR"
VT_CKPT="$RUN_DIR/vision_tower/pytorch_model.bin"
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform-Android/nlp_output_noLLMclass_50_2000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform-Android/CrossPlatform_android_pcaps_split_npy_v3_balacned_50_2000"
OUTPUT_DIR="$RUN_DIR/eval_results-2"


mkdir -p "$OUTPUT_DIR"

echo ">>> [Android phase2_direct Eval] (with cls_head fallback)"
echo "    Checkpoint  : $CHECKPOINT_PATH"
echo "    Output      : $OUTPUT_DIR"

conda run -n mambanetbust python \
    /root/autodl-tmp/mmTraffic/tinyllava/eval/eval_cls_head_qwen_sample_no_LLMclass_mGPU_v2.py \
    --checkpoint_path   "$CHECKPOINT_PATH" \
    --vision_tower_path "$VT_CKPT" \
    --eval_data_path    "$EVAL_DATA" \
    --image_folder      "$IMAGE_FOLDER" \
    --output_dir        "$OUTPUT_DIR" \
    --samples_per_class 20 \
    --batch_size        4 \
    --max_new_tokens    400 \
    --temperature       0.0 \
    --conv_version      qwen3_instruct \
    --num_gpus          4 \
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
