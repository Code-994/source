#!/bin/bash
# ==============================================================================
# iOS phase2_direct 评估 —— 评测"phase1→直接全解冻 phase2"的输出
# 对应训练：scripts/train/train_mambanetburst_crossplatform_ios_phase2_direct.sh
#
# 该模型 = base Qwen + 全新 LoRA（适配全在 adapter，未合并）→ eval 加载 base+adapter
# 正好对，无合并权重坑，直接评估（生成+分类 → 文本指标）。
# ==============================================================================

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16

RUN_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_phase2_direct"

CHECKPOINT_PATH="$RUN_DIR"
VT_CKPT="$RUN_DIR/vision_tower/pytorch_model.bin"
CLS_HEAD_PATH="$RUN_DIR/cls_head"
TRAIN_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/train.jsonl"
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
OUTPUT_DIR="$RUN_DIR/eval_results"

if [ ! -f "$CHECKPOINT_PATH/adapter_model.safetensors" ] || [ ! -f "$VT_CKPT" ] || [ ! -f "$CLS_HEAD_PATH/pytorch_model.bin" ]; then
    echo "✗ 训练产物不完整：$RUN_DIR（可能尚未训完）"; exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ">>> [iOS phase2_direct Eval] (with cls_head fallback)"
echo "    Checkpoint  : $CHECKPOINT_PATH"
echo "    Output      : $OUTPUT_DIR"

conda run -n mambanetbust python \
    /root/autodl-tmp/mmTraffic/tinyllava/eval/eval_cls_head_qwen_sample_LLMclass_mGPU_v3.py \
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
