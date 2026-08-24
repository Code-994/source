#!/bin/bash
# 推理评估: MambaNetBurst + mmTraffic on CrossPlatform-iOS (v3: cls_head 兜底)
# 评测对象: Phase 2 死掉的 checkpoint-2000 (vision_tower/connector/cls_head 由 extract_ckpt2000.py 从 DeepSpeed 状态抽出)
# 输出两套指标: class_exact_match_rate(纯LLM) 和 class_exact_match_with_fallback(LLM+cls_head兜底)

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16

CHECKPOINT_PATH="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora/checkpoint-2000"
VT_CKPT="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora/vision_tower/pytorch_model.bin"
CLS_HEAD_PATH="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora/cls_head"
TRAIN_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/train.jsonl"
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora/eval_results_v3"

mkdir -p "$OUTPUT_DIR"

echo ">>> [Eval v3] MambaNetBurst + mmTraffic on CrossPlatform-iOS (with cls_head fallback)"
echo "    Checkpoint  : $CHECKPOINT_PATH"
echo "    VT ckpt     : $VT_CKPT"
echo "    CLS head    : $CLS_HEAD_PATH"
echo "    Test data   : $EVAL_DATA"
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
