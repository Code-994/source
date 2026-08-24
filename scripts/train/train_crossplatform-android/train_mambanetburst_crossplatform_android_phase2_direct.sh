#!/bin/bash
# ==============================================================================
# Android — phase1 → phase2_direct（标准最优两 run 流程）
# ------------------------------------------------------------------------------
# 前置：phase1 已完成（mambanetburst_lora_phase1，212类）。
# 本阶段：加载 phase1 感知预热权重（encoder+connector+cls_head）后，
#         直接全解冻（encoder/connector/llm = full/full/lora）+ normal lr 5e-5，
#         一个 run 完成联合训练。用 train_v2.py。
#
# 关键：加载 phase1 cls_head（MMTRAFFIC_CLS_HEAD_INIT）是防崩的核心 trick——
#       cls_head 从 0 才是原始 phase2 崩溃的主因（详见
#       docs/mmtraffic_ios_ood_diagnosis_and_v2_fix.md §9.6）。
#       该流程已在 iOS 验证为最优且最简（JClsAcc 80.41%，两 run，非合并模型 eval 简单）。
#
# 用法：bash scripts/train/train_mambanetburst_crossplatform_android_phase2_direct.sh
# ==============================================================================

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="/root/autodl-tmp/mmTraffic/data/Crossplatform-Android/nlp_output_noLLMclass_50_2000/train.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform-Android/CrossPlatform_android_pcaps_split_npy_v3_balacned_50_2000"
LLM_PATH="/root/autodl-tmp/mmTraffic/model/Qwen3-1.7B"
PHASE1_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-Android/mambanetburst_lora_phase1"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-Android/mambanetburst_lora_phase2_direct"

# [V2 关键] 加载 phase1 感知预热的 cls_head（防崩核心）
export MMTRAFFIC_CLS_HEAD_INIT="$PHASE1_DIR/cls_head"
# λ_cls 默认 0.1

mkdir -p "$OUTPUT_DIR"

echo ">>> [Android phase2_direct] 加载 phase1 预热权重 + 直接全解冻 + normal lr 5e-5"
echo "    Phase1 ckpt  : $PHASE1_DIR"
echo "    cls_head init: $MMTRAFFIC_CLS_HEAD_INIT"
echo "    Output       : $OUTPUT_DIR"

/root/miniconda3/envs/mambanetbust/bin/deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29627 \
    /root/autodl-tmp/mmTraffic/tinyllava/train/train_v2.py \
    --deepspeed /root/autodl-tmp/mmTraffic/scripts/zero2.json \
    --model_name_or_path "$LLM_PATH" \
    --vision_tower mambanetburst \
    --vision_tower2 "" \
    --pretrained_vision_tower_path "$PHASE1_DIR/vision_tower" \
    --pretrained_connector_path "$PHASE1_DIR/connector" \
    --connector_type mlp2x_gelu \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --is_multimodal True \
    --conv_version qwen3_instruct \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --fp16 False \
    --bf16 True \
    --training_recipe lora \
    --tune_type_llm lora \
    --tune_type_vision_tower full \
    --tune_type_connector full \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_bias none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 30 \
    --gradient_checkpointing True \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --save_steps 2000 \
    --model_max_length 3500 \
    --output_dir "$OUTPUT_DIR" \
    --report_to none \
    2>&1 | tee "$OUTPUT_DIR/train.log"
