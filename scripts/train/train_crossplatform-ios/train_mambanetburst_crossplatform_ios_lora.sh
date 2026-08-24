#!/bin/bash
# ==============================================================================
# iOS 单阶段联合训练（路线 1）—— 用 v3-native encoder，复刻 Android 成功配方
# ------------------------------------------------------------------------------
# 背景：iOS 之前单阶段失败，是因为 encoder（训在 dataset_sampled 5×320）喂 v3 是 OOD
#       （可分性仅 55%），单阶段抬不动。现已在 v3 上重训出 v3-native encoder：
#       test acc 91.79%（远超 55% OOD、逼近 dataset_sampled 94.11%）。
#
# 本脚本：用这个 92% 的 v3-native encoder 做干净单阶段联合训练（无 phase1/validate/full）：
#   - pretrained_vision_tower_path → crossplatform-ios-v3/checkpoint-best.pth（v3-native, 92%）
#   - 全新 connector + cls_head 从 0 + LoRA，一起联合（= Android 配方）
#   - encoder 起点 92% > Android 的 70% → 单阶段预期 ≥ 两阶段的 79.77%
#   - 用原始 train.py（已含 class_list 修复），非 train_v2
#
# 用法：bash scripts/train/train_mambanetburst_crossplatform_ios_singlestage.sh
# ==============================================================================

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/train.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
LLM_PATH="/root/autodl-tmp/mmTraffic/model/Qwen3-1.7B"
# ★ v3-native encoder（在 v3 上重训得到 91.79%），替代旧的 OOD standalone
VT_CKPT="/root/autodl-tmp/MambaNetBurst/output/crossplatform-ios-v3/checkpoint-best.pth"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetbust_lora_full"

mkdir -p "$OUTPUT_DIR"

echo ">>> [iOS 单阶段联合训练 / 路线1] v3-native encoder"
echo "    VT ckpt (v3-native): $VT_CKPT"
echo "    Data               : $DATA_PATH"
echo "    Output             : $OUTPUT_DIR"

/root/miniconda3/envs/mambanetbust/bin/deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29615 \
    /root/autodl-tmp/mmTraffic/tinyllava/train/train.py \
    --deepspeed /root/autodl-tmp/mmTraffic/scripts/zero2.json \
    --model_name_or_path "$LLM_PATH" \
    --vision_tower mambanetburst \
    --vision_tower2 "" \
    --pretrained_vision_tower_path "$VT_CKPT" \
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
