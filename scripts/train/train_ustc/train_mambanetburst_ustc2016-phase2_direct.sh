#!/bin/bash
# ==============================================================================
# USTC-TFC-2016（12 类）— phase1 → 直接 phase2_direct（不冻、normal lr，单 run 联合训练）
# ------------------------------------------------------------------------------
# 配置 = 单阶段联合训练，唯一区别是【加载 phase1 感知预热权重】：
#   - pretrained_vision_tower_path → phase1/vision_tower
#   - pretrained_connector_path    → phase1/connector
#   - MMTRAFFIC_CLS_HEAD_INIT      → phase1/cls_head（加载，非从 0）← 防崩关键
#   - encoder/connector/llm 全部解冻（full/full/lora），normal lr 5e-5
#   - 用 train_v2.py
#
# 前置：先跑完 phase1，并确保 phase1/{vision_tower,connector,cls_head} 就绪
#      （connector_only recipe 不存 cls_head → 需从 phase1/checkpoint-*/model.safetensors 抽取）
# ==============================================================================

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="/root/autodl-tmp/mmTraffic/data/USTC-TFC-2016/nlp_output_LLMclass_3000_6000/train.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/USTC-TFC-2016/USTC-TFC-2016_npy_v3_balacned_3000_6000"
LLM_PATH="/root/autodl-tmp/mmTraffic/model/Qwen3-1.7B"
PHASE1_DIR="/root/autodl-tmp/mmTraffic/output/USTC-TFC-2016/phase1"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/USTC-TFC-2016/phase2"

export MMTRAFFIC_CLS_HEAD_INIT="$PHASE1_DIR/cls_head"

mkdir -p "$OUTPUT_DIR"

for sub in vision_tower connector cls_head; do
    if [ ! -e "$PHASE1_DIR/$sub" ]; then
        echo "!!! 缺少 $PHASE1_DIR/$sub —— 请先跑完 phase1 并抽取 cls_head 再执行本脚本"
        exit 1
    fi
done

echo ">>> [USTC phase2] 加载 phase1 预热权重 + 直接全解冻 + normal lr 5e-5"
echo "    Phase1 ckpt  : $PHASE1_DIR"
echo "    cls_head init: $MMTRAFFIC_CLS_HEAD_INIT"
echo "    Output       : $OUTPUT_DIR"

/root/miniconda3/envs/mambanetbust/bin/deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29633 \
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
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 15 \
    --gradient_checkpointing True \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --save_steps 5000 \
    --model_max_length 3500 \
    --output_dir "$OUTPUT_DIR" \
    --report_to none \
    2>&1 | tee "$OUTPUT_DIR/train.log"
