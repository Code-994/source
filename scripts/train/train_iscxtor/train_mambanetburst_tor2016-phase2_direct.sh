#!/bin/bash
# ==============================================================================
# ISCX-Tor-2016（8 类）— phase1 → 直接 phase2_direct（不冻、normal lr，单 run 联合训练）
# ------------------------------------------------------------------------------
# 流程：phase1（感知预热，冻结 LLM + 纯 cls_loss）→ 本脚本（全解冻联合微调）。
#
# 配置 = 单阶段联合训练，唯一区别是【加载 phase1 的感知预热权重】：
#   - pretrained_vision_tower_path → phase1/vision_tower
#   - pretrained_connector_path    → phase1/connector
#   - MMTRAFFIC_CLS_HEAD_INIT      → phase1/cls_head（加载，非从 0）← 防崩关键
#   - encoder/connector/llm 全部解冻（full/full/lora），normal lr 5e-5
#   - 用 train_v2.py（支持从 MMTRAFFIC_CLS_HEAD_INIT 加载 cls_head）
#
# 用法：bash scripts/train/iscxtor/train_mambanetburst_tor2016-phase2_direct.sh
# 前置：先跑完 train_mambanetburst_tor2016-phase1.sh（产出 phase1/{vision_tower,connector,cls_head}）
# ==============================================================================

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="/root/autodl-tmp/mmTraffic/data/ISCX-Tor-2016/nlp_output_LLMclass_3000_10000/train.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/ISCX-Tor-2016/Tor_split_pcap_merged_npy_v3_balacned_3000_10000"
LLM_PATH="/root/autodl-tmp/mmTraffic/model/Qwen3-1.7B"
PHASE1_DIR="/root/autodl-tmp/mmTraffic/output/ISCX-Tor-2016/phase1"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/ISCX-Tor-2016/phase2"

# [V2 关键] 加载 phase1 感知预热的 cls_head（与原始崩溃 phase2 的唯一差异）
export MMTRAFFIC_CLS_HEAD_INIT="$PHASE1_DIR/cls_head"
# λ_cls 默认 0.1（可用 MMTRAFFIC_LAMBDA_CLS 覆盖）

mkdir -p "$OUTPUT_DIR"

# 前置检查：phase1 权重是否就绪
for sub in vision_tower connector cls_head; do
    if [ ! -e "$PHASE1_DIR/$sub" ]; then
        echo "!!! 缺少 $PHASE1_DIR/$sub —— 请先跑完 phase1 再执行本脚本"
        exit 1
    fi
done

echo ">>> [Tor phase2] 加载 phase1 预热权重 + 直接全解冻 + normal lr 5e-5"
echo "    Phase1 ckpt  : $PHASE1_DIR"
echo "    cls_head init: $MMTRAFFIC_CLS_HEAD_INIT"
echo "    Data         : $DATA_PATH"
echo "    Output       : $OUTPUT_DIR"

/root/miniconda3/envs/mambanetbust/bin/deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29631 \
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
    --save_steps 6000 \
    --model_max_length 3500 \
    --output_dir "$OUTPUT_DIR" \
    --report_to none \
    2>&1 | tee "$OUTPUT_DIR/train.log"
