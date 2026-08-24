#!/bin/bash
# 实验3-v2: MambaNetBurst(12cls预训练) + mmTraffic on USTC-TFC-2016
# 使用 12类重训的 MambaNetBurst 权重，消除 Neris 等幽灵类干扰

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"

DATA_PATH="/root/autodl-tmp/mmTraffic/data/USTC-TFC-2016/nlp_output_LLMclass_3000_6000/train.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/USTC-TFC-2016/USTC-TFC-2016_npy_v3_balacned_3000_6000"
LLM_PATH="/root/autodl-tmp/mmTraffic/model/Qwen3-1.7B"
VT_CKPT="/root/autodl-tmp/MambaNetBurst/output/ustc2016_12cls/checkpoint-best.pth"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/USTC-TFC-2016/mambanetburst_lora_12cls"

mkdir -p "$OUTPUT_DIR"

echo ">>> [Train3-v2] MambaNetBurst(12cls) + mmTraffic on USTC-TFC-2016"
echo "    VT ckpt    : $VT_CKPT"
echo "    Output     : $OUTPUT_DIR"

/root/miniconda3/envs/mambanetbust/bin/deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29602 \
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
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 10 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --save_steps 1000 \
    --output_dir "$OUTPUT_DIR" \
    --report_to none \
    2>&1 | tee "$OUTPUT_DIR/train.log"
