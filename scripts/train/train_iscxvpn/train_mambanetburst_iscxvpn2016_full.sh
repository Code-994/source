#!/bin/bash
# 实验2: MambaNetBurst + mmTraffic on ISCX-Tor-2016 BGTD 数据集
# 感知模块: MambaNetBurst (Mamba-2, d_model=256)，替换原 NetMamba
# 输出池化: AdaptiveAvgPool1d(1600→400)
# 微调策略: LLM LoRA + 感知模块全解冻(tune_type_vision_tower=full) + Connector全量
# 硬件: 4× vGPU-48GB

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"

DATA_PATH="/root/autodl-tmp/mmTraffic/data/ISCX-Tor-2016/nlp_output_LLMclass_3000_10000/train.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/ISCX-Tor-2016/Tor_split_pcap_merged_npy_v3_balacned_3000_10000"
LLM_PATH="/root/autodl-tmp/mmTraffic/model/Qwen3-1.7B"
VT_CKPT="/root/autodl-tmp/MambaNetBurst/output/tor2016/checkpoint-best.pth"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/ISCX-Tor-2016/mambanetburst_lora"

mkdir -p "$OUTPUT_DIR"

echo ">>> [Exp2] MambaNetBurst + mmTraffic on ISCX-Tor-2016"
echo "    LLM           : $LLM_PATH"
echo "    VT checkpoint : $VT_CKPT"
echo "    Output dir    : $OUTPUT_DIR"

/root/miniconda3/envs/mambanetbust/bin/deepspeed --include localhost:0,1,2,3 --master_port 29601 \
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
