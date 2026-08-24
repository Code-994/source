#!/bin/bash
# 两阶段训练 Phase 1：预热 connector + 视觉塔
#
# 目标：冻结 LLM，只训练 connector + vision_tower + cls_head
#       让 connector 在 cls_loss 引导下建立类别区分映射，
#       为 Phase 2 的联合训练提供更好的初始化。
#
# 关键设置：
#   - training_recipe=connector_only：不挂 LoRA，避免 lora_recipe 因 targets 为空报错
#   - tune_type_llm=frozen：LLM 权重不更新
#   - tune_type_vision_tower=full：视觉塔可训练
#   - tune_type_connector=full：connector 可训练
#   - num_train_epochs=2：快速预热，不需要太多轮
#   - 输出到 mambanetburst_lora_phase1，供 Phase 2 加载

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="/root/autodl-tmp/mmTraffic/data/ISCXVPN2016/nlp_output_LLMclass_200_6000/train.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/ISCXVPN2016/ISCXVPN2016_npy_split_npy_v3_balacned_200_6000"
LLM_PATH="/root/autodl-tmp/mmTraffic/model/Qwen3-1.7B"
VT_CKPT="/root/autodl-tmp/MambaNetBurst/output/iscxvpn2016/checkpoint-best.pth"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/ISCXVPN/phase1"

mkdir -p "$OUTPUT_DIR"

echo ">>> [Phase 1] Warmup connector + VT (LLM frozen)"
echo "    Data       : $DATA_PATH"
echo "    VT ckpt    : $VT_CKPT"
echo "    Output     : $OUTPUT_DIR"

/root/miniconda3/envs/mambanetbust/bin/deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29628 \
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
    --training_recipe connector_only \
    --tune_type_llm frozen \
    --tune_type_vision_tower full \
    --tune_type_connector full \
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
