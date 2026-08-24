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
#   - num_train_epochs：由脚本参数 <epoch> 指定（敏感性实验唯一自变量）
#   - 输出到 <epoch>epoch/phase1，供 Phase 2 加载
#
# 用法：bash train-phase1.sh <epoch>      例：bash train-phase1.sh 5
#   跑完自动从 checkpoint 抽取 cls_head → phase1/cls_head/（phase2 依赖）

# ── 参数敏感性实验自变量：phase1 训练轮数 ──
EPOCH="${1:?用法: bash train-phase1.sh <epoch>，例如 1 / 3 / 5 / 7 / 10}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SA_ROOT="/root/autodl-tmp/mmTraffic/output/parameter_sensitivity_analysis"
DATA_PATH="/root/autodl-tmp/mmTraffic/data/Crossplatform_ios/nlp_output_noLLMclass_50_3000/train.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform_ios/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
LLM_PATH="/root/autodl-tmp/mmTraffic/model/Qwen3-1.7B"
VT_CKPT="/root/autodl-tmp/MambaNetBurst/output/crossplatform_ios/checkpoint-best.pth"
OUTPUT_DIR="$SA_ROOT/${EPOCH}epoch/phase1"

mkdir -p "$OUTPUT_DIR"

echo ">>> [Phase 1] Warmup connector + VT (LLM frozen)"
echo "    Data       : $DATA_PATH"
echo "    VT ckpt    : $VT_CKPT"
echo "    Output     : $OUTPUT_DIR"

/root/miniconda3/envs/mambanetbust/bin/deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29604 \
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
    --num_train_epochs "$EPOCH" \
    --logging_steps 10 \
    --save_steps 5000 \
    --model_max_length 3500 \
    --output_dir "$OUTPUT_DIR" \
    --report_to none \
    2>&1 | tee "$OUTPUT_DIR/train.log"

TRAIN_RC=${PIPESTATUS[0]}
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "✗ [Phase 1] 训练异常退出 (rc=$TRAIN_RC)，跳过 cls_head 抽取"; exit "$TRAIN_RC"
fi

# ── connector_only recipe 不导出独立 cls_head，从 checkpoint 抽出供 phase2 加载 ──
echo ">>> [Phase 1] 抽取 cls_head → $OUTPUT_DIR/cls_head/"
/root/miniconda3/envs/mambanetbust/bin/python \
    /root/autodl-tmp/mmTraffic/scripts/parameters_sensitivity_analysis/extract_cls_head.py \
    "$OUTPUT_DIR" "$DATA_PATH"
