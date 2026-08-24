#!/bin/bash
# ==============================================================================
# iOS V2 — full 阶段：从 validate 输出"链式续训" + 小学习率
# ------------------------------------------------------------------------------
# 前置：已跑完 validate（mambanetburst_lora_v2_validate，JClsAcc 77.85%）。
#
# 本阶段目标：解冻 encoder+connector，温和精修，博取 77.85% → 更高的提升空间，
#            同时避免重蹈原始 phase2"被 gen_loss 猛拽崩溃"的覆辙。
#
# 三个关键设计（对应之前讨论）：
#   ① 链式续训：--pretrained_model_path 指向 validate 输出。
#      validate 目录名含 'lora' + 有 adapter_config.json → base.py 走 LoRA 分支：
#      加载 base LLM → 合并 validate 的 LoRA（"会读特征的大脑"）→ 派生加载 VT/connector，
#      再叠一层全新 LoRA 继续训。这样解冻 encoder 时，下游 LLM 已经会读特征。
#   ② cls_head 接 validate 的（已与特征对齐），解冻时第一步就能提供有意义的 cls 梯度护住特征。
#   ③ 小学习率 1e-5（validate 用的是 5e-5）：温和精修 encoder，不猛拽。
#      注：框架的分组 lr 只支持 connector（mm_projector_lr），没有单独的 vision_tower_lr，
#         所以这里用"整体小 lr"。从 validate 续训、LLM 已合并好，整体慢学即足够温和。
#
# 用法：
#   bash scripts/train/train_mambanetburst_crossplatform_ios_v2_full.sh
# ==============================================================================

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 路径 ──────────────────────────────────────────────────────────────────────
DATA_PATH="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/train.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
LLM_PATH="/root/autodl-tmp/mmTraffic/model/Qwen3-1.7B"
VALIDATE_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_v2_validate"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_v2_full"

# ── [V2 关键] cls_head 接 validate（不再回到 phase1 / 不从 0）──────────────────
export MMTRAFFIC_CLS_HEAD_INIT="$VALIDATE_DIR/cls_head"
# λ_cls 不设 → 默认 0.1（与既往一致）

# 小学习率（validate 是 5e-5；full 温和精修用 1e-5）
LR=1e-5

if [ ! -d "$VALIDATE_DIR" ]; then
    echo "✗ validate 输出不存在：$VALIDATE_DIR"
    echo "  请先跑：STAGE=validate bash scripts/train/train_mambanetburst_crossplatform_ios_phase2.sh"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ">>> [iOS V2 / full 链式续训] 解冻 encoder+connector + 小学习率"
echo "    续训来源     : $VALIDATE_DIR (合并其 LoRA-LLM)"
echo "    cls_head init: $MMTRAFFIC_CLS_HEAD_INIT"
echo "    learning_rate: $LR"
echo "    tune VT/Conn : full / full"
echo "    Output       : $OUTPUT_DIR"

/root/miniconda3/envs/mambanetbust/bin/deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29613 \
    /root/autodl-tmp/mmTraffic/tinyllava/train/train_v2.py \
    --deepspeed /root/autodl-tmp/mmTraffic/scripts/zero2.json \
    --model_name_or_path "$LLM_PATH" \
    --vision_tower mambanetburst \
    --vision_tower2 "" \
    --pretrained_model_path "$VALIDATE_DIR" \
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
    --learning_rate "$LR" \
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
