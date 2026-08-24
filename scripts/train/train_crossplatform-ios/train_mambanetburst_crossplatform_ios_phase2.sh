#!/bin/bash
# ==============================================================================
# iOS 联合训练 V2（Phase 2 修复版）—— 修复 phase2 把 phase1 80% 判别特征冲垮的问题
# ------------------------------------------------------------------------------
# 用 tinyllava/train/train_v2.py（原 train.py 不动），核心修复：
#   - 加载 phase1 训好的 cls_head（不再每次随机初始化）  → MMTRAFFIC_CLS_HEAD_INIT
#   - λ_cls 维持 0.1（与既往实验一致；如需调强可 export MMTRAFFIC_LAMBDA_CLS=0.3）
#   - 加载 phase1 的 vision_tower + connector（已适配 v3，cls_head 80%）
#
# 两个阶段用 STAGE 切换：
#   STAGE=validate （默认）冻结 encoder+connector，只训 LoRA+cls_head
#                  → 验证"phase1 的 80% 特征够好、LLM 能否从中学会分类"
#                  → 看 cls_head 是否维持 ~80%、JClsAcc 是否回到 ~70-80%
#   STAGE=full     encoder+connector 端到端微调（确认有效后再跑这一程）
#
# 用法：
#   bash train_mambanetburst_crossplatform_ios_phase2.sh            # 默认 validate
#   STAGE=full bash train_mambanetburst_crossplatform_ios_phase2.sh # 完整微调
# ==============================================================================

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 路径 ──────────────────────────────────────────────────────────────────────
DATA_PATH="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/train.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
LLM_PATH="/root/autodl-tmp/mmTraffic/model/Qwen3-1.7B"
PHASE1_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_phase1"

# ── [V2 关键] 加载 phase1 训好的 cls_head（train_v2.py 读这个环境变量）──────────
export MMTRAFFIC_CLS_HEAD_INIT="$PHASE1_DIR/cls_head"
# λ_cls 不设 → train_v2.py 用默认 0.1（与既往一致）。如需调强：
# export MMTRAFFIC_LAMBDA_CLS=0.3

# ── 阶段切换 ──────────────────────────────────────────────────────────────────
STAGE="${STAGE:-validate}"
if [ "$STAGE" = "validate" ]; then
    TUNE_VT="frozen"          # 冻结 encoder
    TUNE_CONN="frozen"        # 冻结 connector
    OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_v2_validate"
    MASTER_PORT=29611
elif [ "$STAGE" = "full" ]; then
    TUNE_VT="full"            # 端到端微调 encoder
    TUNE_CONN="full"          # 端到端微调 connector
    OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_v2_full"
    MASTER_PORT=29612
else
    echo "未知 STAGE=$STAGE（应为 validate 或 full）"; exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ">>> [iOS V2 / STAGE=$STAGE] 联合训练"
echo "    cls_head init : $MMTRAFFIC_CLS_HEAD_INIT"
echo "    lambda_cls    : ${MMTRAFFIC_LAMBDA_CLS:-0.1(默认)}"
echo "    tune VT/Conn  : $TUNE_VT / $TUNE_CONN"
echo "    Phase1 ckpt   : $PHASE1_DIR"
echo "    Output        : $OUTPUT_DIR"

/root/miniconda3/envs/mambanetbust/bin/deepspeed \
    --include localhost:0,1,2,3 \
    --master_port $MASTER_PORT \
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
    --tune_type_vision_tower "$TUNE_VT" \
    --tune_type_connector "$TUNE_CONN" \
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

# ==============================================================================
# 【原 Phase 2 内容（已注释保留备份）】
# 两阶段训练 Phase 2：联合训练（从 Phase 1 checkpoint 继续）
#   - pretrained_*_path=phase1：加载 VT + connector
#   - training_recipe=lora / tune_type_llm=lora
#   - tune_type_vision_tower=full / tune_type_connector=full
#   - 用 train.py（每次重置 cls_head、λ=0.1）——iOS 在此配置下 cls_head 崩到 ~18%
# ------------------------------------------------------------------------------
# OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora"
# mkdir -p "$OUTPUT_DIR"
# /root/miniconda3/envs/mambanetbust/bin/deepspeed \
#     --include localhost:0,1,2,3 \
#     --master_port 29604 \
#     /root/autodl-tmp/mmTraffic/tinyllava/train/train.py \
#     --deepspeed /root/autodl-tmp/mmTraffic/scripts/zero2.json \
#     --model_name_or_path "$LLM_PATH" \
#     --vision_tower mambanetburst \
#     --vision_tower2 "" \
#     --pretrained_vision_tower_path "$PHASE1_DIR/vision_tower" \
#     --pretrained_connector_path "$PHASE1_DIR/connector" \
#     --connector_type mlp2x_gelu \
#     --data_path "$DATA_PATH" \
#     --image_folder "$IMAGE_FOLDER" \
#     --is_multimodal True \
#     --conv_version qwen3_instruct \
#     --mm_vision_select_layer -2 \
#     --image_aspect_ratio square \
#     --fp16 False \
#     --bf16 True \
#     --training_recipe lora \
#     --tune_type_llm lora \
#     --tune_type_vision_tower full \
#     --tune_type_connector full \
#     --lora_r 32 \
#     --lora_alpha 64 \
#     --lora_dropout 0.1 \
#     --lora_bias none \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 30 \
#     --gradient_checkpointing True \
#     --learning_rate 5e-5 \
#     --warmup_ratio 0.1 \
#     --weight_decay 0.01 \
#     --max_grad_norm 1.0 \
#     --num_train_epochs 10 \
#     --logging_steps 10 \
#     --save_steps 2000 \
#     --model_max_length 3500 \
#     --output_dir "$OUTPUT_DIR" \
#     --report_to none \
#     2>&1 | tee "$OUTPUT_DIR/train.log"
# ==============================================================================
