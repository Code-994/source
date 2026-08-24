#!/bin/bash
# ==============================================================================
# iOS — phase1 → 直接 phase2（不冻、normal lr，把 validate+full 合成一个 run）
# ------------------------------------------------------------------------------
# 目的：测试"只靠加载 phase1 cls_head（而非冻结 + 链式 + 小lr），能否让直接解冻的
#       单 run phase2 不崩"——若成功，就能把三 run（phase1→validate→full）减成两 run
#       （phase1→phase2）。
#
# 配置 = 单阶段训练，唯一区别是【加载 phase1 的感知预热权重】：
#   - pretrained_vision_tower_path → phase1/vision_tower（感知预热后的 encoder）
#   - pretrained_connector_path    → phase1/connector（感知预热后的 connector）
#   - MMTRAFFIC_CLS_HEAD_INIT       → phase1/cls_head（加载，非从 0）
#   - encoder/connector/llm 全部解冻（full/full/lora），normal lr 5e-5（不小）
#   - 用 train_v2.py
#
# 与当初崩溃的原始 phase2 的唯一差异：那次 cls_head 从 0（train.py），这次加载 phase1 cls_head。
#
# 用法：bash scripts/train/train_mambanetburst_crossplatform_ios_phase2_direct.sh
# ==============================================================================

# 用法：bash train-phase2.sh <epoch>      例：bash train-phase2.sh 5
#   <epoch> = 对应 phase1 的训练轮数，仅用于定位 phase1 产物与命名输出目录。
#   注意：phase2 自身训练轮数固定 10（与主实验一致），不随该参数变化。

# ── <epoch> 指向对应 phase1 的产物；phase2 自身轮数固定 ──
EPOCH="${1:?用法: bash train-phase2.sh <epoch>，例如 1 / 3 / 5 / 7 / 10（= 对应的 phase1 轮数）}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SA_ROOT="/root/autodl-tmp/mmTraffic/output/parameter_sensitivity_analysis"
DATA_PATH="/root/autodl-tmp/mmTraffic/data/Crossplatform_ios/nlp_output_noLLMclass_50_3000/train.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform_ios/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
LLM_PATH="/root/autodl-tmp/mmTraffic/model/Qwen3-1.7B"
PHASE1_DIR="$SA_ROOT/${EPOCH}epoch/phase1"
OUTPUT_DIR="$SA_ROOT/${EPOCH}epoch/phase2"

# [V2 关键] 加载 phase1 感知预热的 cls_head（与原始崩溃 phase2 的唯一差异）
export MMTRAFFIC_CLS_HEAD_INIT="$PHASE1_DIR/cls_head"
# λ_cls 默认 0.1

# ── 前置检查：phase1 三件产物必须齐全 ──
# 尤其 cls_head/pytorch_model.bin：缺失时 train_v2.py 会「静默」退回随机初始化
# （正是当初 cls_head 从 80%→18% 的崩溃根因），必须硬失败而非放行。
for f in "$PHASE1_DIR/vision_tower/pytorch_model.bin" \
         "$PHASE1_DIR/connector/pytorch_model.bin" \
         "$PHASE1_DIR/cls_head/pytorch_model.bin"; do
    if [ ! -f "$f" ]; then
        echo "✗ [Phase 2] 缺少 phase1 产物：$f"
        echo "  请先跑 train-phase1.sh $EPOCH（它会自动抽取 cls_head）。已中止。"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

echo ">>> [iOS phase2 direct] 加载 phase1 预热权重 + 直接全解冻 + normal lr 5e-5"
echo "    Phase1 ckpt  : $PHASE1_DIR"
echo "    cls_head init: $MMTRAFFIC_CLS_HEAD_INIT"
echo "    Output       : $OUTPUT_DIR"

/root/miniconda3/envs/mambanetbust/bin/deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29625 \
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
