#!/bin/bash
# ==============================================================================
# iOS V2 评估 —— 评测 train_v2.py 的联合训练输出（含 cls_head 兜底）
# ------------------------------------------------------------------------------
# 与训练脚本 train_mambanetburst_crossplatform_ios_phase2.sh 对应，用 STAGE 切换：
#   STAGE=validate （默认）评测 mambanetburst_lora_v2_validate（冻结感知层那一程）
#   STAGE=full     评测 mambanetburst_lora_v2_full（端到端微调那一程）
#
# 输出两套指标：
#   - class_exact_match_rate         ：纯 LLM 生成类别的准确率（JClsAcc，真指标）
#   - class_exact_match_with_fallback：LLM 缺失时用 cls_head argmax 兜底后的准确率
# 重点看 JClsAcc 是否从之前的 ~22% 回升到 ~70-80%。
#
# 用法：
#   bash eval_mambanetburst_crossplatform_ios_v2.sh             # 默认 validate
#   STAGE=full bash eval_mambanetburst_crossplatform_ios_v2.sh  # 评测 full
# ==============================================================================

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16

# ── 阶段切换（与训练脚本输出目录对应）─────────────────────────────────────────
STAGE="${STAGE:-validate}"
if [ "$STAGE" = "validate" ]; then
    RUN_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_v2_validate"
elif [ "$STAGE" = "full" ]; then
    RUN_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_v2_full"
else
    echo "未知 STAGE=$STAGE（应为 validate 或 full）"; exit 1
fi

# 训练完成后 training_recipe.save() 把所有产物写到 RUN_DIR 根目录：
#   adapter_config.json / adapter_model.safetensors（LoRA）
#   language_model/  vision_tower/  connector/  cls_head/  + tokenizer + config
CHECKPOINT_PATH="$RUN_DIR"
VT_CKPT="$RUN_DIR/vision_tower/pytorch_model.bin"
CLS_HEAD_PATH="$RUN_DIR/cls_head"
TRAIN_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/train.jsonl"
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
OUTPUT_DIR="$RUN_DIR/eval_results"

if [ ! -d "$RUN_DIR" ]; then
    echo "✗ 训练输出不存在：$RUN_DIR"
    echo "  请先运行：STAGE=$STAGE bash scripts/train/train_mambanetburst_crossplatform_ios_phase2.sh"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ">>> [iOS V2 Eval / STAGE=$STAGE] (with cls_head fallback)"
echo "    Checkpoint  : $CHECKPOINT_PATH"
echo "    VT ckpt     : $VT_CKPT"
echo "    CLS head    : $CLS_HEAD_PATH"
echo "    Test data   : $EVAL_DATA"
echo "    Output      : $OUTPUT_DIR"

conda run -n mambanetbust python \
    /root/autodl-tmp/mmTraffic/tinyllava/eval/eval_cls_head_qwen_sample_LLMclass_mGPU_v3.py \
    --checkpoint_path   "$CHECKPOINT_PATH" \
    --vision_tower_path "$VT_CKPT" \
    --cls_head_path     "$CLS_HEAD_PATH" \
    --train_data_path   "$TRAIN_DATA" \
    --eval_data_path    "$EVAL_DATA" \
    --image_folder      "$IMAGE_FOLDER" \
    --output_dir        "$OUTPUT_DIR" \
    --samples_per_class 20 \
    --batch_size        4 \
    --max_new_tokens    400 \
    --temperature       0.0 \
    --conv_version      qwen3_instruct \
    --num_gpus          4 \
    2>&1 | tee "$OUTPUT_DIR/eval.log"
