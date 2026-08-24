#!/bin/bash
# ==============================================================================
# iOS V2 full 阶段评估 —— 评测链式续训(train_..._v2_full.sh)的输出
# ------------------------------------------------------------------------------
# 对应训练：scripts/train/train_mambanetburst_crossplatform_ios_v2_full.sh
# 训练完成后 training_recipe.save() 把所有产物写到 RUN_DIR 根目录：
#   adapter_config.json / adapter_model.safetensors（LoRA）
#   language_model/  vision_tower/  connector/  cls_head/  + tokenizer + config
#
# 输出两套关键指标：
#   - class_exact_match_rate         ：纯 LLM 生成类别准确率（JClsAcc，真指标）
#   - class_exact_match_with_fallback：LLM 缺失时用 cls_head argmax 兜底后的准确率
# 重点：JClsAcc 能否突破 validate 阶段的 77.85%。
#
# 用法：
#   bash scripts/eval/eval_mambanetburst_crossplatform_ios_v2_full.sh
# ==============================================================================

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16

RUN_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_v2_full"

CHECKPOINT_PATH="$RUN_DIR"
VT_CKPT="$RUN_DIR/vision_tower/pytorch_model.bin"
CLS_HEAD_PATH="$RUN_DIR/cls_head"
TRAIN_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/train.jsonl"
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
OUTPUT_DIR="$RUN_DIR/eval_results"

# 训练产物完整性检查
if [ ! -f "$CHECKPOINT_PATH/adapter_model.safetensors" ] || [ ! -f "$VT_CKPT" ] || [ ! -f "$CLS_HEAD_PATH/pytorch_model.bin" ]; then
    echo "✗ full 训练产物不完整（可能尚未训完）。需要："
    echo "    $CHECKPOINT_PATH/adapter_model.safetensors"
    echo "    $VT_CKPT"
    echo "    $CLS_HEAD_PATH/pytorch_model.bin"
    echo "  请等 full 训练完成（screen: ios_v2_full）后再评估。"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ">>> [iOS V2 full Eval] (with cls_head fallback)"
echo "    Checkpoint  : $CHECKPOINT_PATH"
echo "    VT ckpt     : $VT_CKPT"
echo "    CLS head    : $CLS_HEAD_PATH"
echo "    Test data   : $EVAL_DATA"
echo "    Output      : $OUTPUT_DIR"

# ── ① 生成预测 + 分类指标（4 卡 LLM 生成）──────────────────────────────────────
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

# ── ② 文本指标（ROUGE-L + BERTScore）──────────────────────────────────────────
if [ -f "$OUTPUT_DIR/predictions.jsonl" ]; then
    echo ">>> [文本指标] ROUGE-L / BERTScore"
    conda run -n mambanetbust python \
        /root/autodl-tmp/mmTraffic/tinyllava/eval/evaluate_predictions.py \
        --input  "$OUTPUT_DIR/predictions.jsonl" \
        --output "$OUTPUT_DIR/all_results.json" \
        --text \
        --bert-model-path /root/autodl-tmp/mmTraffic/model/robert-large \
        2>&1 | tee "$OUTPUT_DIR/text_metrics.log"
else
    echo "✗ 未生成 predictions.jsonl，跳过文本指标"
fi
