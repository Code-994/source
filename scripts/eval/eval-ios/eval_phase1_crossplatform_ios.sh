#!/bin/bash
# 评估 Phase1 cls_head 分类精度（不启动 LLM 推理，仅 vision_tower → cls_head）
# 权重从 checkpoint-2450 抽出（运行过 extract_phase1_ckpt2450.py 后生效）

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16

CHECKPOINT_PATH="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_phase1"
VT_CKPT="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_phase1/vision_tower/pytorch_model.bin"
CLS_HEAD_PATH="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_phase1/cls_head"
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_phase1/eval_results"

mkdir -p "$OUTPUT_DIR"

echo ">>> [Phase1 Eval] cls_head-only 分类 on CrossPlatform-iOS (196 类)"
echo "    Checkpoint : $CHECKPOINT_PATH"
echo "    VT ckpt    : $VT_CKPT"
echo "    CLS head   : $CLS_HEAD_PATH"
echo "    Test data  : $EVAL_DATA"
echo "    Output     : $OUTPUT_DIR"

conda run -n mambanetbust python \
    /root/autodl-tmp/mmTraffic/tinyllava/eval/eval_cls_head_qwen_sample_LLMclass_mGPU_v2.py \
    --checkpoint_path   "$CHECKPOINT_PATH" \
    --vision_tower_path "$VT_CKPT" \
    --cls_head_path     "$CLS_HEAD_PATH" \
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
