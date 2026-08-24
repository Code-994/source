#!/bin/bash
# 推理评估: MambaNetBurst + mmTraffic on CrossPlatform-iOS
# frozen VT 训练，使用 LLMclass_mGPU_v2.py（LLM 文本预测类别）
# 196 类 iOS 应用识别

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16

CHECKPOINT_PATH="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_frozen"
VT_CKPT="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_frozen/vision_tower/pytorch_model.bin"
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/nlp_output_noLLMclass_50_3000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/Crossplatform-iOS/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_frozen/eval_results"

mkdir -p "$OUTPUT_DIR"

echo ">>> [Eval] MambaNetBurst + mmTraffic on CrossPlatform-iOS (frozen VT)"
echo "    Checkpoint : $CHECKPOINT_PATH"
echo "    VT ckpt    : $VT_CKPT"
echo "    Test data  : $EVAL_DATA"
echo "    Output     : $OUTPUT_DIR"

conda run -n mambanetbust python \
    /root/autodl-tmp/mmTraffic/tinyllava/eval/eval_cls_head_qwen_sample_LLMclass_mGPU_v2.py \
    --checkpoint_path   "$CHECKPOINT_PATH" \
    --vision_tower_path "$VT_CKPT" \
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
