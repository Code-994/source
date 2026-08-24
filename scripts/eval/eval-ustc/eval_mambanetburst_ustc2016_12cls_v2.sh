#!/bin/bash
# 实验3-v2推理评估: MambaNetBurst(12cls预训练) + mmTraffic on USTC-TFC-2016
# 使用 eval_cls_head_qwen_sample_LLMclass_mGPU_v2.py（JSON 格式约束）

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16

CHECKPOINT_PATH="/root/autodl-tmp/mmTraffic/output/USTC-TFC-2016/mambanetburst_lora_12cls"
VT_CKPT="/root/autodl-tmp/mmTraffic/output/USTC-TFC-2016/mambanetburst_lora_12cls/vision_tower/pytorch_model.bin"
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/USTC-TFC-2016/nlp_output_LLMclass_3000_6000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/USTC-TFC-2016/USTC-TFC-2016_npy_v3_balacned_3000_6000"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/USTC-TFC-2016/mambanetburst_lora_12cls/eval_results"

mkdir -p "$OUTPUT_DIR"

echo ">>> [Eval3-12cls-v2] MambaNetBurst(12cls) + mmTraffic on USTC-TFC-2016"
echo "    Checkpoint : $CHECKPOINT_PATH"
echo "    VT ckpt    : $VT_CKPT"
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
    --max_new_tokens    500 \
    --temperature       0.0 \
    --conv_version      qwen3_instruct \
    --num_gpus          4 \
    2>&1 | tee "$OUTPUT_DIR/eval.log"
