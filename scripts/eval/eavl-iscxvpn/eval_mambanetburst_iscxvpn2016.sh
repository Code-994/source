#!/bin/bash
# 实验1推理评估: MambaNetBurst + mmTraffic on ISCXVPN2016 BGTD
# 对应训练: train_mambanetburst_iscxvpn2016_full.sh
# 参数与原始 mmTraffic 推理保持一致（samples_per_class=20, batch_size=4, max_new_tokens=400）

export PYTHONPATH="/root/autodl-tmp/mmTraffic:/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"
export TORCH_DTYPE=bfloat16

CHECKPOINT_PATH="/root/autodl-tmp/mmTraffic/output/ISCXVPN2016/mambanetburst_lora"
# VT_CKPT="/root/autodl-tmp/MambaNetBurst/output/iscxvpn2016/checkpoint-best.pth"  # 预训练权重（训练前）
VT_CKPT="/root/autodl-tmp/mmTraffic/output/ISCXVPN2016/mambanetburst_lora/vision_tower/pytorch_model.bin"  # 微调后权重
EVAL_DATA="/root/autodl-tmp/mmTraffic/data/ISCXVPN2016/nlp_output_LLMclass_200_6000/test.jsonl"
IMAGE_FOLDER="/root/autodl-tmp/mmTraffic/data/ISCXVPN2016/ISCXVPN2016_npy_split_npy_v3_balacned_200_6000"
OUTPUT_DIR="/root/autodl-tmp/mmTraffic/output/ISCXVPN2016/mambanetburst_lora/eval_results"

mkdir -p "$OUTPUT_DIR"

echo ">>> [Eval] MambaNetBurst + mmTraffic on ISCXVPN2016"
echo "    Checkpoint : $CHECKPOINT_PATH"
echo "    VT ckpt    : $VT_CKPT"
echo "    Test data  : $EVAL_DATA"
echo "    Output     : $OUTPUT_DIR"

conda run -n mambanetbust python \
    /root/autodl-tmp/mmTraffic/tinyllava/eval/eval_cls_head_qwen_sample_LLMclass_mGPU.py \
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
