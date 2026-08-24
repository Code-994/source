#!/bin/bash
# 单卡评测 CrossPlatform-Android best checkpoint 

export PYTHONPATH="/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"

CKPT="/root/autodl-tmp/MambaNetBurst/output/crossplatform_android/checkpoint-best.pth"
DATA_PATH="/root/autodl-tmp/MambaNetBurst/data/Crossplatform-Android/dataset_sampled"
OUTPUT_DIR="/root/autodl-tmp/MambaNetBurst/output/crossplatform_android"

echo ">>> [MambaNetBurst] CrossPlatform-Android 212类 test 评估"
echo "    Checkpoint : $CKPT"
echo "    Data       : $DATA_PATH"

conda run -n mambanetbust python \
    /root/autodl-tmp/MambaNetBurst/src/eval_test.py \
    --data_path "$DATA_PATH" \
    --nb_classes 212 \
    --resume "$CKPT" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/eval_test.log"
