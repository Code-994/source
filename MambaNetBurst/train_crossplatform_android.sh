#!/bin/bash
# MambaNetBurst 在 CrossPlatform-Android 上训练（4卡 DDP）
# 212 类，38673 样本，if_augment=False

export PYTHONPATH="/root/autodl-tmp/MambaNetBurst/src:${PYTHONPATH}"

DATA_PATH="/root/autodl-tmp/MambaNetBurst/data/Crossplatform-Android/dataset_sampled"
OUTPUT_DIR="/root/autodl-tmp/MambaNetBurst/output/crossplatform_android"

mkdir -p "$OUTPUT_DIR"

echo ">>> [MambaNetBurst] CrossPlatform-Android 212类训练（4卡 DDP）"
echo "    Data   : $DATA_PATH"
echo "    Output : $OUTPUT_DIR"

conda run -n mambanetbust torchrun --nproc_per_node=4 \
    /root/autodl-tmp/MambaNetBurst/src/train.py \
    --data_path "$DATA_PATH" \
    --nb_classes 212 \
    --num_workers 0 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
