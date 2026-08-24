#!/bin/bash
# ==============================================================================
# LLM zero-shot 基线评估（论文主表基线之一）
#
# 设计：用原始 Qwen3-1.7B（不加载任何训练产物），输入 v3 字节矩阵的十六进制
# 文本 + 类别候选 + 与 mmTraffic 相同的 JSON 报告 schema，直接生成结构化报告。
# 证明：没有感知模块与对齐训练，同一个 LLM 做不了这个任务（分类≈随机、证据幻觉）。
#
# 采样协议：--sample_ids_from 复用各数据集双阶段评估（20/类）的同一批 sample_id，
# 保证与主表其他系统严格同集对比。
#
# 产物：output/<dataset>/zeroshot_qwen3_1.7b/
#   - predictions.jsonl  （与 evaluate_predictions.py 兼容）
#   - summary.json       （acc / json_parse_rate / 采样信息）
#   - results.json       （evaluate_predictions.py 全套指标，含 ROUGE/BERTScore）
#
# 用法：
#   bash scripts/eval/eval-zeroshot-llm.sh            # 跑全部 5 个数据集
#   bash scripts/eval/eval-zeroshot-llm.sh USTC iOS   # 只跑指定数据集
# ==============================================================================
set -u

ROOT="/root/autodl-tmp/mmTraffic"
PY="/root/miniconda3/envs/mambanetbust/bin/python"
MODEL_PATH="$ROOT/model/Qwen3-1.7B"
BERT_MODEL="$ROOT/model/robert-large"
BATCH_SIZE=16
MAX_NEW_TOKENS=512

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# 数据集配置：test.jsonl | npy根目录 | 参考样本集(双阶段20/类) | 输出目录
declare -A TEST_JSONL NPY_ROOT REF_PRED OUT_DIR

TEST_JSONL[ISCXVPN]="$ROOT/data/ISCXVPN2016/nlp_output_LLMclass_200_6000/test.jsonl"
NPY_ROOT[ISCXVPN]="$ROOT/data/ISCXVPN2016/ISCXVPN2016_npy_split_npy_v3_balacned_200_6000"
REF_PRED[ISCXVPN]="$ROOT/output/ISCXVPN/phase2/eval_results_regen/predictions.jsonl"
OUT_DIR[ISCXVPN]="$ROOT/output/ISCXVPN/zeroshot_qwen3_1.7b"

TEST_JSONL[Tor]="$ROOT/data/ISCX-Tor-2016/nlp_output_LLMclass_3000_10000/test.jsonl"
NPY_ROOT[Tor]="$ROOT/data/ISCX-Tor-2016/Tor_split_pcap_merged_npy_v3_balacned_3000_10000"
REF_PRED[Tor]="$ROOT/output/ISCX-Tor-2016/phase2/eval_results_regen/predictions.jsonl"
OUT_DIR[Tor]="$ROOT/output/ISCX-Tor-2016/zeroshot_qwen3_1.7b"

TEST_JSONL[USTC]="$ROOT/data/USTC-TFC-2016/nlp_output_LLMclass_3000_6000/test.jsonl"
NPY_ROOT[USTC]="$ROOT/data/USTC-TFC-2016/USTC-TFC-2016_npy_v3_balacned_3000_6000"
REF_PRED[USTC]="$ROOT/output/USTC-TFC-2016/phase2/eval_results_regen/predictions.jsonl"
OUT_DIR[USTC]="$ROOT/output/USTC-TFC-2016/zeroshot_qwen3_1.7b"

TEST_JSONL[Android]="$ROOT/data/Crossplatform-Android/nlp_output_noLLMclass_50_2000/test.jsonl"
NPY_ROOT[Android]="$ROOT/data/Crossplatform-Android/CrossPlatform_android_pcaps_split_npy_v3_balacned_50_2000"
REF_PRED[Android]="$ROOT/output/Crossplatform-Android/mambanetburst_lora_phase2_direct/eval_results_regen/predictions.jsonl"
OUT_DIR[Android]="$ROOT/output/Crossplatform-Android/zeroshot_qwen3_1.7b"

TEST_JSONL[iOS]="$ROOT/data/CrossPlatform_ios/nlp_output_noLLMclass_50_3000/test.jsonl"
NPY_ROOT[iOS]="$ROOT/data/CrossPlatform_ios/CrossPlatform_ios_pcaps_split_npy_v3_balacned_50_3000"
REF_PRED[iOS]="$ROOT/output/Crossplatform-iOS/mambanetburst_lora_phase2_direct/eval_results_regen/predictions.jsonl"
OUT_DIR[iOS]="$ROOT/output/Crossplatform-iOS/zeroshot_qwen3_1.7b"

# 小数据集在前，快速拿到部分结果
DATASETS=("${@:-ISCXVPN Tor USTC Android iOS}")
[ $# -eq 0 ] && DATASETS=(ISCXVPN Tor USTC Android iOS)

for DS in "${DATASETS[@]}"; do
    if [ -z "${TEST_JSONL[$DS]:-}" ]; then
        echo "✗ 未知数据集: $DS（可选: ISCXVPN Tor USTC Android iOS）"; exit 1
    fi
    if [ ! -f "${REF_PRED[$DS]}" ]; then
        echo "✗ [$DS] 参考样本集不存在: ${REF_PRED[$DS]}"; exit 1
    fi

    OUT="${OUT_DIR[$DS]}"
    mkdir -p "$OUT"
    echo ""
    echo "════════════════════════════════════════════════════"
    echo ">>> [Zero-shot $DS] $(date '+%F %T')"
    echo "    参考样本集: ${REF_PRED[$DS]}"
    echo "    输出      : $OUT"
    echo "════════════════════════════════════════════════════"

    "$PY" "$ROOT/tinyllava/eval/eval_zeroshot_llm.py" \
        --test_jsonl "${TEST_JSONL[$DS]}" \
        --npy_root   "${NPY_ROOT[$DS]}" \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUT" \
        --sample_ids_from "${REF_PRED[$DS]}" \
        --batch_size $BATCH_SIZE \
        --max_new_tokens $MAX_NEW_TOKENS \
        2>&1 | tee "$OUT/run.log" || { echo "✗ [$DS] 推理失败"; exit 1; }

    echo ">>> [$DS] 计算全套指标（ROUGE-L / BERTScore）..."
    "$PY" "$ROOT/tinyllava/eval/evaluate_predictions.py" \
        --input  "$OUT/predictions.jsonl" \
        --output "$OUT/results.json" \
        --text --bert-model-path "$BERT_MODEL" \
        2>&1 | tee "$OUT/metrics.log" || echo "⚠ [$DS] 指标计算失败（可稍后单独重跑）"

    echo ">>> [$DS] 完成 $(date '+%F %T')"
done

echo ""
echo "全部完成。汇总："
for DS in "${DATASETS[@]}"; do
    S="${OUT_DIR[$DS]}/summary.json"
    [ -f "$S" ] && echo "[$DS] $(cat "$S" | tr -d '\n')"
done
