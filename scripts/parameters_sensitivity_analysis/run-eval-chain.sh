#!/bin/bash
# ==============================================================================
# 参数敏感性实验 —— eval 串式启动器
# 依次对每个 phase1 轮数 <k> 跑 eval-v4.sh <k>，前一组成功产出 all_results.json
# 才启动下一组；任一组失败则中止整条链并标 FAILED。
#
# 用法：
#   bash run-eval-chain.sh              # 默认依次跑 1 3 5 7
#   bash run-eval-chain.sh 1 3 5 7 10   # 自定义组别与顺序
#
# 建议用 screen 后台启动（每组 eval 独占 4 卡，必须串行）：
#   screen -dmS ps_eval_chain bash -lc "cd <此目录> && bash run-eval-chain.sh"
# ==============================================================================

set -u
SC="$(cd "$(dirname "$0")" && pwd)"
SA_ROOT="/root/autodl-tmp/mmTraffic/output/parameter_sensitivity_analysis"
BATCH_LOG="$SA_ROOT/eval_chain.log"

EPOCHS=("$@")
[ ${#EPOCHS[@]} -eq 0 ] && EPOCHS=(1 3 5 7)

mkdir -p "$SA_ROOT"
: > "$BATCH_LOG"
echo "===== EVAL CHAIN START [${EPOCHS[*]}] $(date) =====" | tee -a "$BATCH_LOG"

for k in "${EPOCHS[@]}"; do
    echo "===== EVAL EPOCH $k START $(date) =====" | tee -a "$BATCH_LOG"

    bash "$SC/eval-v4.sh" "$k"

    # eval-v4.sh 内部不保证以非 0 退出，故以「是否产出 all_results.json」判定成败
    RESULT="$SA_ROOT/${k}epoch/phase2/eval_results_regen/all_results.json"
    if [ ! -f "$RESULT" ]; then
        echo "===== EVAL EPOCH $k FAILED（未产出 all_results.json） $(date) — 链中止 =====" | tee -a "$BATCH_LOG"
        exit 1
    fi
    echo "===== EVAL EPOCH $k DONE $(date) =====" | tee -a "$BATCH_LOG"
done

echo "===== EVAL CHAIN ALL DONE [${EPOCHS[*]}] $(date) =====" | tee -a "$BATCH_LOG"
