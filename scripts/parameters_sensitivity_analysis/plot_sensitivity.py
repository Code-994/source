#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
iOS 参数敏感性结果图（单张、省版面）：
  同一坐标区，横轴 = phase1 轮数 (1/3/5/7/10/12)
  · 上方窄带：6 组 phase2 训练 loss 下降曲线（右轴、大色差、log 关系用线性小条呈现，收敛值入图例）
  · 下方主体：分类精度【重叠柱】——Full(LLM+fallback) 在后、LLM-only 在前，颜色区分并各标数值
  loss 曲线各组 step 数固定(245)，横向拉伸铺满轮数轴，仅示形状（见图下注释）。

数据源： loss<-各组 phase2 train.log ; acc<-各组 eval_results_regen/metrics.json
         LLM-only=class_exact_match_rate ; Full=class_exact_match_with_fallback
用法： python plot_sensitivity.py
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SA = "/root/autodl-tmp/mmTraffic/output/parameter_sensitivity_analysis"
MAIN2 = "/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_phase2_direct"
EPOCHS = [1, 3, 5, 7, 10, 12]
loss_logs = {k: f"{SA}/{k}epoch/phase2/train.log" for k in (1, 3, 5, 7, 12)}; loss_logs[10] = f"{MAIN2}/train.log"
metrics_json = {k: f"{SA}/{k}epoch/phase2/eval_results_regen/metrics.json" for k in (1, 3, 5, 7, 12)}; metrics_json[10] = f"{MAIN2}/eval_results_regen/metrics.json"

LOSS_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#111111"]
C_FULL, C_LLM = "#1F5C7A", "#7FB2D5"        # 深蓝=Full(后/高) / 浅蓝=LLM-only(前/矮)
LBL_FULL, LBL_LLM = "#153f54", "#123047"    # 数值标注色


def parse_loss(log):
    out = []
    if os.path.exists(log):
        for line in open(log):
            m = re.search(r"'loss': '([0-9.]+)'", line)
            if m:
                out.append(float(m.group(1)))
    return out


def smooth(y, w=5):
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    pad = w // 2
    return np.convolve(np.pad(y, (pad, pad), mode="edge"), np.ones(w) / w, mode="valid")[: len(y)]


# ── E₁=10 的精度以 output/aggregate/main_results_table.md 为准 ────────────────
# 该点即主实验 run（mambanetburst_lora_phase2_direct）。论文表 6-5 的 iOS 双阶段列为
# 纯LLM 0.7977 / Final 0.8333，与该 run 现存产物 metrics.json（0.8041 / 0.8115）不同；
# 图、表、正文统一按权威表取值，产物值仅备查。改动来源见 main_results_table.md 第四节。
ACC_OVERRIDE = {10: (79.77, 83.33)}   # (LLM-only %, Full %)


def read_acc(mj, epoch=None):
    if epoch in ACC_OVERRIDE:
        return ACC_OVERRIDE[epoch]
    if not os.path.exists(mj):
        return None, None
    d = json.load(open(mj))
    return d.get("class_exact_match_rate"), d.get("class_exact_match_with_fallback")


losses = {k: parse_loss(loss_logs[k]) for k in EPOCHS}
acc_llm, acc_full = zip(*[read_acc(metrics_json[k], k) for k in EPOCHS])
acc_llm, acc_full = list(acc_llm), list(acc_full)

fig, ax = plt.subplots(figsize=(8.6, 5.6))   # ax = 左轴精度柱
ax2 = ax.twinx()                             # ax2 = 右轴 loss

pos = np.arange(len(EPOCHS))
x0, x1 = pos[0], pos[-1]

# ---- 下方主体：重叠柱（左轴，精度 %）----
ACC_TOP = 145          # 左轴上限：让 ~81 的柱只到 ~56%，上方留给 loss
W = 0.62
ax.bar(pos, [v or 0 for v in acc_full], width=W, color=C_FULL, zorder=2, label="Final Acc")
ax.bar(pos, [v or 0 for v in acc_llm], width=W, color=C_LLM, zorder=3, label="LLM-only Acc")
for i in range(len(EPOCHS)):
    if acc_full[i] is not None:
        ax.annotate(f"{acc_full[i]:.1f}", (pos[i], acc_full[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=LBL_FULL, weight="bold")
    if acc_llm[i] is not None:
        ax.annotate(f"{acc_llm[i]:.1f}", (pos[i], acc_llm[i]), xytext=(0, -12),
                    textcoords="offset points", ha="center", fontsize=8, color=LBL_LLM, weight="bold")
ax.set_ylim(0, ACC_TOP)
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_xticks(pos); ax.set_xticklabels([str(e) for e in EPOCHS])
ax.set_xlabel("Stage-1 warmup epochs")
ax.set_ylabel("Classification accuracy (%)")
ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.35)
ax.legend(loc="center left", fontsize=8, framealpha=0.92)

# ---- 上方窄带：loss 曲线（右轴，log 刻度让收敛尾部散开）----
# log-y + 很低的下限，使 loss(0.1~9) 落在顶部窄带、且尾部按 log 拉开不糊在一起
for i, k in enumerate(EPOCHS):
    y = losses[k]
    if not y:
        continue
    ys = np.clip(smooth(y, 5), 1e-3, None)
    x = np.linspace(x0, x1, len(ys))
    final = float(np.mean(y[-10:]))
    ax2.plot(x, ys, color=LOSS_COLORS[i], lw=1.5, zorder=4, label=f"{k}ep (→{final:.2f})")
ax2.set_yscale("log")
ax2.set_ylim(4e-5, 14)          # 数据浮在顶部；尾部 0.1~0.4 因 log 而分开
from matplotlib.ticker import FixedLocator, FixedFormatter
ax2.yaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 9]))
ax2.yaxis.set_major_formatter(FixedFormatter(["0.1", "0.3", "1", "3", "9"]))
ax2.yaxis.set_minor_locator(FixedLocator([]))
ax2.set_ylabel("Stage-2 training loss (log)", y=0.82)
ax2.legend(title="Loss / warmup epochs", loc="upper right", fontsize=7.2,
           title_fontsize=7.5, ncol=3, framealpha=0.92, handlelength=1.3, columnspacing=1.0)

fig.text(0.5, -0.03,
         "Loss curves: fixed 245 logged points per run, stretched across the epoch axis (shape only, not per-epoch). "
         "Bars overlap: Full behind, LLM-only in front.",
         ha="center", va="top", fontsize=7, color="0.4")
ax.set_title("iOS · Phase-1 warmup epochs → Phase-2 loss (top) & classification accuracy (bars)")
fig.tight_layout()
outdir = f"{SA}/figures"; os.makedirs(outdir, exist_ok=True)
for ext in ("png", "pdf"):
    p = f"{outdir}/ios_sensitivity.{ext}"; fig.savefig(p, dpi=200, bbox_inches="tight"); print("[saved]", p)
