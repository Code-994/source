#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
iOS 文本质量五边形雷达图：5 指标 × 6 轮(phase1 warmup epochs)叠一圈。
  轴(顺时针，从顶开始)：FCR, Ev-RL, De-RL, De-BS, Ev-BS
    FCR   = field_completeness_rate (metrics.json, /100)
    Ev-RL = evidence_rouge_l        De-RL = description_rouge_l
    Ev-BS = evidence_bertscore_f1   De-BS = description_bertscore_f1
  样式：外圈实线 + 内圈 0.2/0.4/0.6/0.8/1.0 虚线；6 轮细线、明显区分色、实/虚交替。
  Acc 已单独出图，此处不含。
用法： python plot_text_radar.py
输出： output/parameter_sensitivity_analysis/figures/ios_text_radar.{png,pdf}
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SA = "/root/autodl-tmp/mmTraffic/output/parameter_sensitivity_analysis"
M2 = "/root/autodl-tmp/mmTraffic/output/Crossplatform-iOS/mambanetburst_lora_phase2_direct"
EPOCHS = [1, 3, 5, 7, 10, 12]
# 顺时针从顶：Ev-BS(12点) → FCR(右上) → Ev-RL(右下) → De-RL(左下) → De-BS(左上)
# 使三个高值轴(Ev-BS/FCR/De-BS)在上方、两个低值 ROUGE 轴在底部对称 → 风筝摆正
AXES = ["Ev-BERTScore", "FCR", "Ev-ROUGE-L", "Des-ROUGE-L", "Des-BERTScore"]

# 6 组：明显区分色 + 实/虚交替 + 交替标记
COLORS = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#17BECF"]
LSTYLES = ["-", "--", "-", "--", "-", "--"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def basedir(k):
    return f"{SA}/{k}epoch/phase2/eval_results_regen" if k != 10 else f"{M2}/eval_results_regen"


# ── E₁=10 的文本指标以 output/aggregate/main_results_table.md 为准 ───────────
# 该点即主实验 run（mambanetburst_lora_phase2_direct）。论文表 6-5 的 iOS 双阶段列与
# 该 run 现存产物不同（产物：FCR 0.9985 / Ev-RL 0.6872 / Ev-BS 0.9383 / De-RL 0.6154 /
# De-BS 0.9341）；图、表、正文统一按权威表取值。见 main_results_table.md 第四节。
TEXT_OVERRIDE = {
    10: {"FCR": 0.9859, "Ev-ROUGE-L": 0.6429, "Ev-BERTScore": 0.9280, "Des-ROUGE-L": 0.6022, "Des-BERTScore": 0.9363},
}


def row(k):
    if k in TEXT_OVERRIDE:
        return [TEXT_OVERRIDE[k][ax] for ax in AXES]
    a = json.load(open(f"{basedir(k)}/all_results.json"))["text_metrics"]
    m = json.load(open(f"{basedir(k)}/metrics.json"))
    vals = {
        "FCR": m["field_completeness_rate"] / 100,
        "Ev-ROUGE-L": a["evidence_rouge_l"],
        "Des-ROUGE-L": a["description_rouge_l"],
        "Des-BERTScore": a["description_bertscore_f1"],
        "Ev-BERTScore": a["evidence_bertscore_f1"],
    }
    return [vals[ax] for ax in AXES]


data = {k: row(k) for k in EPOCHS}
N = len(AXES)
ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
ang_closed = np.concatenate([ang, ang[:1]])

# 非等距径向刻度：抬高下限去掉空心圆心 + 非均匀虚线圈，6 组在各轴铺开且不变形
FLOOR = 0.4
def rmap(v):
    return (np.clip(np.asarray(v, float), FLOOR, 1.0) - FLOOR) / (1.0 - FLOOR)

fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)      # 顶部为起点
ax.set_theta_direction(-1)          # 顺时针

for i, k in enumerate(EPOCHS):
    v = np.array(data[k])
    vc = rmap(np.concatenate([v, v[:1]]))
    ax.plot(ang_closed, vc, color=COLORS[i], lw=1.2, ls=LSTYLES[i],
            marker=MARKERS[i], ms=4.2, mew=0.6, zorder=5 + i, label=f"{k} ep")

highlight_epochs1 = [10]
k = 10
idx_ep = EPOCHS.index(k)
color_h = COLORS[idx_ep]
v_raw = np.array(data[k])
v_mapped = rmap(v_raw)

text_cfg = [
    (0.032,  -0.05),   # i0 Ev‑BS
    (0.018,  -0.05),   # i1 FCR
    (0.030,  0.10),   # i2 Ev‑RL
    (0.07,  0.1),   # i3 De‑RL
    (0.05,  -0.05)    # i4 De‑BS
]

for i in range(len(ang)):
    angle_origin = ang[i]
    r_origin = v_mapped[i]
    val_i = v_raw[i]

    dr, dtheta = text_cfg[i]
    # 计算新位置：角度偏转，径向拉长
    new_angle = angle_origin + dtheta
    new_r = r_origin + dr

    ax.text(new_angle, new_r, f"{val_i:.2f}",
            ha="center", va="bottom", fontsize=7.2, color=color_h, fontweight="semibold")

highlight_epochs2 = [12]
k = 12
idx_ep = EPOCHS.index(k)
color_h = COLORS[idx_ep]
v_raw = np.array(data[k])
v_mapped = rmap(v_raw)

# AXES：
# i=0:Ev‑BS(顶部)
# i=1:FCR(右上)
# i=2:Ev‑RL(右下)
# i=3:De‑RL(左下)
# i=4:De‑BS(左上)
# 每一项：(dr径向偏移, dtheta角度偏移)
# dr>0向外；dtheta>0顺时针，dtheta<0逆时针
text_cfg = [
    (0.018,  0.08),   # i0 Ev‑BS
    (0.012,  0.10),   # i1 FCR
    (0.030,  0.10),   # i2 Ev‑RL
    (0.05,  -0.08),   # i3 De‑RL
    (0.025,  -0.10)    # i4 De‑BS
]

for i in range(len(ang)):
    angle_origin = ang[i]
    r_origin = v_mapped[i]
    val_i = v_raw[i]

    dr, dtheta = text_cfg[i]
    # 计算新位置：角度偏转，径向拉长
    new_angle = angle_origin + dtheta
    new_r = r_origin + dr

    ax.text(new_angle, new_r, f"{val_i:.2f}",
            ha="center", va="bottom", fontsize=7.2, color=color_h, fontweight="semibold")

# 轴标签（缩写、加粗）
ax.set_xticks(ang)
ax.set_xticklabels(AXES, fontsize=10, fontweight="semibold")

# 径向：非均匀虚线圈(外密内疏)；最外单独一条实线圈
RINGS = [0.4, 0.6, 0.8, 0.9, 1.0]
ax.set_ylim(0, rmap(1.0) * 1.06)
ax.set_yticks([rmap(r) for r in RINGS])
ax.set_yticklabels([f"{r:.1f}" for r in RINGS], fontsize=6, color="black")
ax.set_rlabel_position(255)
ax.yaxis.grid(True, linestyle=(0, (4, 3)), color="0.72", lw=0.8, alpha=0.9)  # 内圈虚线
ax.xaxis.grid(True, linestyle="-", color="0.85", lw=0.7)                     # 角向细实线
ax.spines["polar"].set_visible(True)
ax.spines["polar"].set_color("0.30")
ax.spines["polar"].set_linewidth(1.4)      # 最外实线圈

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=6,
          fontsize=9, columnspacing=1.1, handlelength=2.0, frameon=False)

fig.tight_layout()
outdir = f"{SA}/figures"; os.makedirs(outdir, exist_ok=True)
for ext in ("png", "pdf"):
    p = f"{outdir}/ios_text_radar.{ext}"; fig.savefig(p, dpi=200, bbox_inches="tight"); print("[saved]", p)
