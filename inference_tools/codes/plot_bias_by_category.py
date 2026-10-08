"""
강수 구간별 평균 bias 막대 그림 (Mean Bias Relative to ... by Precipitation Category) — HSR 버전.

입력: station_preds_dump_val2021.py 산출 npz (obs=ASOS, pb=radar pretrain(base), pf=MLP ft(MSE, lag1h), hsr_hr=원시 HSR 500m 최근접)
출력 (inference_tools/):
  bias_by_category_val2021_vs_asos.png   기준 = ASOS, 구간 = ASOS 강도.  bias = 예측 - ASOS (음수 = 과소).  막대: Raw HSR(모델 전) / Radar pretrain / MLP ft
  bias_by_category_val2021_vs_rawhsr.png 기준 = 원시 HSR 500m, 구간 = 원시 HSR 강도. 막대: Radar pretrain / MLP ft
bias = 예측 - 기준값 (mm/hour).  Val 2021 JJA, 96개 ASOS 지점의 매시 값.

사용: python plot_bias_by_category.py [npz]
env: liif
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
NPZ = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/station_eval/preds_npz/station_preds_val2021_lag1h_mse.npz"
EDGES = [0, 1, 2, 5, 10, 20, 50, np.inf]
LABELS = ["0–1", "1–2", "2–5", "5–10", "10–20", "20–50", "≥50"]
C_RAW, C_PRE, C_FT = "#2e9b2e", "#dc2a4c", "#4f78e0"     # 원 그림과 같은 초록/빨강/파랑

d = np.load(NPZ)
obs, pb, pf, hsr = d["obs"], d["pb"], d["pf"], d["hsr_hr"]


def bars(ref, series, title, xlabel, out, ylabel, flip=False):
    """ref 로 구간을 나누고, series = [(이름, 예측값, 색)] 각각의 (값-ref) 평균을 막대로.
    flip=True 이면 부호를 반대로 (ref-값): 양수 = 값이 기준보다 작음(과소)."""
    ok = np.isfinite(ref)
    for _, v, _ in series:
        ok &= np.isfinite(v)
    cat = np.digitize(ref, EDGES) - 1
    nb, k = len(LABELS), len(series)
    means = np.full((k, nb), np.nan); ns = np.zeros(nb, int)
    for b in range(nb):
        m = ok & (cat == b); ns[b] = m.sum()
        for i, (_, v, _) in enumerate(series):
            if m.sum():
                means[i, b] = ((ref[m] - v[m]) if flip else (v[m] - ref[m])).mean()
    fig, ax = plt.subplots(figsize=(19, 10))
    w = 0.26 if k == 3 else 0.34
    x = np.arange(nb)
    lo = np.nanmin(means); hi = np.nanmax(means)
    span = max(hi - min(lo, 0), 0.5)
    for i, (name, _, col) in enumerate(series):
        off = (i - (k - 1) / 2) * w
        b = ax.bar(x + off, means[i], w, color=col, edgecolor="k", linewidth=1.2, label=name, alpha=0.95)
        for xi, mv in zip(x + off, means[i]):
            if np.isnan(mv):
                continue
            ax.text(xi, mv + (0.02 * span if mv >= 0 else -0.02 * span), f"{mv:+.2f}", rotation=70,
                    ha="left" if mv >= 0 else "right", va="bottom" if mv >= 0 else "top",
                    fontsize=15, fontweight="bold", style="italic")
    ax.axhline(0, color="k", ls="--", lw=2.5)
    ymin = min(lo, 0) - 0.42 * span; ymax = max(hi, 0) + (0.30 if flip else 0.18) * span
    ax.set_ylim(ymin, ymax)
    for xi, n in zip(x, ns):
        ax.text(xi, ymin + 0.02 * (ymax - ymin), f"n={n:,}", rotation=90, ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=20, fontweight="bold")
    ax.tick_params(axis="y", labelsize=18)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.set_xlabel(xlabel, fontsize=26, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=24, fontweight="bold")
    ax.set_title(title, fontsize=30, fontweight="bold", pad=14)
    ax.grid(axis="y", ls=":", alpha=0.6)
    if flip:
        ax.legend(loc="upper left", prop={"weight": "bold", "size": 17})
    else:
        ax.legend(loc="lower left", bbox_to_anchor=(0.005, 0.15), prop={"weight": "bold", "size": 17})
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print("saved", out)
    print(f"  {'구간':7s}" + "".join(f"{s[0][:18]:>20s}" for s in series) + "        n")
    for b in range(nb):
        print(f"  {LABELS[b]:7s}" + "".join(f"{means[i, b]:20.3f}" for i in range(k)) + f"  {ns[b]:8d}")


bars(obs, [("Before - Raw HSR", hsr, C_RAW), ("Radar pretrain", pb, C_PRE), ("MLP ft (MSE)", pf, C_FT)],
     "Mean Bias Relative to ASOS by ASOS Precipitation Category (Val 2021 JJA)",
     "ASOS precipitation category (mm/hour)", f"{HERE}/station_eval/figures/bias_by_category/bias_by_category_val2021_vs_asos.png",
     "Mean Bias [Prediction - ASOS] (mm/hour)")
bars(hsr, [("Radar pretrain - Raw HSR", pb, C_PRE), ("MLP ft (MSE) - Raw HSR", pf, C_FT)],
     "Mean Bias Relative to Raw HSR by Precipitation Category (Val 2021 JJA)",
     "Raw HSR precipitation category (mm/hour)", f"{HERE}/station_eval/figures/bias_by_category/bias_by_category_val2021_vs_rawhsr.png",
     "Mean Bias [Prediction - Raw HSR] (mm/hour)")
