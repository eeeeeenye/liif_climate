"""
hexbin_hsr_preds_vs_asos_2021jja_8x_lte_2019_2020_wmae_ep542.py 와 같은 데이터(ep542 epoch-best-loss, 2021 JJA 시간별, 필터 없음)를
hexbin 대신 산점도로 그리되, 같은 (지점,시각) 점이 양쪽 패널에서 같은 색이 되도록 ASOS 강도 구간별로 칠한다.
  - 색 기준 = ASOS 강도 구간 (양쪽 패널 공통). 강한 비가 위에 오도록 약한 구간부터 그린다.
출력: station_eval/figures/hexbin_hsr_preds_vs_asos/scatter_bincolor_hsr_preds_vs_asos_2021jja_8x_lte_2019_2020_wmae_ep542_<CKPT>_nofilter.png  (기존 파일은 건드리지 않음)
env: liif
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
CKPT = os.environ.get("CKPT", "epoch-best-loss")
NPZ = f"{HERE}/station_eval/preds_npz/station_preds_val2021_8x_lte_ft_2019_2020_wmae_ep542_{CKPT}.npz"
OUT = f"{HERE}/station_eval/figures/hexbin_hsr_preds_vs_asos/scatter_bincolor_hsr_preds_vs_asos_2021jja_8x_lte_2019_2020_wmae_ep542_{CKPT}_nofilter.png"
LIM = 80
EDGES = [0, 0.1, 1, 5, 10, 20, 30, 1e9]
LABELS = ["0", "0.1-1", "1-5", "5-10", "10-20", "20-30", ">=30"]
COLORS = ["#c8c8c8", "#9ecae1", "#4292c6", "#41ab5d", "#f1c40f", "#e67e22", "#d62728"]

d = np.load(NPZ)
m = np.isfinite(d["pf"]) & np.isfinite(d["hsr_hr"])
asos, hsr, pred = d["obs"][m], d["hsr_hr"][m], d["pf"][m]
cls = np.clip(np.digitize(asos, EDGES) - 1, 0, len(LABELS) - 1)   # 구간: [0,0.1) [0.1,1) [1,5) ...
cls = np.where(asos == 0, 0, np.maximum(cls, 1))                  # 첫 구간은 정확히 0 mm/h 만


def stats(x, y):
    return dict(r=np.corrcoef(x, y)[0, 1], bias=np.mean(x - y), rmse=np.sqrt(np.mean((x - y) ** 2)), ratio=x.mean() / y.mean())


fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=130)
for ax, x, title, xl in [(axes[0], hsr, "HSR vs ASOS  (hourly)", "HSR radar hourly precip (mm/h)"),
                         (axes[1], pred, "preds vs ASOS  (hourly)", "LTE preds hourly precip (mm/h)")]:
    s = stats(x, asos)
    for c in range(len(LABELS)):                      # 약한 구간부터 -> 강한 비가 위
        k = cls == c
        ax.scatter(x[k], asos[k], s=3 if c < 4 else 6, c=COLORS[c], alpha=0.5 if c < 4 else 0.85, linewidths=0, rasterized=True,
                   label=f"ASOS {LABELS[c]} (n={k.sum()})")
    ax.plot([0, LIM], [0, LIM], color="0.4", ls="--", lw=1)
    b, a = np.polyfit(x, asos, 1); xx = np.array([0, LIM]); ax.plot(xx, a + b * xx, color="k", lw=1.4, label=f"OLS: y = {a:.2f} + {b:.2f}x")
    ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel(xl); ax.set_ylabel("ASOS hourly precip (mm/h)")
    ax.set_title(f"{title}\nn={len(x)}  r={s['r']:.2f}  bias={s['bias']:.2f}  RMSE={s['rmse']:.2f}  ratio={s['ratio']:.2f}", fontsize=9)
leg = axes[1].legend(loc="upper left", fontsize=8, markerscale=3, framealpha=0.9)
for h in leg.legend_handles: h.set_alpha(1)
fig.suptitle("2021 JJA  hourly precip  (nearest 500m grid, all hours, no filter)\ncolor = ASOS intensity bin (same points in both panels)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT, bbox_inches="tight"); print("saved", OUT)
