"""
2021 JJA 시간별: HSR 500m vs ASOS / 8x ft(2019+2020) 예측 vs ASOS hexbin (x=레이더/예측, y=ASOS).
  - 입력: station_preds_dump_val2021_8x_ft_2019_2020.py 가 만든 npz (지점 최근접 500m 격자 예측 pf, 원시 HSR hsr_hr, ASOS obs; mm/h).
  - 짝: ASOS >= THR(기본 0.1, THR=0 이면 필터 없음) mm/h 인 (지점,시각) 만 (lag1h 정렬은 obs csv 에 이미 반영).
  - 통계: r, bias = mean(x - y), RMSE, ratio = mean(x)/mean(y). OLS: y(ASOS) = a + b x.
사용: python hexbin_hsr_preds_vs_asos_2021jja_8x_2019_2020.py     # CKPT=epoch-best-loss (기본)
출력: inference_tools/station_eval/figures/hexbin_hsr_preds_vs_asos/hexbin_hsr_preds_vs_asos_2021jja_8x_2019_2020_<CKPT>.png
env: liif
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
CKPT = os.environ.get("CKPT", "epoch-best-loss")
NPZ = f"{HERE}/station_eval/preds_npz/station_preds_val2021_8x_lte_ft_2019_2020_wmae_ep542_{CKPT}.npz"
LIM, GRID = 80, 60
THR = float(os.environ.get("THR", "0.1"))   # THR=0 이면 필터 없음(무강수 시간 포함)
SUF = "_nofilter" if THR <= 0 else ""
OUT = f"{HERE}/station_eval/figures/hexbin_hsr_preds_vs_asos/hexbin_hsr_preds_vs_asos_2021jja_8x_lte_2019_2020_wmae_ep542_{CKPT}{SUF}.png"

d = np.load(NPZ)
m = (d["obs"] >= THR) & np.isfinite(d["pf"]) & np.isfinite(d["hsr_hr"])
asos, hsr, pred = d["obs"][m], d["hsr_hr"][m], d["pf"][m]
print(f"n={m.sum()}")


def stats(x, y):
    return dict(r=np.corrcoef(x, y)[0, 1], bias=np.mean(x - y), rmse=np.sqrt(np.mean((x - y) ** 2)), ratio=x.mean() / y.mean())


def counts_max(x, y):
    fig, ax = plt.subplots()
    hb = ax.hexbin(x, y, gridsize=GRID, extent=(0, LIM, 0, LIM), mincnt=1)
    c = hb.get_array().max(); plt.close(fig); return c


vmax = max(counts_max(hsr, asos), counts_max(pred, asos))
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=130)
for ax, x, title, xl in [(axes[0], hsr, "HSR vs ASOS  (hourly)", "HSR radar hourly precip (mm/h)"),
                         (axes[1], pred, "preds vs ASOS  (hourly)", "LTE preds hourly precip (mm/h)")]:
    s = stats(x, asos)
    hb = ax.hexbin(x, asos, gridsize=GRID, extent=(0, LIM, 0, LIM), mincnt=1, norm=LogNorm(1, vmax), cmap="viridis")
    ax.plot([0, LIM], [0, LIM], color="0.5", ls="--", lw=1, label="1:1")
    b, a = np.polyfit(x, asos, 1)
    xx = np.array([0, LIM]); ax.plot(xx, a + b * xx, color="red", ls="-", lw=1.6, label=f"OLS: y = {a:.2f} + {b:.2f}x")
    ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel(xl); ax.set_ylabel("ASOS hourly precip (mm/h)")
    ax.set_title(f"{title}\nn={len(x)}  r={s['r']:.2f}  bias={s['bias']:.2f}  RMSE={s['rmse']:.2f}  ratio={s['ratio']:.2f}", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    print(title, {k: round(float(v), 3) for k, v in s.items()}, f"OLS a={a:.3f} b={b:.3f}")
SUBT = "all hours, no filter" if THR <= 0 else f"ASOS rain hours >= {THR} mm/h"
fig.suptitle(f"2021 JJA  hourly precip  (nearest 500m grid, {SUBT})", fontsize=13)
fig.tight_layout(rect=(0, 0, 0.92, 1))
cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
cb = fig.colorbar(hb, cax=cax); cb.set_label("count (log)")
fig.savefig(OUT); print("saved", OUT)
