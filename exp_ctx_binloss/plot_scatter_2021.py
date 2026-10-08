"""[새 파일] eval_station_2021.py 가 저장한 2021 JJA 예측 npz 로 2패널 hexbin 산점도(HSR vs ASOS | 모델 preds vs ASOS) 를 그린다.
형식은 scatter_hourly_hsr_pred_vs_asos_jja2021_reg.png 와 같다 (x=HSR/예측, y=ASOS, viridis log count, 빨간 점선 1:1, 검정 OLS).
사용 (liif 루트): NPZ=<npz> LABEL="A 문맥 어댑터" EP=<epoch> THR=0 python exp_ctx_binloss/plot_scatter_2021.py     (THR=0 필터 없음, THR=0.1 이면 ASOS>=0.1)
출력: exp_ctx_binloss/results/figs/scatter_2021_<LABEL 기호>_ep<EP>_<nofilter|ge0.1>.png
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ, LABEL, EP = os.environ["NPZ"], os.environ.get("LABEL", "model"), os.environ.get("EP", "?"); THR = float(os.environ.get("THR", "0")); TAG = os.environ.get("TAG", "X")
d = np.load(NPZ); m = d["obs"] >= THR; a, h, p = d["obs"][m], d["hsr_hr"][m], d["pf"][m].astype(float); LIM = 80
def stats(x, y): return dict(r=np.corrcoef(x, y)[0, 1], bias=np.mean(x - y), rmse=np.sqrt(np.mean((x - y) ** 2)), ratio=x.mean() / y.mean())
def cmax(x, y):
    f, ax = plt.subplots(); hb = ax.hexbin(x, y, gridsize=60, extent=(0, LIM, 0, LIM), mincnt=1); v = hb.get_array().max(); plt.close(f); return v
vmax = max(cmax(h, a), cmax(p, a)); fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), dpi=130)
for ax, x, title, xl in [(axes[0], h, "HSR vs ASOS  (hourly)", "HSR radar hourly precip (mm/h)"), (axes[1], p, "preds vs ASOS  (hourly)", "LTE preds hourly precip (mm/h)")]:
    s = stats(x, a); hb = ax.hexbin(x, a, gridsize=60, extent=(0, LIM, 0, LIM), mincnt=1, norm=LogNorm(1, vmax), cmap="viridis")
    ax.plot([0, LIM], [0, LIM], "r--", lw=1, label="1:1"); b, c0 = np.polyfit(x, a, 1); xx = np.array([0, LIM]); ax.plot(xx, c0 + b * xx, "k-", lw=1.6, label=f"OLS: y = {c0:.2f} + {b:.2f}x")
    ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlabel(xl); ax.set_ylabel("ASOS hourly precip (mm/h)")
    ax.set_title(f"{title}\nn={len(x)}  r={s['r']:.2f}  bias={s['bias']:.2f}  RMSE={s['rmse']:.2f}  ratio={s['ratio']:.2f}", fontsize=9); ax.legend(loc="upper left", fontsize=9)
    print(title, {k: round(float(v), 3) for k, v in s.items()}, f"OLS a={c0:.3f} b={b:.3f}")
flt = "all hours, no filter" if THR <= 0 else f"ASOS rain hours >= {THR:g} mm/h"
fig.suptitle(f"2021 JJA  hourly precip  (nearest 500m grid, {flt})  |  {LABEL} (best-loss ep {EP}, intermediate checkpoint)", fontsize=12)
fig.subplots_adjust(right=0.9); cax = fig.add_axes([0.93, 0.15, 0.013, 0.7]); fig.colorbar(hb, cax=cax, label="count (log)")
out = f"{ROOT}/exp_ctx_binloss/results/figs/scatter_2021_{TAG}_ep{EP}_{'nofilter' if THR <= 0 else 'ge%g' % THR}.png"; fig.savefig(out, bbox_inches="tight"); print("saved", out)
