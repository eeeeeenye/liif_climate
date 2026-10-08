"""
2x ft 모델 hexbin 4패널, 전체 영상(nc 방식) 기준 — scatter_hourly_hsr_pred_vs_asos_jja2021_reg.png 와 같은 형식(검정 OLS, 빨간 점선 1:1, ASOS>=THR)에 패널 2개 추가:
  1: HSR vs ASOS | 2: 어댑터 입력 p_k(방향별 x4) vs ASOS | 3: 어댑터 출력 g(p_k)(방향별 x4) vs ASOS | 4: 최종 preds vs ASOS
p_k/가중치: preds_npz/pk_w_fullimage_<YEAR>_2x_pre2x.npz (collect_pk_weights_fullimage_2x.py, 2x 사전학습 동결 imnet 공통). 어댑터: 각 run epoch-best-loss.
사용: YEAR=2021 python (기본 THR=0 필터 없음; THR=0.1 이면 ASOS>=0.1)  예) THR=0.1 python hexbin_4panel_fullimage_2x_ft.py   (THR=0 이면 필터 없음)
출력: station_eval/figures/hexbin_hsr_preds_vs_asos/2x_ft/hexbin_4panel_fullimage_<YEAR>jja_<label>_ge<THR>.png
env: liif (CPU)
"""
import os
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
YEAR = int(os.environ.get("YEAR", "2021")); THR = float(os.environ.get("THR", "0"))
H = "/home/inhye_yoo/deeplearning/inr/liif"; S = f"{H}/save"; E = f"{H}/inference_tools/station_eval"; V = 74.8784; LIM, GRID = 80, 60
d = np.load(f"{E}/preds_npz/pk_w_fullimage_{YEAR}_2x_pre2x.npz"); P, W = d["P"], d["W"]
ok = np.isfinite(P[:, 0]); print("유효 행", ok.sum(), "/", len(P))
JOBS = [("2019only", "_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h", "2x_ft_2019only"),
        ("2019_2020", "_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020", "2x_ft_2019_2020")]
def stats(x, y): return dict(r=np.corrcoef(x, y)[0, 1], bias=np.mean(x - y), rmse=np.sqrt(np.mean((x - y) ** 2)), ratio=x.mean() / y.mean())
for tag, run, label in JOBS:
    ck = torch.load(f"{S}/{run}/epoch-best-loss.pth", map_location="cpu", weights_only=False); ep = ck["epoch"]
    ad = {k.replace("add_layer.asos_adapter.", ""): v for k, v in ck["model"]["sd"].items() if "add_layer.asos_adapter" in k}
    L = []
    for i, o in zip([1, 32, 16], [32, 16, 32]): L += [nn.Linear(i, o), nn.ReLU()]
    net = nn.Sequential(*L, nn.Linear(32, 1)); net.load_state_dict(ad); net.eval()
    Pn = np.nan_to_num(P)
    with torch.no_grad(): G = np.clip(net(torch.tensor(Pn / V, dtype=torch.float32).reshape(-1, 1)).numpy().reshape(P.shape) * V, 0, None)
    pred = (G * W).sum(1) / W.sum(1)
    m = ok & (d["obs"] >= THR); asos, hsr, pr = d["obs"][m], d["hsr_hr"][m], pred[m]; pk_ = P[m].reshape(-1); g = G[m].reshape(-1); a4 = np.repeat(asos, 4)
    panels = [(hsr, asos, "HSR vs ASOS  (hourly)", "HSR radar hourly precip (mm/h)"), (pk_, a4, "adapter input p_k vs ASOS  (per direction x4)", "adapter input p_k (mm/h)"),
              (g, a4, "adapter output g(p_k) vs ASOS  (per direction x4)", "adapter output g(p_k) (mm/h)"), (pr, asos, "preds vs ASOS  (hourly)", "LIIF preds hourly precip (mm/h)")]
    def cmax(x, y):
        f, a = plt.subplots(); h = a.hexbin(x, y, gridsize=GRID, extent=(0, LIM, 0, LIM), mincnt=1); v = h.get_array().max(); plt.close(f); return v
    vmax = max(cmax(x, y) for x, y, _, _ in panels)
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.2), dpi=130)
    for ax, (x, y, title, xl) in zip(axes, panels):
        s = stats(x, y); hb = ax.hexbin(x, y, gridsize=GRID, extent=(0, LIM, 0, LIM), mincnt=1, norm=LogNorm(1, vmax), cmap="viridis")
        ax.plot([0, LIM], [0, LIM], "r--", lw=1, label="1:1"); b, a = np.polyfit(x, y, 1); xx = np.array([0, LIM]); ax.plot(xx, a + b * xx, "k-", lw=1.6, label=f"OLS: y = {a:.2f} + {b:.2f}x")
        ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlabel(xl); ax.set_ylabel("ASOS hourly precip (mm/h)")
        ax.set_title(f"{title}\nn={len(x)}  r={s['r']:.2f}  bias={s['bias']:.2f}  RMSE={s['rmse']:.2f}  ratio={s['ratio']:.2f}", fontsize=9); ax.legend(loc="upper left", fontsize=9)
        print(f"  [{label}]", title, {k: round(float(v), 3) for k, v in s.items()}, f"OLS a={a:.3f} b={b:.3f}")
    flt = "all hours, no filter" if THR <= 0 else f"ASOS rain hours >= {THR} mm/h"
    fig.suptitle(f"{YEAR} JJA  hourly precip  (nearest 500m grid, {flt})  |  adapter: {label} (ep {ep}), full-image input", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.955, 0.96)); cax = fig.add_axes([0.965, 0.15, 0.01, 0.7]); fig.colorbar(hb, cax=cax).set_label("count (log)")
    out = f"{E}/figures/hexbin_hsr_preds_vs_asos/2x_ft/hexbin_4panel_fullimage_{YEAR}jja_{label}_ge{THR:g}.png"; fig.savefig(out); plt.close(fig); print("   saved", out)
