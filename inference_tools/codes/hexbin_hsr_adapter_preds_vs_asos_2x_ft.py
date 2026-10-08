"""
2x ft 모델의 hexbin 4패널 (hexbin_hsr_adapter_preds_vs_asos_8x_ft.py 와 같은 형식, 2021 JJA, 기본 필터 없음):
  1: HSR 500m vs ASOS | 2: 어댑터 입력 p_k(방향별 4개) vs ASOS | 3: 어댑터 출력 g(p_k) vs ASOS | 4: 최종 pred (g 의 면적가중평균) vs ASOS
p_k/가중치: preds_npz/pk_w_cache_2x_val2021.npz (2x 사전학습 동결 imnet; 2x ft 모델들은 imnet 이 같아 공통, collect_pk_weights_2x.py 가 생성).
어댑터: 각 run 의 epoch-best-loss.pth (복사본 사용). pred 재현 오차를 station_preds_*.npz 의 pf 와 비교해 출력.
사용: THR=0(기본)/0.1 python hexbin_hsr_adapter_preds_vs_asos_2x_ft.py
출력: station_eval/figures/hexbin_hsr_preds_vs_asos/2x_ft/hexbin_hsr_adapter_preds_vs_asos_2021jja_2x_ft_<tag>_epoch-best-loss{_nofilter}_4panel.png
env: liif (CPU)
"""
import os, shutil
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
H = "/home/inhye_yoo/deeplearning/inr/liif"; S = f"{H}/save"; E = f"{H}/inference_tools/station_eval"; V = 74.8784; LIM, GRID = 80, 60
USE_NC = os.environ.get("USE_NC") == "1"   # 2019only 모델: 마지막 패널을 전체 영상(nc) 예측으로 교체
THR = float(os.environ.get("THR", "0")); SUF = "_nofilter" if THR <= 0 else ""
JOBS = [("2019_2020", "_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020", "station_preds_val2021_2x_ft_2019_2020_epoch-best-loss.npz", "2x_ft_2019_2020"),
        ("2019only", "_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h", "station_preds_val2021_lag1h_mse_new.npz", "2x_ft_2019only")]
w = np.load(f"{E}/preds_npz/pk_w_cache_2x_val2021.npz"); P, W = w["P"], w["W"]

def stats(x, y): return dict(r=np.corrcoef(x, y)[0, 1], bias=np.mean(x - y), rmse=np.sqrt(np.mean((x - y) ** 2)), ratio=x.mean() / y.mean())
for tag, run, npz, label in JOBS:
    d = np.load(f"{E}/preds_npz/{npz}"); assert np.abs(w["O"] - d["obs"]).max() < 1e-3, "순서 불일치"
    ck = torch.load(f"{S}/{run}/epoch-best-loss.pth", map_location="cpu", weights_only=False); ep = ck["epoch"]
    ad = {k.replace("add_layer.asos_adapter.", ""): v for k, v in ck["model"]["sd"].items() if "add_layer.asos_adapter" in k}
    L = []
    for i, o in zip([1, 32, 16], [32, 16, 32]): L += [nn.Linear(i, o), nn.ReLU()]
    net = nn.Sequential(*L, nn.Linear(32, 1)); net.load_state_dict(ad); net.eval()
    with torch.no_grad(): G = np.clip(net(torch.tensor(P / V, dtype=torch.float32).reshape(-1, 1)).numpy().reshape(P.shape) * V, 0, None)
    pred = (G * W).sum(1) / W.sum(1); print(f"[{label}] ep {ep}  pred 재현 오차(max) {np.abs(pred - d['pf']).max():.4f}")
    if USE_NC and label == "2x_ft_2019only":
        nz = np.load(f"{E}/preds_npz/station_preds_nc_2021_2x_pre2x.npz"); assert np.abs(nz["obs"] - d["obs"]).max() < 1e-3; pred = nz["pf_nc"]
    m = d["obs"] >= THR; asos, hsr = d["obs"][m], d["hsr_hr"][m]; pr = pred[m]; pk_ = P[m].reshape(-1); g = G[m].reshape(-1); a4 = np.repeat(asos, 4)
    panels = [(hsr, asos, "HSR vs ASOS  (hourly)", "HSR radar hourly precip (mm/h)"), (pk_, a4, "adapter input p_k vs ASOS  (per direction x4)", "adapter input p_k (mm/h)"),
              (g, a4, "adapter output g(p_k) vs ASOS  (per direction x4)", "adapter output g(p_k) (mm/h)"), (pr, asos, "final preds vs ASOS  (hourly)" + ("  [full-image nc]" if USE_NC and label == "2x_ft_2019only" else ""), "LIIF preds hourly precip (mm/h)")]
    def cmax(x, y):
        f, a = plt.subplots(); h = a.hexbin(x, y, gridsize=GRID, extent=(0, LIM, 0, LIM), mincnt=1); v = h.get_array().max(); plt.close(f); return v
    vmax = max(cmax(x, y) for x, y, _, _ in panels)
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.2), dpi=130)
    for ax, (x, y, title, xl) in zip(axes, panels):
        s = stats(x, y); hb = ax.hexbin(x, y, gridsize=GRID, extent=(0, LIM, 0, LIM), mincnt=1, norm=LogNorm(1, vmax), cmap="viridis")
        ax.plot([0, LIM], [0, LIM], color="0.5", ls="--", lw=1, label="1:1"); b, a = np.polyfit(x, y, 1); xx = np.array([0, LIM])
        ax.plot(xx, a + b * xx, color="red", lw=1.6, label=f"OLS: y = {a:.2f} + {b:.2f}x")
        ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel(xl); ax.set_ylabel("ASOS hourly precip (mm/h)")
        ax.set_title(f"{title}\nn={len(x)}  r={s['r']:.2f}  bias={s['bias']:.2f}  RMSE={s['rmse']:.2f}  ratio={s['ratio']:.2f}", fontsize=9); ax.legend(loc="upper left", fontsize=9)
        print("  ", title, {k: round(float(v), 3) for k, v in s.items()}, f"OLS a={a:.3f} b={b:.3f}")
    fig.suptitle(f"2021 JJA  hourly precip  (nearest 500m grid, {'all hours, no filter' if THR <= 0 else f'ASOS rain hours >= {THR} mm/h'})  |  adapter: {label} (ep {ep})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.955, 0.96)); cax = fig.add_axes([0.965, 0.15, 0.01, 0.7]); fig.colorbar(hb, cax=cax).set_label("count (log)")
    NCS = "_ncfinal" if USE_NC and label == "2x_ft_2019only" else ""
    out = f"{E}/figures/hexbin_hsr_preds_vs_asos/2x_ft/hexbin_hsr_adapter_preds_vs_asos_2021jja_{label}_epoch-best-loss{SUF}{NCS}_4panel.png"; fig.savefig(out); plt.close(fig); print("   saved", out)
