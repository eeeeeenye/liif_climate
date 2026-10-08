"""
2021 JJA 시간별: 예측(MLP ft lag1h, 500m) vs ASOS 시간별 강수 를 joint PDF(2D 히스토그램 밀도)로. 비교용으로 Raw HSR(GT) vs ASOS 를 옆에 같이 그림.
  - 짝 = lag1h obs csv 의 (지점, 시각) 행 (라벨 = HSR 시각 -> ASOS[T] <-> HSR[T-1h] 이미 반영), 지점 위치의 최근접 500m 격자값.
  - 기록이 있는 (지점,시각) 만 사용 (건조 시각은 csv 에 행이 없음 -> 표본이 비 온 시각 쪽으로 치우침; 원 그림 "every matched hour" 와 같은 정의).
  - 그림: x = ASOS (mm/h), y = 레이더/예측 (mm/h) (축 교환본). 통계: r, bias(x-y), RMSE, MAE, ratio(mean x / mean y).
  - 밀도 = 개수/(N * bin면적) [(mm/h)^-2], LogNorm. 선형축 그림 + log-log 축 그림 2장.
사용: python scatter_jointpdf_pred_vs_asos_2021jja.py
출력: inference_tools/station_eval/figures/scatter_jointpdf_pred_vs_asos/scatter_jointpdf_pred_vs_asos_2021jja{,_loglog}.png, pairs npz
env: liif
"""
import os, glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
PRED = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h/preds_nc/hourly_2021_epoch-best-loss"
OBS = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv"
PAIRS = f"{HERE}/station_eval/preds_npz/pairs_pred_gt_asos_2021jja.npz"

if not os.path.exists(PAIRS):
    obs = pd.read_csv(OBS, encoding="utf-8-sig"); obs["t"] = pd.to_datetime(obs["time"])
    obs = obs[obs.v.notna() & (obs.v >= 0)].copy()
    obs["day"] = obs.t.dt.strftime("%Y-%m-%d"); obs["hour"] = obs.t.dt.hour
    ref = xr.open_dataset(sorted(glob.glob(f"{PRED}/hsr2x_ft_mlp_hourly_2021-0[678]*.nc"))[0])
    lat, lon = ref.lat.values, ref.lon.values
    obs["r"] = np.abs(lat[None, :] - obs.y.values[:, None]).argmin(1); obs["c"] = np.abs(lon[None, :] - obs.x.values[:, None]).argmin(1)
    P, G, A, D = [], [], [], []
    for day, g in obs.groupby("day"):
        f = f"{PRED}/hsr2x_ft_mlp_hourly_{day}.nc"
        if not os.path.exists(f):
            continue
        ds = xr.open_dataset(f); pr = ds.precip.values; gt = ds.precip_gt.values
        h, r, c = g.hour.values, g.r.values, g.c.values
        P.append(pr[h, r, c]); G.append(gt[h, r, c]); A.append(g.v.values); D += [day] * len(g)
        print(day, len(g), flush=True)
    P, G, A = np.concatenate(P), np.concatenate(G), np.concatenate(A)
    np.savez(PAIRS, pred=P, gt=G, asos=A, day=np.array(D))
d = np.load(PAIRS); P, G, A = d["pred"], d["gt"], d["asos"]
ok = np.isfinite(P) & np.isfinite(G)                                   # 레이더 영역 밖 제거
P, G, A = P[ok], G[ok], A[ok]
print(f"짝 {len(A):,}개 (ASOS>0: {(A > 0).sum():,}, ASOS>=1: {(A >= 1).sum():,})")


def stats(x, y):
    return dict(r=np.corrcoef(x, y)[0, 1], bias=np.mean(x - y), rmse=np.sqrt(np.mean((x - y) ** 2)), mae=np.mean(np.abs(x - y)), ratio=x.mean() / y.mean())


def panel(ax, x, y, title, log):
    s = stats(x, y)                                              # 통계는 x=레이더/예측, y=ASOS 기준 그대로 (bias = 레이더 - ASOS)
    nx, ny = x.max(), y.max()
    x, y = y, x                                                  # 그림은 축 교환: x축 = ASOS, y축 = 레이더/예측
    if log:
        e = np.logspace(-2, 2, 90); xe = ye = e
        m = (x > 0.01) & (y > 0.01)
        H, _, _ = np.histogram2d(x[m], y[m], bins=[xe, ye])
        area = np.outer(np.diff(xe), np.diff(ye)); pdf = H.T / (m.sum() * area)
        ax.set_xscale("log"); ax.set_yscale("log"); lim = (0.01, 100)
        note = f"(log 축은 x,y>0.01 만: n={m.sum():,})"
    else:
        xe = ye = np.linspace(0, 100, 201)                       # 0.5 mm/h bin
        H, _, _ = np.histogram2d(x, y, bins=[xe, ye])
        pdf = H.T / (len(x) * np.outer(np.diff(xe), np.diff(ye))); lim = (0, 100); note = ""
    pdf = np.ma.masked_where(pdf <= 0, pdf)
    im = ax.pcolormesh(xe, ye, pdf, cmap="viridis", norm=LogNorm(vmin=1e-6 if not log else 1e-5, vmax=pdf.max()), shading="flat", rasterized=True)
    ax.plot(lim, lim, "r--", lw=1.2, label="1:1")
    a, b = np.polyfit(x, y, 1)                                   # OLS: 레이더/예측 = a*ASOS + b
    xx = np.linspace(lim[0], lim[1], 200)
    ax.plot(xx, a * xx + b, color="k", lw=1.6, label=f"OLS: y={a:.2f}x{b:+.2f}")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"{title}\nr={s['r']:.2f}  bias={s['bias']:+.2f}  RMSE={s['rmse']:.2f}  MAE={s['mae']:.2f}  ratio={s['ratio']:.2f}", fontsize=11)
    ax.text(0.03, 0.97, f"n={len(x):,}\nmax pred {nx:.0f}\nmax ASOS {ny:.0f}\n{note}", transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", ec="0.7", alpha=0.85))
    ax.grid(alpha=0.3)
    return im, s


for log, out in [(False, f"{HERE}/station_eval/figures/scatter_jointpdf_pred_vs_asos/scatter_jointpdf_pred_vs_asos_2021jja_reg_swapped.png"), (True, f"{HERE}/station_eval/figures/scatter_jointpdf_pred_vs_asos/scatter_jointpdf_pred_vs_asos_2021jja_loglog_reg_swapped.png")]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    im, s1 = panel(axes[0], G, A, "Raw HSR radar vs ASOS", log)
    im, s2 = panel(axes[1], P, A, "MLP ft prediction (500 m) vs ASOS", log)
    for ax, yl in zip(axes, ["HSR radar hourly (mm/h)", "prediction hourly (mm/h)"]):
        ax.set_xlabel("ASOS hourly (mm/h)"); ax.set_ylabel(yl)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02); cb.set_label("joint PDF  [(mm/h)$^{-2}$]")
    fig.suptitle("2021 JJA hourly precipitation  -  joint PDF vs ASOS  (1-h lag aligned; every matched hour, val)", fontsize=13)
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig); print("saved", out)
