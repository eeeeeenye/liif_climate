"""
[bias ±0.5(0.1 bins), MAE/corr 변화 ±0.3(0.05 bins) 판 — raw_vs_ft_pm0.3 그림과 같은 범위]
지점별 지표 지도 3x3 (Val 2021 JJA):  행 = mean bias(pred-obs) / MAE / corr,  열 = base / finetuned / 변화(ft-base).
입력 CSV (station_metrics_val2021*.py 산출): lat, lon, n, bias_b/f, mae_b/f, corr_b/f
  변화 패널: bias/MAE 는 RdBu_r (파랑=음수=개선, 빨강=양수=악화). corr 도 값 기준(RdBu_r, 파랑=작음/음수, 빨강=큼/양수)으로 통일
  -> corr 는 양수(빨강)가 개선(corr_f-corr_b > 0). bias/MAE 만 "파랑=개선", corr 은 "빨강=개선"이니 색과 방향이 반대임에 주의.

사용:  python plot_station_metric_maps.py [csv] [out.png] [base 라벨]
env: liif (cartopy)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
CSV = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/station_eval/tables/station_metrics_val2021_lag1h.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/station_eval/figures/station_metric_maps/finetune_station_metric_maps_val2021_lag1h.png"
BASE = sys.argv[3] if len(sys.argv) > 3 else "base"   # 좌측 열/변화 패널 기준 라벨 (예: "raw HSR")
MIN_N = 30
EXTENT = [124.2, 131.7, 32.7, 38.8]

df = pd.read_csv(CSV)
n_total = int(df["n"].sum())
d_bias = df.bias_f.abs() - df.bias_b.abs()
d_mae = df.mae_f - df.mae_b
d_corr = df.corr_f - df.corr_b

ROWS = [  # (행 이름, base 값, ft 값, 변화 값, 값 levels, 값 cmap, 값 라벨, 변화 cmap, 변화 제목, 값 제목 접미)
    ("bias", df.bias_b, df.bias_f, d_bias, np.arange(-2.2, 2.21, 0.2), "RdBu_r", "mm/hr", "RdBu_r",
     "|bias| change (ft-base), blue=better [0.1 bins, ±0.5]", "mean bias (pred-obs) [0.2 bins]"),
    ("mae", df.mae_b, df.mae_f, d_mae, np.arange(0, 3.01, 0.2), "YlOrRd", "mm/hr", "RdBu_r",
     "MAE change (ft-base), blue=better [0.05 bins, ±0.3]", "MAE [0.2 bins]"),
    ("corr", df.corr_b, df.corr_f, d_corr, np.arange(-1, 1.01, 0.1), "YlGnBu", "corr", "RdBu_r",
     "corr change (ft-base), red=increase(better) [0.05 bins, ±0.3]", "corr with ASOS [0.1 bins, -1~1]"),
]
DLEV = np.arange(-0.5, 0.51, 0.1)
DLEV3 = np.arange(-0.3, 0.301, 0.05)
DLEVS = {"bias": DLEV, "mae": DLEV3, "corr": DLEV3}   # 변화 패널 컬러바 범위(행별)

fig, axes = plt.subplots(3, 3, figsize=(20, 17), subplot_kw={"projection": ccrs.PlateCarree()})


def base_map(ax):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 11}; gl.ylabel_style = {"size": 11}


def draw(ax, vals, levels, cmap, extend):
    norm = BoundaryNorm(levels, plt.get_cmap(cmap).N, extend=extend)
    sc = ax.scatter(df.lon, df.lat, c=vals, cmap=cmap, norm=norm, s=85, edgecolor="k", linewidth=0.6,
                    transform=ccrs.PlateCarree(), zorder=5)
    return sc


for r, (key, vb, vf, dv, lev, cmap, lab, dcmap, dtitle, vtitle) in enumerate(ROWS):
    ext_val = "neither"
    for c, (v, title) in enumerate([(vb, f"{BASE}: {vtitle}"), (vf, f"finetuned: {vtitle}"), (dv, dtitle.replace("ft-base", f"ft-{BASE}"))]):
        ax = axes[r, c]
        base_map(ax)
        if c < 2:
            sc = draw(ax, v.clip(lev[0], lev[-1]), lev, cmap, ext_val)
        else:
            sc = draw(ax, v, DLEVS[key], dcmap, "both")
        ax.set_title(title, fontsize=13)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.72)   # 세 열 모두 컬러바 (지도 크기 통일)
        cb.set_label(lab if c < 2 else ("corr" if key == "corr" else "mm/hr"), fontsize=11)
        if c < 2:
            cb.set_ticks(lev[::2])

fig.suptitle(f"2020 (Full Year), per-station metrics vs ASOS (n={n_total} station-hours, station n>={MIN_N})",
             fontsize=17, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(OUT, dpi=126)
print("saved", OUT)
print(f"stations={len(df)}  평균 |bias| base {df.bias_b.abs().mean():.3f} -> ft {df.bias_f.abs().mean():.3f} | "
      f"MAE {df.mae_b.mean():.3f} -> {df.mae_f.mean():.3f} | corr {df.corr_b.mean():.3f} -> {df.corr_f.mean():.3f}")
print(f"ft 가 더 나은 지점 비율: |bias| {100*(d_bias<0).mean():.0f}%  MAE {100*(d_mae<0).mean():.0f}%  corr {100*(d_corr>0).mean():.0f}%")
