"""
case_daily_hsr_lag1h.py 산출 npz 로 case 일누적 지도 그림.
열: Raw(500m) | Input(1km) | Radar pretrain | MLP ft(MSE) — 원 = ASOS 일합(같은 색), 부제 = 관측소 최근접 격자 대비 r / bias / RMSE.
사용: python plot_case_daily.py 2021-07-06 2021-08-23
env: liif (cartopy)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
args = sys.argv[1:]
LAG = 1                                             # ASOS[T] <-> HSR[T-LAG h].  --lag N 으로 변경 (기본 1 = 정렬 정답)
if "--lag" in args:
    i = args.index("--lag"); LAG = int(args[i + 1]); del args[i:i + 2]
dates = args or ["2021-07-06", "2021-08-23"]
OBS_RAW = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_val.csv"   # 라벨 T 그대로 (lag 0)
EXTENT = [124.2, 131.7, 32.7, 38.8]
LEV = [1, 5, 10, 20, 30, 50, 80, 100, 150, 200, 300]
# extend="both" 이면 색 bin 이 len(LEV)+1 개: 맨 아래(<1 mm) 흰색, 맨 위(>300 mm) 짙은 자주
cmap = ListedColormap(["white"] + list(plt.cm.turbo(np.linspace(0.08, 0.98, len(LEV) - 1))) + ["#4a0a2a"])
norm = BoundaryNorm(LEV, cmap.N, extend="both")


def nearest_vals(field, flat, flon, slat, slon):
    r = np.abs(flat[None, :] - slat[:, None]).argmin(1); c = np.abs(flon[None, :] - slon[:, None]).argmin(1)
    return field[r, c]


fig, axes = plt.subplots(len(dates), 4, figsize=(24, 5.6 * len(dates) + 1.4), subplot_kw={"projection": ccrs.PlateCarree()})
axes = np.atleast_2d(axes)
for r, date in enumerate(dates):
    d = np.load(f"{HERE}/case_studies/daily/case_daily_{date}_lag1h.npz")
    valid = d["valid"]
    cols = [("Raw HSR (500 m)", d["raw"], d["lat"], d["lon"], True),
            ("Input (1 km, 2x LR)", d["inp"], d["lr_lat"], d["lr_lon"], False),
            ("Radar pretrain (500 m)", d["pre"], d["lat"], d["lon"], True),
            (f"MLP ft (MSE, ep {int(d['ep_ft'])}) (500 m)", d["prd"], d["lat"], d["lon"], True)]
    # HSR 하루 = KST D 00~23시.  ASOS 창 = 보고시각 T 가 [D+LAG h, D+24h+LAG h) 인 것의 합  (LAG=1 -> 01~24시 = 달력일합)
    o = pd.read_csv(OBS_RAW, encoding="utf-8-sig"); o["t"] = pd.to_datetime(o["time"])
    t0 = pd.Timestamp(date) + pd.Timedelta(hours=LAG)
    o = o[(o.t >= t0) & (o.t < t0 + pd.Timedelta(hours=24)) & o.v.notna()]
    st = o.groupby(["x", "y"]).v.sum().reset_index()
    a_lat, a_lon, a_val = st.y.values, st.x.values, st.v.values
    for c, (title, fld, flat, flon, hi) in enumerate(cols):
        ax = axes[r, c]
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        show = fld.astype(float).copy()
        if hi:
            show[~valid] = np.nan
        ax.pcolormesh(flon, flat, show, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        sc = ax.scatter(a_lon, a_lat, c=a_val, cmap=cmap, norm=norm, s=42, edgecolor="k", linewidth=0.6,
                        transform=ccrs.PlateCarree(), zorder=5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4); gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {"size": 10}; gl.ylabel_style = {"size": 10}
        v = nearest_vals(np.nan_to_num(fld), flat, flon, a_lat, a_lon)
        rr = np.corrcoef(v, a_val)[0, 1]
        ax.set_title(f"{title}\nr={rr:.2f}  bias={np.mean(v - a_val):+.1f}  RMSE={np.sqrt(np.mean((v - a_val) ** 2)):.1f} mm/day"
                     f"   (max {np.nanmax(show):.0f})", fontsize=12)
        if c == 0:
            ax.text(-0.27, 0.5, f"{date} (KST)  [lag {LAG}h]\nASOS max {a_val.max():.0f} mm/day\nn={len(a_val)} stations", transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=14, fontweight="bold")
cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.025, pad=0.05, extend="both", ticks=LEV)
cb.set_label(f"daily precipitation (mm/day)  |  circles = ASOS daily total (ASOS[T] <-> HSR[T-{LAG}h], lag {LAG}h)", fontsize=13)
fig.suptitle("Heavy-rain case days (Val 2021 JJA): raw / input / prediction, daily accumulation", fontsize=18, y=0.965)
out = f"{HERE}/case_studies/daily/case_daily_{'_'.join(dates)}_lag{LAG}h.png"
fig.savefig(out, dpi=100, bbox_inches="tight")
print("saved", out)
