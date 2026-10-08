"""
2021 JJA 한 날의 시간별(mm/h) 지도: Raw HSR (500m) | Input (1 km) | x2 (500 m, MLP ft lag1h).  (plot_case_3hourly_hsr_2x_finer.py 의 시간별판, 3열)
행 = 시각(HH 시작, 1시간). 원 = 그 시각 ASOS 시간강수 (같은 색; lag1h -> obs csv 라벨이 HSR 시각). 패널 좌하단 = 관측소 값 vs 지도 r / bias / n.
자료: x2 / Raw = save/..._ft_mlp_1e-3_lag1h/preds_nc/hourly_2021_epoch-best-loss/hsr2x_ft_mlp_hourly_<날짜>.nc (precip, precip_gt),
      Input = HSR_val_norm_2x_obscrop.nc precip_norm * PMAX,  ASOS = hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv
사용:  python plot_case_hourly_hsr_2x.py 2021-08-23 18 19 20      # 날짜, 시각들(HH). 시각 생략 시 18 19 20
env: liif (cartopy)
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid"
PRED = ("/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h/"
        "preds_nc/hourly_2021_epoch-best-loss/hsr2x_ft_mlp_hourly_{d}.nc")
OBS = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv"
PMAX = 74.8784
date = sys.argv[1] if len(sys.argv) > 1 else "2021-08-23"
hours = [int(h) for h in sys.argv[2:]] or [18, 19, 20]
DOT = int(os.environ.get("DOT", "30"))
LEV = [float(v) for v in os.environ["LEV"].split(",")] if os.environ.get("LEV") else [0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 80]   # mm/h
cmap = ListedColormap(["white"] + list(plt.cm.turbo(np.linspace(0.08, 0.98, len(LEV) - 1))) + ["#4a0a2a"])
norm = BoundaryNorm(LEV, cmap.N, extend="both")


def box(lat, lon):
    dy, dx = np.diff(lat).mean(), np.diff(lon).mean()
    return [lon[0] - dx / 2, lon[-1] + dx / 2, lat[0] - dy / 2, lat[-1] + dy / 2]


def at(field, flat, flon, slat, slon):
    r = np.abs(flat[None, :] - slat[:, None]).argmin(1); c = np.abs(flon[None, :] - slon[:, None]).argmin(1)
    return field[r, c]


pr = xr.open_dataset(PRED.format(d=date)); lat, lon = pr.lat.values, pr.lon.values
lr = xr.open_dataset(f"{M}/HSR_val_norm_2x_obscrop.nc")
tl = pd.to_datetime(lr.time.values)
obs = pd.read_csv(OBS, encoding="utf-8-sig"); obs["t"] = pd.to_datetime(obs["time"]); obs = obs[obs.v.notna()]
EXT = box(lat, lon)

fig, axes = plt.subplots(len(hours), 3, figsize=(15.5, len(hours) * 4.3 + 1.6), subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
for r, h in enumerate(hours):
    ts = pd.Timestamp(f"{date} {h:02d}:00")
    raw = pr.precip_gt.isel(time=h).values; x2 = pr.precip.isel(time=h).values
    x2 = np.where(np.isfinite(raw), x2, np.nan)                                          # 레이더 영역 밖은 Raw 와 같이 비움 (3시간판과 동일)
    inp = lr.precip_norm.isel(time=int(np.where(tl == ts)[0][0])).values * PMAX
    o = obs[obs.t == ts]
    slat, slon, sv = o.y.values, o.x.values, o.v.values
    inb = (slon >= EXT[0]) & (slon <= EXT[1]) & (slat >= EXT[2]) & (slat <= EXT[3])
    slat, slon, sv = slat[inb], slon[inb], sv[inb]
    cols = [("Raw HSR (500 m, GT)", raw, lat, lon), ("Input (1 km)", inp, lr.lat.values, lr.lon.values), ("MLP ft x2: 500 m", x2, lat, lon)]
    for c, (title, arr, fl, fo) in enumerate(cols):
        ax = axes[r, c]
        ax.set_extent(EXT, crs=ccrs.PlateCarree())
        cm = cmap.copy(); cm.set_bad("white")
        ax.imshow(np.where(np.isfinite(arr), arr, np.nan), origin="lower", extent=box(fl, fo), transform=ccrs.PlateCarree(), cmap=cm, norm=norm,
                  interpolation="nearest", aspect="auto", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
        sc = ax.scatter(slon, slat, c=sv, cmap=cmap, norm=norm, s=DOT, edgecolor="k", linewidth=0.5, transform=ccrs.PlateCarree(), zorder=5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        gl.top_labels = False; gl.right_labels = False; gl.left_labels = (c == 0); gl.bottom_labels = (r == len(hours) - 1)
        gl.xlabel_style = {"size": 9}; gl.ylabel_style = {"size": 9}
        v = at(np.nan_to_num(arr), fl, fo, slat, slon)
        st = f"r={np.corrcoef(v, sv)[0, 1]:.2f}  bias={np.mean(v - sv):+.1f}  n={len(sv)}" if len(sv) >= 5 else f"n={len(sv)}"
        ax.text(0.02, 0.02, st, transform=ax.transAxes, fontsize=10, va="bottom", zorder=6, bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=2))
        if r == 0:
            ax.set_title(title, fontsize=14)
        if c == 0:
            ax.text(-0.30, 0.5, f"{h:02d}–{h + 1:02d} KST", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=16, fontweight="bold")
        print(f"  {h:02d}h {title:22s} {st}")
cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.03, pad=0.05, extend="both", ticks=LEV)
cb.set_label("hourly precipitation (mm/h)  |  circles = ASOS hourly (ASOS[T] <-> HSR[T-1h], lag 1h)", fontsize=13)
fig.suptitle(f"{date} (KST): hourly precipitation, 1 km input -> x2 (whole domain, Val 2021 JJA, MLP ft)", fontsize=16, y=0.995)
out = f"{HERE}/case_studies/hourly/case_hourly_{date}_2x_{'_'.join(f'{h:02d}' for h in hours)}.png"
fig.savefig(out, dpi=70, bbox_inches="tight"); print("saved", out)
