"""
plot_case_3hourly_zoom.py 의 *_full 과 같은 npz/범위(남한 전체)로, 배경 필드 없이 ASOS 지점 산점만 그린다.
열: ASOS 관측 | Raw HSR | Input(1km) | Radar pretrain | MLP ft (점 색 = 그 지점 3시간 누적; 모델 열은 지점에서의 모델 값)
사용: python plot_case_3hourly_scatter_full.py   (DOT_S 로 점 크기 조절, 기본 45)
env: liif (cartopy)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
NPZ_TAG = os.environ.get("NPZ_TAG", "lag1h_ext")
DOT_S = float(os.environ.get("DOT_S", "28"))
date, name, (clat, clon), EXT = "2023-07-15", "Osong (approx.)", (36.6200, 127.3300), [124.2, 131.7, 32.7, 38.8]
LEV = [1, 1.4, 2, 3, 4, 6, 8, 12, 16, 22, 30, 42, 60, 85, 120, 150]
cmap = ListedColormap(["white"] + list(plt.cm.turbo(np.linspace(0.08, 0.98, len(LEV) - 1))) + ["#4a0a2a"])
norm = BoundaryNorm(LEV, cmap.N, extend="both")


def at_stations(field, flat, flon, slat, slon):
    r = np.abs(flat[None, :] - slat[:, None]).argmin(1); c = np.abs(flon[None, :] - slon[:, None]).argmin(1)
    return field[r, c]


def corr(v, o):
    return np.corrcoef(v, o)[0, 1] if len(v) >= 3 and v.std() > 0 and o.std() > 0 else np.nan


d = np.load(f"{HERE}/case_studies/3hourly/case_3h_{date}_{NPZ_TAG}.npz")
ep = int(d["ep_ft"])
inbox = (d["st_lon"] >= EXT[0]) & (d["st_lon"] <= EXT[1]) & (d["st_lat"] >= EXT[2]) & (d["st_lat"] <= EXT[3])
slat, slon, sasos = d["st_lat"][inbox], d["st_lon"][inbox], d["st_asos"][:, inbox]
cols = [("Raw HSR (500 m)", d["raw"], d["lat"], d["lon"]), ("Input (1 km, 2x LR)", d["inp"], d["lr_lat"], d["lr_lon"]),
        ("Radar pretrain (500 m)", d["pre"], d["lat"], d["lon"]), (f"MLP ft (MSE, ep {ep}) (500 m)", d["prd"], d["lat"], d["lon"])]
stv = np.stack([np.stack([at_stations(np.nan_to_num(f[b]), la, lo, slat, slon) for b in range(8)]) for _, f, la, lo in cols])

for tag, b0 in [("00_12", 0), ("12_24", 4)]:
    fig, axes = plt.subplots(4, 5, figsize=(26, 4 * 4.1 + 1.6), subplot_kw={"projection": ccrs.PlateCarree()})
    for r in range(4):
        b = b0 + r
        asos = sasos[b]; ok = np.isfinite(asos)
        for c in range(5):
            ax = axes[r, c]
            ax.set_extent(EXT, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="0.95"); ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            val = asos if c == 0 else stv[c - 1, b]
            sc = ax.scatter(slon[ok], slat[ok], c=val[ok], cmap=cmap, norm=norm, s=DOT_S, edgecolor="k", linewidth=0.5, transform=ccrs.PlateCarree(), zorder=5)
            ax.plot(clon, clat, marker="*", color="magenta", ms=9, mec="k", mew=0.6, transform=ccrs.PlateCarree(), zorder=4)
            if c > 0:
                st = f"r={corr(val[ok], asos[ok]):.2f}  bias={np.mean(val[ok] - asos[ok]):+.1f}  n={ok.sum()}" if ok.sum() >= 3 else f"n={ok.sum()}"
                ax.text(0.02, 0.02, st, transform=ax.transAxes, fontsize=10, va="bottom", zorder=7, bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=2))
            gl = ax.gridlines(draw_labels=(c == 0), linewidth=0.3, alpha=0.4)
            gl.top_labels = False; gl.right_labels = False; gl.bottom_labels = (r == 3)
            gl.xlabel_style = {"size": 9}; gl.ylabel_style = {"size": 9}
            if r == 0:
                ax.set_title("ASOS observed (3-h total)" if c == 0 else cols[c - 1][0], fontsize=14)
            if c == 0:
                ax.text(-0.30, 0.5, f"{3 * b:02d}–{3 * b + 3:02d} KST", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=16, fontweight="bold")
    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.02, pad=0.03, extend="both", ticks=LEV)
    cb.set_label("3-hour precipitation (mm/3h)  |  dots = ASOS stations (model columns: value at the station, ASOS[T] <-> HSR[T-1h])  |  star = " + name, fontsize=13)
    fig.suptitle(f"{date} (KST), {tag.replace('_', '-')}h, South Korea ({name} marked): station scatter only (out-of-sample date)", fontsize=18, y=0.995)
    out = f"{HERE}/case_studies/3hourly/case_3h_{date}_scatter_osong_full_{tag}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight"); plt.close(fig); print("saved", out)
