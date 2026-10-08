"""
case_3hourly_lte_lora.py 산출 npz 로 3시간 누적 지도. 날짜별 2장 (00-12시 / 12-24시), 4개 시간대(행) x (Raw | Input | LTE pretrain | LoRA 체크포인트들)(열).
원 = ASOS 3시간합. 사용: python plot_case_3hourly_lte_lora.py 2021-07-06
env: liif (cartopy)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
dates = sys.argv[1:] or ["2021-07-06"]
EXTENT = [124.2, 131.7, 32.7, 38.8]
LEV = [1, 2, 5, 10, 20, 30, 50, 80, 100, 150]
cmap = ListedColormap(["white"] + list(plt.cm.turbo(np.linspace(0.08, 0.98, len(LEV) - 1))) + ["#4a0a2a"])
norm = BoundaryNorm(LEV, cmap.N, extend="both")


def at_stations(field, flat, flon, slat, slon):
    r = np.abs(flat[None, :] - slat[:, None]).argmin(1); c = np.abs(flon[None, :] - slon[:, None]).argmin(1)
    return field[r, c]


for date in dates:
    d = np.load(f"{HERE}/case_studies/3hourly/case_3h_{date}_lte_lora_r16.npz")
    valid = d["valid"]; names = list(d["names"]); eps = d["epochs"]
    cols = [("Raw HSR (500 m)", d["raw"], d["lat"], d["lon"], True),
            ("Input (1 km, 2x LR)", d["inp"], d["lr_lat"], d["lr_lon"], False),
            ("Radar pretrain LTE (500 m)", d["pred_lte"], d["lat"], d["lon"], True)]
    for n, e in zip(names[1:], eps[1:]):
        cols.append((f"LoRA r16 ep {e} (500 m)", d[f"pred_{n}"], d["lat"], d["lon"], True))
    nc = len(cols)
    for b0, tag in [(0, "00-12"), (4, "12-24")]:
        fig, axes = plt.subplots(4, nc, figsize=(5.5 * nc, 4 * 4.1 + 1.6), subplot_kw={"projection": ccrs.PlateCarree()})
        for r in range(4):
            b = b0 + r
            asos = d["st_asos"][b]; ok = np.isfinite(asos)
            for c, (title, fld, flat, flon, hi) in enumerate(cols):
                ax = axes[r, c]
                ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
                show = fld[b].astype(float).copy()
                if hi:
                    show[~valid] = np.nan
                ax.pcolormesh(flon, flat, show, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
                sc = ax.scatter(d["st_lon"][ok], d["st_lat"][ok], c=asos[ok], cmap=cmap, norm=norm, s=30, edgecolor="k",
                                linewidth=0.5, transform=ccrs.PlateCarree(), zorder=5)
                gl = ax.gridlines(draw_labels=(c == 0), linewidth=0.3, alpha=0.4)
                gl.top_labels = False; gl.right_labels = False; gl.bottom_labels = (r == 3)
                gl.xlabel_style = {"size": 9}; gl.ylabel_style = {"size": 9}
                if ok.sum() >= 5:
                    v = at_stations(np.nan_to_num(fld[b]), flat, flon, d["st_lat"][ok], d["st_lon"][ok])
                    st = f"r={np.corrcoef(v, asos[ok])[0, 1]:.2f}  bias={np.mean(v - asos[ok]):+.1f}  n={ok.sum()}"
                else:
                    st = f"n={ok.sum()}"
                ax.text(0.02, 0.02, st, transform=ax.transAxes, fontsize=10, va="bottom",
                        bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=2))
                if r == 0:
                    ax.set_title(title, fontsize=14)
                if c == 0:
                    ax.text(-0.30, 0.5, f"{3 * b:02d}–{3 * b + 3:02d} KST", transform=ax.transAxes, rotation=90,
                            va="center", ha="center", fontsize=16, fontweight="bold")
        cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.02, pad=0.03, extend="both", ticks=LEV)
        cb.set_label("3-hour precipitation (mm/3h)  |  circles = ASOS 3-h total (ASOS[T] <-> HSR[T-1h], lag 1h)", fontsize=13)
        fig.suptitle(f"{date} (KST), {tag}h: 3-hourly accumulation, raw / input / LTE / LoRA (Val 2021 JJA)", fontsize=18, y=0.995)
        out = f"{HERE}/case_studies/3hourly/case_3h_{date}_lte_lora_r16_{tag.replace('-', '_')}.png"
        fig.savefig(out, dpi=70, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)
