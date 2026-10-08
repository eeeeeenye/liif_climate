"""
plot_case_3hourly_hsr_2x_multires_pow2.py 와 동일하나, 가장 오른쪽에 ASOS 관측소 위치만 표시하는 열을 하나 추가.
사용: python plot_case_3hourly_hsr_2x_multires_pow2_asoscol.py 2021-07-06
env: liif (cartopy)
"""
import os
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
TAG = os.environ.get("TAG", "_ft")
LABEL = os.environ.get("LABEL", "MLP ft")
SHOW_DOTS = os.environ.get("DOTS", "1") != "0"
SCALES = [("500", 1, "x2"), ("250", 2, "x4"), ("125", 4, "x8"), ("62.5", 8, "x16")]
LEV = [1, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 55, 75, 100, 140]
cmap = ListedColormap(["white"] + list(plt.cm.turbo(np.linspace(0.08, 0.98, len(LEV) - 1))) + ["#4a0a2a"])
norm = BoundaryNorm(LEV, cmap.N, extend="both")


def box(lat, lon):
    dy, dx = np.diff(lat).mean(), np.diff(lon).mean()
    return [lon[0] - dx / 2, lon[-1] + dx / 2, lat[0] - dy / 2, lat[-1] + dy / 2]


def at(field, flat, flon, slat, slon):
    r = np.abs(flat[None, :] - slat[:, None]).argmin(1); c = np.abs(flon[None, :] - slon[:, None]).argmin(1)
    return field[r, c]


for date in dates:
    d = np.load(f"{HERE}/case_studies/3hourly/case_3h_{date}_2x_multires_pow2{TAG}.npz")
    ep = int(d["ep"]); lat, lon = d["lat"], d["lon"]
    EXT = box(lat, lon)
    valid = d["valid"]
    cols = [("Raw HSR (500 m)", "raw", d["raw"], lat, lon, valid),
            ("Input (1 km, 2x LR)", "inp", d["inp"], d["lr_lat"], d["lr_lon"], None)]
    for name, k, xtag in SCALES:
        vm = np.kron(valid, np.ones((k, k), bool)) if k > 1 else valid
        cols.append((f"{LABEL} {name} m  [{xtag}]", f"m{name}", d[f"m{name}"], d[f"lat{name}"], d[f"lon{name}"], vm))
    ncol = len(cols) + 1  # +1: ASOS station-only column
    inb = (d["st_lon"] >= EXT[0]) & (d["st_lon"] <= EXT[1]) & (d["st_lat"] >= EXT[2]) & (d["st_lat"] <= EXT[3])
    slat, slon, sasos = d["st_lat"][inb], d["st_lon"][inb], d["st_asos"][:, inb]
    sv = {key: np.stack([at(f[b], fl, fo, slat, slon) for b in range(8)]) for _, key, f, fl, fo, _ in cols}

    for tag, b0 in [("00_12", 0), ("12_24", 4)]:
        fig, axes = plt.subplots(4, ncol, figsize=(5.2 * ncol, 4 * 4.3 + 1.6), subplot_kw={"projection": ccrs.PlateCarree()})
        for r in range(4):
            b = b0 + r
            asos = sasos[b]; ok = np.isfinite(asos)
            for c, (title, key, arr, fl, fo, vm) in enumerate(cols):
                ax = axes[r, c]
                ax.set_extent(EXT, crs=ccrs.PlateCarree())
                show = arr[b].astype(np.float32)
                if vm is not None:
                    show = np.where(vm, show, np.nan)
                cm = cmap.copy(); cm.set_bad("white")
                im = ax.imshow(show, origin="lower", extent=box(fl, fo), transform=ccrs.PlateCarree(), cmap=cm, norm=norm,
                                interpolation="nearest", aspect="auto", zorder=1)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
                if SHOW_DOTS:
                    sc = ax.scatter(slon[ok], slat[ok], c=asos[ok], cmap=cmap, norm=norm, s=70, edgecolor="k", linewidth=0.7,
                                    transform=ccrs.PlateCarree(), zorder=5)
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
                gl.top_labels = False; gl.right_labels = False
                gl.left_labels = (c == 0); gl.bottom_labels = (r == 3)
                gl.xlabel_style = {"size": 9}; gl.ylabel_style = {"size": 9}
                if ok.sum() >= 5:
                    v = sv[key][b][ok]
                    st = f"r={np.corrcoef(v, asos[ok])[0, 1]:.2f}  bias={np.mean(v - asos[ok]):+.1f}  n={ok.sum()}"
                else:
                    st = f"n={ok.sum()}"
                ax.text(0.02, 0.02, st, transform=ax.transAxes, fontsize=10, va="bottom", zorder=6,
                        bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=2))
                if r == 0:
                    ax.set_title(title, fontsize=14)
                if c == 0:
                    ax.text(-0.30, 0.5, f"{3 * b:02d}–{3 * b + 3:02d} KST", transform=ax.transAxes, rotation=90,
                            va="center", ha="center", fontsize=16, fontweight="bold")
            # 마지막 열: ASOS 관측소 위치만 (원만, 배경 래스터 없음)
            ax = axes[r, ncol - 1]
            ax.set_extent(EXT, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
            ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
            asc = ax.scatter(slon[ok], slat[ok], c=asos[ok], cmap=cmap, norm=norm, s=70, edgecolor="k", linewidth=0.7,
                              transform=ccrs.PlateCarree(), zorder=5)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
            gl.top_labels = False; gl.right_labels = False
            gl.left_labels = False; gl.bottom_labels = (r == 3)
            gl.xlabel_style = {"size": 9}; gl.ylabel_style = {"size": 9}
            ax.text(0.02, 0.02, f"n={ok.sum()}", transform=ax.transAxes, fontsize=10, va="bottom", zorder=6,
                    bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=2))
            if r == 0:
                ax.set_title("ASOS stations", fontsize=14)
        cb = fig.colorbar(sc if SHOW_DOTS else im, ax=axes, orientation="horizontal", fraction=0.02, pad=0.03, extend="both", ticks=LEV)
        cb.set_label("3-hour precipitation (mm/3h)" + ("  |  circles = ASOS 3-h total (ASOS[T] <-> HSR[T-1h], lag 1h)" if SHOW_DOTS else ""), fontsize=13)
        fig.suptitle(f"{date} (KST), {tag.replace('_', '-')}h: 3-hourly accumulation, 1 km input -> x2/x4/x8/x16 + ASOS (zoom on heavy-rain window, Val 2021 JJA, ep {ep})",
                     fontsize=18, y=0.995)
        out = f"{HERE}/case_studies/3hourly/case_3h_{date}_2x_multires_pow2{TAG}_asoscol_{tag}.png"
        fig.savefig(out, dpi=70, bbox_inches="tight"); plt.close(fig); print("saved", out)
