"""
quantile_hsr_8x_mlp_ft_2021jja.npz 를 지도로: 2021 JJA(검증) 시간강수 90th/99th percentile, GT(raw HSR 500m) vs 8x LIIF(base) vs 8x MLP ft(adapter).
  2행(P90/P99) x 3열(GT/Base/FT). 행마다 색 구간 다름(P99 가 P90 보다 값이 훨씬 큼).
사용: python plot_quantile_hsr_8x_mlp_ft_2021jja.py
env: liif (cartopy)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
d = np.load(f"{HERE}/station_eval/preds_npz/quantile_hsr_8x_mlp_ft_2021jja.npz")
lat, lon, valid = d["lat"], d["lon"], d["valid"]
EXT = [lon[0], lon[-1], lat[0], lat[-1]]

LEV90 = [0.1, 0.5, 1, 2, 3, 5, 8, 12, 18]
LEV99 = [1, 2, 5, 10, 15, 20, 30, 45, 65]


def cmap_norm(lev):
    cm = ListedColormap(["white"] + list(plt.cm.turbo(np.linspace(0.08, 0.98, len(lev) - 1))) + ["#4a0a2a"])
    return cm, BoundaryNorm(lev, cm.N, extend="both")


cols = [("Raw HSR (GT)", "gt"), (f"8x LIIF (base, ep {int(d['ep_base'])})", "base"), (f"8x MLP ft (adapter, ep {int(d['ep_ft'])})", "ft")]
rows = [("90th pct", 90, LEV90), ("99th pct", 99, LEV99)]

fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), subplot_kw={"projection": ccrs.PlateCarree()})
for r, (rtitle, q, lev) in enumerate(rows):
    cm, norm = cmap_norm(lev)
    for c, (title, key) in enumerate(cols):
        ax = axes[r, c]
        ax.set_extent(EXT, crs=ccrs.PlateCarree())
        arr = d[f"q{q}_{key}"].copy()
        arr[~valid] = np.nan
        cmx = cm.copy(); cmx.set_bad("white")
        im = ax.imshow(arr, origin="lower", extent=EXT, transform=ccrs.PlateCarree(), cmap=cmx, norm=norm,
                        interpolation="nearest", aspect="auto")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        gl.top_labels = False; gl.right_labels = False
        gl.left_labels = (c == 0); gl.bottom_labels = (r == 1)
        gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}
        if r == 0:
            ax.set_title(title, fontsize=13)
        if c == 0:
            ax.text(-0.28, 0.5, rtitle, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=14, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.045, pad=0.02, extend="both", ticks=lev)
        cb.ax.tick_params(labelsize=8)
fig.suptitle(f"2021 JJA (val, {int(d['n_hours'])}h) hourly precip P90/P99 (mm/h): GT vs 8x LIIF vs 8x MLP ft", fontsize=15, y=0.995)
out = f"{HERE}/station_eval/figures/quantile_hsr_8x_mlp_ft_2021jja.png"
fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
print("saved", out)
