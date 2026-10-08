"""case_3hourly_hsr_8x_lte_wmae.py 산출 npz 로 그림 1장: 행 = 일합 / 최대강수 3시간 구간, 열 = Raw 500m | Input 4km | LTE base | LTE+WMAE. 원 = ASOS (같은 색)."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs, cartopy.feature as cf
date = sys.argv[1]
D = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/case_studies/3hourly"
z = np.load(f"{D}/case_3h_{date}_8x_lte_wmae.npz")
LEV = [1, 2, 5, 10, 20, 30, 50, 80, 100, 150]
cmap = ListedColormap(["white"] + list(plt.cm.turbo(np.linspace(0.08, 0.98, len(LEV) - 1))) + ["#4a0a2a"])
norm = BoundaryNorm(LEV, cmap.N, extend="both")
valid = z["valid"]; b_pk = int(np.nanargmax([np.nansum(z["raw"][b]) for b in range(8)]))
rows = [("Daily sum", slice(None), np.nansum(z["st_asos"], 0)),
        (f"3h peak ({3*b_pk:02d}-{3*b_pk+3:02d} KST)", b_pk, z["st_asos"][b_pk])]
cols = [("HSR raw 500m", "raw", z["lat"], z["lon"]), ("Input 4km", "inp", z["lr_lat"], z["lr_lon"]),
        (f"LTE base (ep{int(z['ep_pre'])})", "pre", z["lat"], z["lon"]), (f"LTE + WMAE ft (ep{int(z['ep_ft'])})", "prd", z["lat"], z["lon"])]
fig, axs = plt.subplots(2, 4, figsize=(17, 9.5), subplot_kw=dict(projection=ccrs.PlateCarree()))
for r, (rn, sl, ast) in enumerate(rows):
    for c, (cn, key, la, lo) in enumerate(cols):
        a = z[key][sl]; a = a.sum(0) if a.ndim == 3 else a
        if key != "inp": a = np.where(valid, a, np.nan)
        ax = axs[r, c]; ax.set_extent([124.2, 131.7, 32.7, 38.8])
        im = ax.pcolormesh(lo, la, a, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto")
        ax.add_feature(cf.COASTLINE, lw=0.5); ax.add_feature(cf.BORDERS, lw=0.3)
        ok = np.isfinite(ast); ax.scatter(z["st_lon"][ok], z["st_lat"][ok], c=ast[ok], cmap=cmap, norm=norm, s=22, edgecolors="k", linewidths=0.4, transform=ccrs.PlateCarree())
        t = f"{date} {rn}\n{cn}"
        if key in ("pre", "prd"):
            ii = [np.abs(la - y).argmin() for y in z["st_lat"][ok]]; jj = [np.abs(lo - x).argmin() for x in z["st_lon"][ok]]
            m = np.array([a[i, j] for i, j in zip(ii, jj)]); o = ast[ok]; g = np.isfinite(m)
            t += f"\nvs ASOS: r={np.corrcoef(m[g], o[g])[0,1]:.2f} bias={np.mean(m[g]-o[g]):+.1f} mm"
        ax.set_title(t, fontsize=9)
cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.03, pad=0.04, aspect=50); cb.set_label("Precipitation (mm)"); cb.set_ticks(LEV)
out = f"{D}/case_3h_{date}_8x_lte_wmae.png"; fig.savefig(out, dpi=110, bbox_inches="tight"); print(out)
