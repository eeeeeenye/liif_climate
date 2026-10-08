"""
case_3hourly_hsr_2x_multires_pow2.py 산출 npz 로 3시간 누적 지도 (호우 창 확대, 배율 비교).  날짜별 2장 (00-12시 / 12-24시), 장마다 4개 시간대(행) x 6열.
열: Raw HSR (500m) | Input (1km, 2x LR) | LIIF x2 (500m, 학습 배율) | x4 (250m) | x8 (125m) | x16 (62.5m).  전부 같은 지리 범위 -> 화소 크기 차이가 배율.
원 = ASOS 3시간합 (같은 색; 창 안 + 그 구간에 기록 있는 지점만). 패널 좌하단 = 창 안 관측소 위치에서 r / bias / n (500m 외 나머지는 GT 가 없어 ASOS 로만 비교).
사용: python plot_case_3hourly_hsr_2x_multires_pow2.py 2021-07-06
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
TAG = os.environ.get("TAG", "")                       # npz/png 접미사 (case_3hourly_hsr_2x_multires_pow2.py 의 TAG 와 동일)
LABEL = os.environ.get("LABEL", "LIIF")               # 모델 열 제목 접두어 (예: "MLP ft")
SHOW_DOTS = os.environ.get("DOTS", "1") != "0"         # ASOS 관측소 산점(원) 표시 여부. r/bias 통계 텍스트는 DOTS=0 이어도 그대로 표시
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
    ncol = len(cols)
    inb = (d["st_lon"] >= EXT[0]) & (d["st_lon"] <= EXT[1]) & (d["st_lat"] >= EXT[2]) & (d["st_lat"] <= EXT[3])
    slat, slon, sasos = d["st_lat"][inb], d["st_lon"][inb], d["st_asos"][:, inb]
    sv = {key: np.stack([at(f[b], fl, fo, slat, slon) for b in range(8)]) for _, key, f, fl, fo, _ in cols}      # (8, n_in)
    print(f"{date}: 창 안 ASOS 지점 {inb.sum()}개")
    for b in range(8):
        ok = np.isfinite(sasos[b])
        if ok.sum() >= 5:
            print(f"  {3 * b:02d}-{3 * b + 3:02d} KST n={ok.sum():2d}  " + "  ".join(
                f"{key}: r={np.corrcoef(sv[key][b][ok], sasos[b][ok])[0, 1]:.2f} b={np.mean(sv[key][b][ok] - sasos[b][ok]):+.1f}" for _, key, *_ in cols))
    okall = np.isfinite(sasos)
    print("  하루 풀링 n=%d  " % okall.sum() + "  ".join(
        f"{key}: r={np.corrcoef(sv[key][okall], sasos[okall])[0, 1]:.2f} b={np.mean(sv[key][okall] - sasos[okall]):+.1f}" for _, key, *_ in cols))

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
        cb = fig.colorbar(sc if SHOW_DOTS else im, ax=axes, orientation="horizontal", fraction=0.02, pad=0.03, extend="both", ticks=LEV)
        cb.set_label("3-hour precipitation (mm/3h)" + ("  |  circles = ASOS 3-h total (ASOS[T] <-> HSR[T-1h], lag 1h)" if SHOW_DOTS else ""), fontsize=13)
        fig.suptitle(f"{date} (KST), {tag.replace('_', '-')}h: 3-hourly accumulation, 1 km input -> x2/x4/x8/x16 (zoom on heavy-rain window, Val 2021 JJA, ep {ep})",
                     fontsize=18, y=0.995)
        out = f"{HERE}/case_studies/3hourly/case_3h_{date}_2x_multires_pow2{TAG}_{tag}.png"
        fig.savefig(out, dpi=70, bbox_inches="tight"); plt.close(fig); print("saved", out)
