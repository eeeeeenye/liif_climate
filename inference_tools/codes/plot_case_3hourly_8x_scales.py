"""
case_3hourly_hsr_8x_scales.py 산출 npz 로 3시간 누적 지도. 날짜별 2장 (00-12시 / 12-24시), 장마다 4개 시간대(행) x 4열.
열: Raw HSR(500m) | Input(1km, 2x LR) | 8x LIIF 출력 500m (x2) | 8x LIIF 출력 200m (x5). (4km 입력 결과를 그리려면 T_INP/S500/S200 환경변수)
원 = ASOS 3시간합 (같은 색; 그 구간에 기록이 있는 지점만). 패널 좌하단 = 관측소 위치에서 r / bias / n.
사용: python plot_case_3hourly_8x_scales.py 2021-07-06 2021-08-23
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
dates = sys.argv[1:] or ["2021-07-06", "2021-08-23"]
TAG = os.environ.get("TAG", "8x_scales_1km")            # npz/png 접미사
LABEL = os.environ.get("LABEL", "8x LIIF")          # 모델 열 제목 (예: "8x MLP ft")
T_INP = os.environ.get("T_INP", "Input (1 km, 2x LR)")
S500 = os.environ.get("S500", "x2")                 # 입력 대비 배율 표기
S200 = os.environ.get("S200", "x5")
SHOW_DOTS = os.environ.get("DOTS", "0") != "0"         # ASOS 관측소 산점(원) 표시 여부. r/bias/n 통계 텍스트는 항상 표시
EXTENT = [124.2, 131.7, 32.7, 38.8]
LEV = [1, 2, 5, 10, 20, 30, 50, 80, 100, 150]
cmap = ListedColormap(["white"] + list(plt.cm.turbo(np.linspace(0.08, 0.98, len(LEV) - 1))) + ["#4a0a2a"])
norm = BoundaryNorm(LEV, cmap.N, extend="both")


def box(lat, lon):
    """규칙 격자 중심 좌표 -> imshow extent (셀 가장자리)."""
    dy, dx = np.diff(lat).mean(), np.diff(lon).mean()
    return [lon[0] - dx / 2, lon[-1] + dx / 2, lat[0] - dy / 2, lat[-1] + dy / 2]


for date in dates:
    d = np.load(f"{HERE}/case_studies/3hourly/case_3h_{date}_{TAG}.npz")
    ep = int(d["ep"]); valid = d["valid"]; lat, lon = d["lat"], d["lon"]
    H2, W2 = d["m200"].shape[1:]
    # 200m 격자의 radar 유효영역 마스크 (500m valid 를 최근접으로 확대)
    ri = np.clip((np.arange(H2) * valid.shape[0] / H2).astype(int), 0, valid.shape[0] - 1)
    ci = np.clip((np.arange(W2) * valid.shape[1] / W2).astype(int), 0, valid.shape[1] - 1)
    valid2 = valid[np.ix_(ri, ci)]
    cols = [("Raw HSR (500 m)", "raw", d["raw"], box(lat, lon), valid),
            (T_INP, "inp", d["inp"], box(d["lr_lat"], d["lr_lon"]), None),
            (f"{LABEL} (ep {ep}): 500 m  [{S500}]", "m500", d["m500"], box(lat, lon), valid),
            (f"{LABEL} (ep {ep}): 200 m  [{S200}]", "m200", d["m200"], box(d["lat2"], d["lon2"]), valid2)]
    for tag, b0 in [("00_12", 0), ("12_24", 4)]:
        fig, axes = plt.subplots(4, 4, figsize=(22, 4 * 4.1 + 1.6), subplot_kw={"projection": ccrs.PlateCarree()})
        for r in range(4):
            b = b0 + r
            asos = d["st_asos"][b]; ok = np.isfinite(asos)
            for c, (title, key, arr, ext, vm) in enumerate(cols):
                ax = axes[r, c]
                ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
                show = arr[b].astype(np.float32)
                if vm is not None:
                    show = np.where(vm, show, np.nan)
                # 위도 오름차순 배열 -> origin='lower'.  규칙 격자라 pcolormesh 대신 imshow (200m 는 10M 셀)
                cmapx = cmap.copy(); cmapx.set_bad("white")
                im = ax.imshow(show, origin="lower", extent=ext, transform=ccrs.PlateCarree(), cmap=cmapx, norm=norm,
                               interpolation="nearest", aspect="auto", zorder=1)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
                if SHOW_DOTS:
                    sc = ax.scatter(d["st_lon"][ok], d["st_lat"][ok], c=asos[ok], cmap=cmap, norm=norm, s=30, edgecolor="k",
                                    linewidth=0.5, transform=ccrs.PlateCarree(), zorder=5)
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
                gl.top_labels = False; gl.right_labels = False
                gl.left_labels = (c == 0); gl.bottom_labels = (r == 3)
                gl.xlabel_style = {"size": 9}; gl.ylabel_style = {"size": 9}
                if ok.sum() >= 5:
                    v = d[f"sv_{key}"][b][ok]
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
        fig.suptitle(f"{date} (KST), {tag.replace('_', '-')}h: 3-hourly accumulation, raw / input / {LABEL} at 500 m and 200 m (Val 2021 JJA)",
                     fontsize=18, y=0.995)
        out = f"{HERE}/case_studies/3hourly/case_3h_{date}_{TAG}_{tag}.png"
        fig.savefig(out, dpi=70, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)
