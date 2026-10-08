"""
case_3hourly_hsr_lag1h_ext.py 산출 npz 로 특정 지역 확대 3시간 누적 지도 + 그 범위 안 ASOS 지점 r/bias.
열: Raw HSR(500m) | Input(1km) | Radar pretrain | MLP ft(MSE).  원 = ASOS 3시간합 (범위 안 지점만).
통계(r, bias, n)는 범위 안 지점만으로 계산해 표준출력(구간별 + 하루 전체 8구간x지점 풀링)에 찍고, 각 패널 좌하단에도 그 구간의 r / bias / n 을 적는다 (지점 3개 미만이면 n 만).
사용: python plot_case_3hourly_zoom.py gangnam_full osong_full   # 남한 전체
      python plot_case_3hourly_zoom.py gangnam   # 2022-08-08 강남
      python plot_case_3hourly_zoom.py osong     # 2023-07-15 오송
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
# 다른 모델의 ext 산출물(같은 키: raw/inp/pre/prd/ep_ft)을 그릴 때 환경변수로 지정. 기본값 = 2x 기존 그림.
NPZ_TAG = os.environ.get("NPZ_TAG", "lag1h_ext")
OUT_TAG = os.environ.get("OUT_TAG", "")                                   # 출력 파일명 접미사 (예: _8x_lte)
T_INP = os.environ.get("T_INP", "Input (1 km, 2x LR)")
T_PRE = os.environ.get("T_PRE", "Radar pretrain (500 m)")
SHOW_STATS = os.environ.get("STATS", "1") != "0"                          # STATS=0 이면 패널 좌하단 r/bias/n 라벨을 그리지 않음 (표준출력 통계는 그대로)
SHOW_DOTS = os.environ.get("DOTS", "0") != "0"                            # ASOS 관측소 산점(원) 표시 여부. star(관측지점) 는 항상 표시
DOT_S = float(os.environ.get("DOT_S", "110"))                               # zoom 그림 ASOS 점 크기 (matplotlib s). 오송 옆 두 지점 겹침 방지는 ~50
T_FT = os.environ.get("T_FT", "MLP ft (MSE, ep {ep}) (500 m)")            # {ep} = npz 의 ep_ft
CASES = {
    # 이름: (날짜, 표시 이름, 중심(lat, lon), 범위 [lon0, lon1, lat0, lat1])
    "gangnam": ("2022-08-08", "Gangnam", (37.4979, 127.0276), [126.5, 127.55, 37.15, 37.85]),
    "osong": ("2023-07-15", "Osong (approx.)", (36.6200, 127.3300), [126.78, 127.88, 36.25, 37.0]),
    # 오송 주변 ASOS 약 10개가 들어오도록 넓힌 zoom (lat ±0.6, lon ±0.75)
    "osong_wide": ("2023-07-15", "Osong (approx.)", (36.6200, 127.3300), [126.58, 128.08, 36.02, 37.22]),
    # 남한 전체 (plot_case_3hourly_new.py 와 같은 범위, 전 지점)
    "gangnam_full": ("2022-08-08", "Gangnam", (37.4979, 127.0276), [124.2, 131.7, 32.7, 38.8]),
    "osong_full": ("2023-07-15", "Osong (approx.)", (36.6200, 127.3300), [124.2, 131.7, 32.7, 38.8]),
}
# 원래 turbo 색감 유지. 로그 등간격(비율 ~1.35~1.5)으로 저강수~고강수 전 구간이 고르게 세분되도록 설계
# (케이스마다 강수 강도 분포가 달라 특정 구간에 맞춰 촘촘히 짜면 다른 케이스에서 반대쪽이 뭉개짐)
LEV = [1, 1.4, 2, 3, 4, 6, 8, 12, 16, 22, 30, 42, 60, 85, 120, 150]
NOBG = os.environ.get("NOBG", "0") != "0"                                 # NOBG=1 이면 강수 없는(< LEV[0]) 영역을 흰색 대신 완전 투명으로 표시
under = (1, 1, 1, 0) if NOBG else "white"
cmap = ListedColormap([under] + list(plt.cm.turbo(np.linspace(0.08, 0.98, len(LEV) - 1))) + ["#4a0a2a"])
norm = BoundaryNorm(LEV, cmap.N, extend="both")


def at_stations(field, flat, flon, slat, slon):
    r = np.abs(flat[None, :] - slat[:, None]).argmin(1); c = np.abs(flon[None, :] - slon[:, None]).argmin(1)
    return field[r, c]


def crop(fld, flat, flon, ext, pad=0.05):
    ri = np.where((flat >= ext[2] - pad) & (flat <= ext[3] + pad))[0]; ci = np.where((flon >= ext[0] - pad) & (flon <= ext[1] + pad))[0]
    return fld[..., ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1], flat[ri[0]:ri[-1] + 1], flon[ci[0]:ci[-1] + 1]


def corr(v, o):
    return np.corrcoef(v, o)[0, 1] if len(v) >= 3 and v.std() > 0 and o.std() > 0 else np.nan


for key in sys.argv[1:] or ["gangnam", "osong"]:
    date, name, (clat, clon), EXT = CASES[key]
    full = key.endswith("_full")           # 전체 범위: 원/별 크기를 줄이고 패널 높이를 원본과 같게
    d = np.load(f"{HERE}/case_studies/3hourly/case_3h_{date}_{NPZ_TAG}.npz")
    valid = d["valid"]; ep = int(d["ep_ft"])
    inbox = (d["st_lon"] >= EXT[0]) & (d["st_lon"] <= EXT[1]) & (d["st_lat"] >= EXT[2]) & (d["st_lat"] <= EXT[3])
    slat, slon, sasos = d["st_lat"][inbox], d["st_lon"][inbox], d["st_asos"][:, inbox]     # (8, n_in)
    cols = [("Raw HSR (500 m)", d["raw"], d["lat"], d["lon"], True),
            (T_INP, d["inp"], d["lr_lat"], d["lr_lon"], False),
            (T_PRE, d["pre"], d["lat"], d["lon"], True),
            (T_FT.format(ep=ep), d["prd"], d["lat"], d["lon"], True)]
    # 지점별 값 (전 구간) 미리 추출 -> (col, 8, n_in)
    stv = np.stack([np.stack([at_stations(np.nan_to_num(f[b]), la, lo, slat, slon) for b in range(8)]) for _, f, la, lo, _ in cols])

    print(f"\n=== {date} {name}: 범위 {EXT}, ASOS 지점 {inbox.sum()}개 ===")
    for la, lo in zip(slat, slon):
        print(f"   지점 lat {la:.3f} lon {lo:.3f}")
    hdr = "  구간(KST)  " + " | ".join(f"{c[0][:22]:>22s}" for c in cols)
    print(hdr)
    for b in range(8):
        ok = np.isfinite(sasos[b])
        cells = []
        for c in range(4):
            if ok.sum() >= 3:
                cells.append(f"r={corr(stv[c, b][ok], sasos[b][ok]):+.2f} b={np.mean(stv[c, b][ok] - sasos[b][ok]):+5.1f} n={ok.sum()}")
            else:
                cells.append(f"n={ok.sum()}")
        print(f"  {3 * b:02d}-{3 * b + 3:02d}      " + " | ".join(f"{s:>22s}" for s in cells))
    okall = np.isfinite(sasos)
    cells = [f"r={corr(stv[c][okall], sasos[okall]):+.2f} b={np.mean(stv[c][okall] - sasos[okall]):+5.1f} n={okall.sum()}" for c in range(4)]
    print("  하루 풀링    " + " | ".join(f"{s:>22s}" for s in cells))
    # 강한 비 구간만 (ASOS 3h >= 10mm)
    okh = okall & (sasos >= 10)
    if okh.sum() >= 3:
        cells = [f"r={corr(stv[c][okh], sasos[okh]):+.2f} b={np.mean(stv[c][okh] - sasos[okh]):+5.1f} n={okh.sum()}" for c in range(4)]
        print("  ASOS>=10mm   " + " | ".join(f"{s:>22s}" for s in cells))

    for part, (b0, tag) in enumerate([(0, "00-12"), (4, "12-24")]):
        fig, axes = plt.subplots(4, 4, figsize=(22, 4 * (4.1 if full else 3.9) + 1.6), subplot_kw={"projection": ccrs.PlateCarree()})
        for r in range(4):
            b = b0 + r
            asos = sasos[b]; ok = np.isfinite(asos)
            for c, (title, fld, flat, flon, hi) in enumerate(cols):
                ax = axes[r, c]
                ax.set_extent(EXT, crs=ccrs.PlateCarree())
                show = fld[b].astype(float).copy()
                if hi:
                    show[~valid] = np.nan
                if c == 3:
                    show[~(d["raw"][b] >= LEV[0])] = np.nan          # plot_case_3hourly_new.py 와 같은 배경 마스킹 (통계엔 영향 없음)
                show, cla, clo = crop(show, flat, flon, EXT)
                mesh = ax.pcolormesh(clo, cla, show, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
                if SHOW_DOTS:
                    sc = ax.scatter(slon[ok], slat[ok], c=asos[ok], cmap=cmap, norm=norm, s=30 if full else DOT_S, edgecolor="k", linewidth=0.5 if full else 1.1,
                                    transform=ccrs.PlateCarree(), zorder=5)
                if SHOW_STATS:
                    if ok.sum() >= 3:
                        st = f"r={corr(stv[c, b][ok], asos[ok]):.2f}  bias={np.mean(stv[c, b][ok] - asos[ok]):+.1f}  n={ok.sum()}"
                    else:
                        st = f"n={ok.sum()}"
                    ax.text(0.02, 0.02, st, transform=ax.transAxes, fontsize=10, va="bottom", zorder=7,
                            bbox=dict(fc="white", ec="0.6", alpha=0.85, pad=2))
                ax.plot(clon, clat, marker="*", color="magenta", ms=8 if full else 11, mec="k", mew=0.8, transform=ccrs.PlateCarree(), zorder=6)
                gl = ax.gridlines(draw_labels=(c == 0), linewidth=0.3, alpha=0.4)
                gl.top_labels = False; gl.right_labels = False; gl.bottom_labels = (r == 3)
                gl.xlabel_style = {"size": 9}; gl.ylabel_style = {"size": 9}
                if r == 0:
                    ax.set_title(title, fontsize=14)
                if c == 0:
                    ax.text(-0.30, 0.5, f"{3 * b:02d}–{3 * b + 3:02d} KST", transform=ax.transAxes, rotation=90, va="center", ha="center",
                            fontsize=16, fontweight="bold")
        cb = fig.colorbar(sc if SHOW_DOTS else mesh, ax=axes, orientation="horizontal", fraction=0.02, pad=0.03, extend="both", ticks=LEV)
        cb.set_label("3-hour precipitation (mm/3h)" + ("  |  circles = ASOS 3-h total (ASOS[T] <-> HSR[T-1h], lag 1h)" if SHOW_DOTS else "") + "  |  star = " + name, fontsize=13)
        where = f"South Korea ({name} marked)" if full else f"zoom on {name}"
        fig.suptitle(f"{date} (KST), {tag}h, {where}: 3-hourly accumulation (out-of-sample date)", fontsize=18, y=0.995)
        pre = "" if full else "zoom_"
        out = f"{HERE}/case_studies/3hourly/case_3h_{date}_{pre}{key}{OUT_TAG}_{tag.replace('-', '_')}.png"
        fig.savefig(out, dpi=70, bbox_inches="tight", transparent=NOBG)
        plt.close(fig)
        print("saved", out)
