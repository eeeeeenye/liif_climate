"""
case_hsr_2x_multires.py 산출 npz 로 슬라이드(High-Resolution Precipitation Reconstruction) 오른쪽 그림 형식의 HSR 버전.
  창(호우 중심) 2x3:  위 = LR (1km) | GT (500m, raw HSR) | LIIF 500m (x2, 학습 배율)
                       아래 = LIIF 200m (x5) | LIIF 100m (x10) | (컬러바)
  200m/100m 는 GT 가 없다 (HSR 원자료가 500m). 학습 배율은 x2 뿐 -> x5, x10 은 외삽.
  산출: 일누적(24h) 창 그림 / 피크 시각 1시간 창 그림 / 전 도메인 개관(일누적+시간, LR | GT | LIIF 500m + 창 표시).
  정량(stdout): 창에서 LIIF 500m vs GT, LIIF 100m 를 5x5 평균해 500m 로 내린 것 vs GT.
사용: python plot_case_hsr_2x_multires.py 2021-07-06
env: liif (cartopy)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
PC = ccrs.PlateCarree()
CMAP = plt.cm.turbo.copy(); CMAP.set_bad("white")
VMAX = {"d": 200.0, "h": 30.0}                                   # 일누적 mm/day, 시간 mm/h (창 안 GT 최대 ~260 / 49)
UNIT = {"d": "Precipitation (mm/day)", "h": "Precipitation (mm/hour)"}
dates = sys.argv[1:] or ["2021-07-06"]


def block_mean(a, k):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean((1, 3))


def stats(name, p, g, m):
    e = (p - g)[m]
    print(f"    {name:24s} r={np.corrcoef(p[m], g[m])[0, 1]:.3f}  RMSE={np.sqrt((e ** 2).mean()):.3f}  bias={e.mean():+.3f}  (GT mean {g[m].mean():.3f})")


def show(ax, fld, ext, vm, mask=None):
    f = fld.astype(np.float32).copy()
    if mask is not None:
        f = np.where(mask, f, np.nan)
    return ax.imshow(f, origin="lower", extent=ext, transform=PC, cmap=CMAP, vmin=0, vmax=vm, interpolation="nearest", aspect="auto")


def decorate(ax, title, ext, left, bottom):
    ax.set_extent(ext, crs=PC)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = gl.right_labels = False; gl.left_labels = left; gl.bottom_labels = bottom
    gl.xlabel_style = gl.ylabel_style = {"size": 9}
    ax.set_title(title, fontsize=13)


for date in dates:
    d = np.load(f"{HERE}/case_studies/multires/case_multires_{date}.npz")
    r0, r1, c0, c1 = [int(v) for v in d["region"]]; ph = int(d["peak_hour"]); ep = int(d["ep"])
    lat, lon = d["lat"], d["lon"]
    dlat, dlon = np.diff(lat).mean(), np.diff(lon).mean()
    ext = [lon[c0] - dlon / 2, lon[c1 - 1] + dlon / 2, lat[r0] - dlat / 2, lat[r1 - 1] + dlat / 2]      # 창 (LR/500m/200m/100m 전부 같은 상자)
    ext_full = [lon[0] - dlon / 2, lon[-1] + dlon / 2, lat[0] - dlat / 2, lat[-1] + dlat / 2]
    valid = d["valid"][r0:r1, c0:c1]
    up = lambda m, f: np.kron(m, np.ones((f, f), bool))                                              # 500m 마스크 -> 200m/100m (최근접)
    valid2 = np.repeat(np.repeat(valid, 5, 0), 5, 1)[::2, ::2][:d["d_m200"].shape[0], :d["d_m200"].shape[1]]  # 200m = 2.5x: 5배 확대 후 2 간격
    valid5 = up(valid, 5)

    for kind, pre, label in [("d", "d_", f"{date} daily accumulation (KST day, 24 h)"),
                             ("h", "h_", f"{date} {ph:02d}:00-{ph + 1:02d}:00 KST (hourly)")]:
        gt = d[pre + "gt"][r0:r1, c0:c1]; m500 = d[pre + "m500"]
        lr = d[pre + "lr"][r0 // 2:r1 // 2, c0 // 2:c1 // 2]
        print(f"\n=== {label}  [ckpt epoch {ep}, 창 lat {ext[2]:.2f}~{ext[3]:.2f} lon {ext[0]:.2f}~{ext[1]:.2f}] ===")
        stats("LIIF 500m vs GT 500m", m500, gt, valid)
        stats("LIIF 100m(5x5 avg) vs GT", block_mean(d[pre + "m100"].astype(np.float32), 5), gt, valid)
        stats("LR 1km (x2 nearest) vs GT", np.kron(lr, np.ones((2, 2), np.float32)), gt, valid)

        vm = VMAX[kind]
        panels = [("LR image (1 km, 175 x 171)", lr, None),
                  ("GT (500 m, raw HSR)", gt, valid),
                  (f"LIIF (500 m)  [x2, trained]", m500, valid),
                  (f"LIIF (200 m)  [x5, extrapolated]", d[pre + "m200"], valid2),
                  (f"LIIF (100 m)  [x10, extrapolated]", d[pre + "m100"], valid5)]
        fig, axes = plt.subplots(2, 3, figsize=(17, 11.2), subplot_kw={"projection": PC})
        for i, (ttl, fld, mk) in enumerate(panels):
            ax = axes[i // 3, i % 3]
            im = show(ax, fld, ext, vm, mk)
            decorate(ax, ttl, ext, left=(i % 3 == 0), bottom=(i // 3 == 1))
        axes[1, 2].axis("off")
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        cax = fig.add_axes([0.70, 0.27, 0.22, 0.022])                                                  # 빈 6번째 칸에 컬러바
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max")
        cb.set_label(UNIT[kind], fontsize=13)
        fig.suptitle(f"HSR 2x LIIF: 1 km input -> 500 / 200 / 100 m  |  {label}  |  val 2021 (out-of-sample)", fontsize=16, y=0.985)
        out = f"{HERE}/case_studies/multires/case_multires_{date}_{'daily' if kind == 'd' else f'hourly{ph:02d}'}.png"
        fig.savefig(out, dpi=90, bbox_inches="tight"); plt.close(fig); print("saved", out)

    # 전 도메인 개관: 위 = 일누적, 아래 = 피크 시각.  LR | GT | LIIF 500m,  창 표시
    fig, axes = plt.subplots(2, 3, figsize=(17, 12.5), subplot_kw={"projection": PC})
    lr_ext = [d["lr_lon"][0] - np.diff(d["lr_lon"]).mean() / 2, d["lr_lon"][-1] + np.diff(d["lr_lon"]).mean() / 2,
              d["lr_lat"][0] - np.diff(d["lr_lat"]).mean() / 2, d["lr_lat"][-1] + np.diff(d["lr_lat"]).mean() / 2]
    for row, (kind, pre) in enumerate([("d", "d_"), ("h", "h_")]):
        vm = VMAX[kind]
        cols = [("LR image (1 km)", d[pre + "lr"], lr_ext, d["lr_valid"]), ("GT (500 m, raw HSR)", d[pre + "gt"], ext_full, d["valid"]),
                ("LIIF (500 m)  [x2]", d[pre + "full500"], ext_full, d["valid"])]
        for c, (ttl, fld, ex, mk) in enumerate(cols):
            ax = axes[row, c]
            im = show(ax, fld, ex, vm, mk)
            decorate(ax, ttl + ("   " + ("daily" if kind == "d" else f"{ph:02d}-{ph + 1:02d} KST")) if c == 0 else ttl, ext_full, left=(c == 0), bottom=(row == 1))
            ax.plot([ext[0], ext[1], ext[1], ext[0], ext[0]], [ext[2], ext[2], ext[3], ext[3], ext[2]], "k-", lw=1.3, transform=PC, zorder=5)
        cb = fig.colorbar(im, ax=axes[row], orientation="vertical", fraction=0.02, pad=0.02, extend="max"); cb.set_label(UNIT[kind], fontsize=12)
    fig.suptitle(f"{date}: whole domain (box = zoom window of the 200 / 100 m figures)  |  val 2021", fontsize=16, y=0.93)
    out = f"{HERE}/case_studies/multires/case_multires_{date}_overview.png"
    fig.savefig(out, dpi=80, bbox_inches="tight"); plt.close(fig); print("saved", out)
