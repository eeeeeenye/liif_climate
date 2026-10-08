"""
_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-3_tanh 예측 확인용.
빵꾸(hole) 났던 13일에 대해 raw(0.1° GT) | prediction 2열 지도.

모델을 다시 돌리지 않고, make_preds.py가 이미 저장해둔 전체 기간 예측 nc를
그대로 읽어서 해당 날짜만 슬라이스한다.

컬러 스케일: 행별 0.1° GT max 를 5로 올림 -> bounds 0..vmax -> BoundaryNorm turbo, 패널별 컬러바.
CKPT 만 바꾸면 preds_check_<CKPT>/ 아래 예측 nc를 자동으로 찾음
(해당 CKPT의 예측 nc가 make_preds.py로 미리 생성돼 있어야 함).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset, num2date

for _fp in ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------ 설정
RUN_DIR = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-3_tanh"
CKPT    = "epoch-best"          # "epoch-best" | "epoch-best-loss" | "epoch-last"

GT_NC   = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_01deg_1998_202504_noleaf.nc"
# make_preds.py 산출물 (preds_check_<CKPT>/ 아래, 파일명 접두사는 make_preds.py의 o_name 규칙)
PRED_NC = f"{RUN_DIR}/preds_nc/preds_check_{CKPT}/pred_liif_mlp_1e-4_{CKPT}_01deg_1998_2024.nc"

OUT_PNG = f"/home/inhye_yoo/deeplearning/inr/liif/inference_tools/adapter_diagnostics/mlp_hole_dates_ft_mlp_1e-3_tanh_{CKPT}.png"

LAT_MIN, LAT_MAX = 33, 43
LON_MIN, LON_MAX = 124, 132

DATES = [
    "1998-08-05", "2003-07-20", "2012-08-23", "2015-07-07", "2016-09-16",
    "2017-10-21", "2018-07-01", "2018-08-26", "2021-06-10", "2023-08-09",
    "2023-08-31", "2024-09-12", "2024-09-21",
]

if not os.path.exists(PRED_NC):
    raise FileNotFoundError(
        f"예측 nc가 없음: {PRED_NC}\n"
        f"먼저 make_preds.py를 CKPT_TAG='{CKPT}'로 실행해 예측 nc를 생성할 것."
    )

# ------------------------------------------------------------------ nc 헬퍼
def open_nc(path):
    """nc를 열고 (Dataset, 날짜 문자열 배열, lat, lon, 강수 변수명) 반환. lat/lon은 파일 네이티브 순서."""
    ds = Dataset(path)
    tv = ds.variables["time"]
    tarr = pd.to_datetime(num2date(
        tv[:], tv.units,
        calendar=getattr(tv, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )).normalize()
    date_strs = np.array([t.strftime("%Y-%m-%d") for t in tarr])

    lat_name = "lat" if "lat" in ds.variables else "latitude"
    lon_name = "lon" if "lon" in ds.variables else "longitude"
    lat = np.array(ds.variables[lat_name][:])
    lon = np.array(ds.variables[lon_name][:])

    var_name = next(v for v in ("precipitation", "precip") if v in ds.variables)
    return ds, date_strs, lat, lon, var_name


def region_idx(lat, lon):
    li = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    oi = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
    return li, oi


def slice_day(ds, var_name, date_strs, li, oi, date):
    hit = np.where(date_strs == date)[0]
    if len(hit) == 0:
        raise ValueError(f"{os.path.basename(ds.filepath())}에 {date} 없음")
    t_idx = int(hit[0])
    arr = np.ma.filled(
        ds.variables[var_name][t_idx, li[0]:li[-1] + 1, oi[0]:oi[-1] + 1], np.nan
    )
    return arr


gt_ds, gt_dates, gt_lat, gt_lon, gt_var = open_nc(GT_NC)
pr_ds, pr_dates, pr_lat, pr_lon, pr_var = open_nc(PRED_NC)

gt_li, gt_oi = region_idx(gt_lat, gt_lon)
pr_li, pr_oi = region_idx(pr_lat, pr_lon)

gt_lat_s, gt_lon_s = gt_lat[gt_li[0]:gt_li[-1] + 1], gt_lon[gt_oi[0]:gt_oi[-1] + 1]
pr_lat_s, pr_lon_s = pr_lat[pr_li[0]:pr_li[-1] + 1], pr_lon[pr_oi[0]:pr_oi[-1] + 1]

# GT와 예측 격자가 같아야 나란히 비교 가능
assert np.allclose(gt_lat_s, pr_lat_s) and np.allclose(gt_lon_s, pr_lon_s), \
    "GT와 예측 nc의 격자가 다름 — pcolormesh 좌표가 어긋남"

# pcolormesh는 lat 오름/내림 상관없이 좌표배열만 맞으면 되지만, 둘의 방향은 일치해야 함
assert (gt_lat_s[0] < gt_lat_s[-1]) == (pr_lat_s[0] < pr_lat_s[-1]), \
    "GT와 예측 nc의 lat 방향이 반대 — 한쪽을 flip 해야 함"

# ------------------------------------------------------------------ 그림
cmap = plt.get_cmap("turbo")

fig, axes = plt.subplots(len(DATES), 2, figsize=(12, 5 * len(DATES)),
                         subplot_kw={"projection": ccrs.PlateCarree()})

for r, date in enumerate(DATES):
    gt = slice_day(gt_ds, gt_var, gt_dates, gt_li, gt_oi, date)
    pr = slice_day(pr_ds, pr_var, pr_dates, pr_li, pr_oi, date)

    vmax = float(np.ceil(np.nanmax(gt) / 5.0) * 5.0)
    bounds = list(range(0, int(vmax) + 5, 5))
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="max")
    ticks = [b for b in bounds if b % 20 == 0]

    for c, (arr, lat, lon, ttl) in enumerate([
        (gt, gt_lat_s, gt_lon_s, "raw (0.1°)"),
        (pr, pr_lat_s, pr_lon_s, f"pred ft_mlp 1e-3 tanh {CKPT}"),
    ]):
        ax = axes[r, c]
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        m = ax.pcolormesh(lon, lat, arr, transform=ccrs.PlateCarree(),
                          cmap=cmap, norm=norm, shading="auto")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = gl.right_labels = False
        ax.set_title(f"{ttl}  |  {date}   (max {np.nanmax(arr):.0f})", fontsize=11)
        cb = plt.colorbar(m, ax=ax, orientation="vertical", shrink=0.8, pad=0.05,
                          ticks=ticks, extend="max")
        cb.set_label("Precipitation (mm/day)")

fig.suptitle(f"_ft_mlp_1e-3_tanh {CKPT} — 빵꾸 났던 13일 (예측 nc 직접 로드)", fontsize=13, y=0.997)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print("저장:", OUT_PNG)

gt_ds.close()
pr_ds.close()
 