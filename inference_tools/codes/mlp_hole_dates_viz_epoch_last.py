import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fm.fontManager.addfont("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

path_raw = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_01deg_1998_202504_noleaf.nc"
path_test = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax_ft_mlp/preds_nc/preds_not_merged_test/pred_liif_minmax_ft_01deg_1998_2024.nc"
path_epoch_last = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-4_drop150/preds_nc/preds_check_epoch-last/pred_liif_mlp_1e-4_epoch-last_01deg_1998_2024.nc"

# ---- mlp adapter가 극한강수 픽셀에서 0으로 꺼지는("구멍") 판정 기준 ----
HOLE_RAW_THRESH = 100.0
HOLE_MLP_THRESH = 1.0

# 예전 mlp_hole_dates_viz.py 스캔에서 찾았던 "test" 버전의 구멍 뚫린 날짜 (동일하게 사용)
target_dates = [
    "1998-08-05", "2003-07-20", "2012-08-23", "2015-07-07", "2016-09-16",
    "2017-10-21", "2018-07-01", "2018-08-26", "2021-06-10", "2023-08-09",
    "2023-08-31", "2024-09-12", "2024-09-21",
]

raw_f = Dataset(path_raw)
test_f = Dataset(path_test)
last_f = Dataset(path_epoch_last)

lon_min, lon_max = 124, 132
lat_min, lat_max = 33, 43


def get_time_strs(nc_f):
    time_var = nc_f.variables["time"]
    times = num2date(time_var[:], units=time_var.units,
                      calendar=getattr(time_var, "calendar", "standard"))
    return np.array([t.strftime("%Y-%m-%d") for t in times])


def get_lat_lon_idx(nc_f, lat_name, lon_name, lon_range, lat_range):
    lat_full = np.array(nc_f.variables[lat_name][:])
    lon_full = np.array(nc_f.variables[lon_name][:])

    lat_idx = np.where((lat_full >= lat_range[0]) & (lat_full <= lat_range[1]))[0]
    lon_idx = np.where((lon_full >= lon_range[0]) & (lon_full <= lon_range[1]))[0]

    lat_sub = lat_full[lat_idx[0]:lat_idx[-1] + 1]
    lon_sub = lon_full[lon_idx[0]:lon_idx[-1] + 1]

    return lat_idx, lon_idx, lat_sub, lon_sub


def load_precip_slice(nc_f, var_name, idx, lat_idx, lon_idx, flip_lat):
    precip = nc_f.variables[var_name][
        idx,
        lat_idx[0]:lat_idx[-1] + 1,
        lon_idx[0]:lon_idx[-1] + 1
    ]
    precip = np.ma.filled(precip, np.nan)
    if flip_lat:
        precip = precip[::-1, :]
    return precip


raw_lat_name = "lat" if "lat" in raw_f.variables else "latitude"
raw_lon_name = "lon" if "lon" in raw_f.variables else "longitude"

raw_lat_idx, raw_lon_idx, raw_lat_sub, raw_lon_sub = get_lat_lon_idx(
    raw_f, raw_lat_name, raw_lon_name, [lon_min, lon_max], [lat_min, lat_max]
)
test_lat_idx, test_lon_idx, test_lat_sub, test_lon_sub = get_lat_lon_idx(
    test_f, "lat", "lon", [lon_min, lon_max], [lat_min, lat_max]
)
last_lat_idx, last_lon_idx = test_lat_idx, test_lon_idx

raw_flip = raw_lat_sub[0] > raw_lat_sub[-1]
test_flip = test_lat_sub[0] > test_lat_sub[-1]
last_flip = test_flip

if raw_flip:
    raw_lat_sub = raw_lat_sub[::-1]
if test_flip:
    test_lat_sub = test_lat_sub[::-1]

raw_time_strs = get_time_strs(raw_f)
test_time_strs = get_time_strs(test_f)
last_time_strs = get_time_strs(last_f)


def make_bounds_5(vmax):
    upper = int(np.ceil(vmax / 5.0) * 5)
    return list(range(0, upper + 5, 5))


cmap = plt.get_cmap("turbo")

fig, axes = plt.subplots(
    len(target_dates), 3,
    figsize=(18, 5 * len(target_dates)),
    subplot_kw={"projection": ccrs.PlateCarree()}
)


def plot_panel(ax, precip, lon, lat, title, panel_cmap, panel_norm, panel_bounds,
               cbar_label, hole_xy=None):
    ax.set_extent([raw_lon_sub.min(), raw_lon_sub.max(), raw_lat_sub.min(), raw_lat_sub.max()],
                   crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")

    mesh = ax.pcolormesh(
        lon, lat, precip,
        transform=ccrs.PlateCarree(),
        cmap=panel_cmap,
        norm=panel_norm,
        shading="auto"
    )

    ax.set_title(title, fontsize=11)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ticks_20 = [b for b in panel_bounds if b % 20 == 0]
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical",
                         shrink=0.8, pad=0.05,
                         ticks=ticks_20, extend="max")
    cbar.set_label(cbar_label)


for date_i, date in enumerate(target_dates):
    idx_raw = int(np.where(raw_time_strs == date)[0][0])
    idx_test = int(np.where(test_time_strs == date)[0][0])
    idx_last = int(np.where(last_time_strs == date)[0][0])

    precip_raw = load_precip_slice(
        raw_f, "precipitation", idx_raw, raw_lat_idx, raw_lon_idx, raw_flip
    )
    precip_test = load_precip_slice(
        test_f, "precip", idx_test, test_lat_idx, test_lon_idx, test_flip
    )
    precip_last = load_precip_slice(
        last_f, "precip", idx_last, last_lat_idx, last_lon_idx, last_flip
    )

    hole_mask = (precip_raw > HOLE_RAW_THRESH) & (precip_test < HOLE_MLP_THRESH)
    ys, xs = np.where(hole_mask)
    hole_lat = test_lat_sub[ys]
    hole_lon = test_lon_sub[xs]

    vmax = np.ceil(np.nanmax(precip_raw) / 5.0) * 5.0
    bounds = make_bounds_5(vmax)
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="max")

    panels = [
        (precip_raw, f"raw (0.1°)  |  {date}", None),
        (precip_test, f"test (epoch-best, 구멍 있음)  |  {date}", (hole_lat, hole_lon)),
        (precip_last, f"epoch_last (수정본, 구멍 복구)  |  {date}", (hole_lat, hole_lon)),
    ]

    for col, (precip, title, hole_xy) in enumerate(panels):
        plot_panel(axes[date_i, col], precip, raw_lon_sub, raw_lat_sub, title,
                   cmap, norm, bounds, "Precipitation (mm/day)", hole_xy=hole_xy)

plt.tight_layout()
out_path = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/adapter_diagnostics/mlp_hole_dates_epoch_last_fixed_no_subdiv_1e-3_last.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장 완료: {out_path}")

raw_f.close()
test_f.close()
last_f.close()
