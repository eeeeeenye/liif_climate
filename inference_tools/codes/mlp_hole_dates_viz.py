import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

path_raw = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_01deg_1998_202504_noleaf.nc"
path_minmax = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax/preds_nc/preds_not_merged_2019_2024/pred_liif_minmax_ft_01deg_1998_2024.nc"
path_minmax_ft = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax_ft/preds_nc/preds_not_merged_test/pred_liif_minmax_ft_01deg_1998_2024.nc"
path_minmax_mlp = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax_ft_mlp/preds_nc/preds_not_merged_test/pred_liif_minmax_ft_01deg_1998_2024.nc"

# ---- mlp adapter가 극한강수 픽셀에서 0으로 꺼지는("구멍") 판정 기준 ----
HOLE_RAW_THRESH = 100.0   # raw(0.1도 원본) 강수량이 이 값(mm/day)보다 크면서
HOLE_MLP_THRESH = 1.0     # mlp 예측이 이 값(mm/day)보다 작으면 "구멍"으로 판정

raw_f = Dataset(path_raw)
minmax_f = Dataset(path_minmax)
minmax_ft_f = Dataset(path_minmax_ft)
minmax_mlp_f = Dataset(path_minmax_mlp)

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
minmax_lat_idx, minmax_lon_idx, minmax_lat_sub, minmax_lon_sub = get_lat_lon_idx(
    minmax_f, "lat", "lon", [lon_min, lon_max], [lat_min, lat_max]
)
minmax_ft_lat_idx, minmax_ft_lon_idx = minmax_lat_idx, minmax_lon_idx
minmax_mlp_lat_idx, minmax_mlp_lon_idx = minmax_lat_idx, minmax_lon_idx

raw_flip = raw_lat_sub[0] > raw_lat_sub[-1]
minmax_flip = minmax_lat_sub[0] > minmax_lat_sub[-1]
minmax_ft_flip = minmax_flip
minmax_mlp_flip = minmax_flip

if raw_flip:
    raw_lat_sub = raw_lat_sub[::-1]
if minmax_flip:
    minmax_lat_sub = minmax_lat_sub[::-1]

# =============================================================================
# 1) 전체 기간(1998-2024)을 스캔해서 "구멍" 뚫린 날짜를 자동으로 찾음
#    (raw > HOLE_RAW_THRESH 인데 mlp < HOLE_MLP_THRESH 인 픽셀이 하나라도 있는 날)
# =============================================================================
raw_time_strs = get_time_strs(raw_f)
mlp_time_strs = get_time_strs(minmax_mlp_f)
raw_idx_map = {d: i for i, d in enumerate(raw_time_strs)}

common_mask = np.array([d in raw_idx_map for d in mlp_time_strs])
mlp_common_idx = np.where(common_mask)[0]
raw_common_idx = np.array([raw_idx_map[mlp_time_strs[i]] for i in mlp_common_idx])

raw_all = raw_f.variables["precipitation"]
mlp_all = np.ma.filled(minmax_mlp_f.variables["precip"][:], np.nan)

raw_sel = np.ma.filled(
    raw_all[raw_common_idx,
            raw_lat_idx[0]:raw_lat_idx[-1] + 1,
            raw_lon_idx[0]:raw_lon_idx[-1] + 1],
    np.nan
)
if raw_flip:
    raw_sel = raw_sel[:, ::-1, :]

mlp_sel = mlp_all[mlp_common_idx][
    :,
    minmax_mlp_lat_idx[0]:minmax_mlp_lat_idx[-1] + 1,
    minmax_mlp_lon_idx[0]:minmax_mlp_lon_idx[-1] + 1
]
if minmax_mlp_flip:
    mlp_sel = mlp_sel[:, ::-1, :]

holes = (raw_sel > HOLE_RAW_THRESH) & (mlp_sel < HOLE_MLP_THRESH)
day_has_hole = holes.any(axis=(1, 2))
hole_dates = mlp_time_strs[mlp_common_idx][day_has_hole]
hole_n_pixels = holes.sum(axis=(1, 2))[day_has_hole]

print(f"구멍 뚫린 날짜 수: {len(hole_dates)}")
for d, n in zip(hole_dates, hole_n_pixels):
    print(f"  {d}  (hole pixels: {n})")

target_dates = list(hole_dates)

# =============================================================================
# 2) 찾은 날짜들만 모아서 raw / minmax / minmax_ft / minmax_mlp 4패널 시각화
#    mlp 패널에는 "구멍" 픽셀 위치를 빨간 x로 표시
# =============================================================================
elev_contour_levels = None  # 등고선은 이 스크립트에서는 생략


def make_bounds_5(vmax):
    upper = int(np.ceil(vmax / 5.0) * 5)
    return list(range(0, upper + 5, 5))


cmap = plt.get_cmap("turbo")

fig, axes = plt.subplots(
    len(target_dates), 4,
    figsize=(24, 5 * len(target_dates)),
    subplot_kw={"projection": ccrs.PlateCarree()}
)
if len(target_dates) == 1:
    axes = axes[np.newaxis, :]


def plot_panel(ax, precip, lon, lat, title, panel_cmap, panel_norm, panel_bounds,
               cbar_label):
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
    idx_minmax = int(np.where(get_time_strs(minmax_f) == date)[0][0])
    idx_minmax_ft = int(np.where(get_time_strs(minmax_ft_f) == date)[0][0])
    idx_minmax_mlp = int(np.where(mlp_time_strs == date)[0][0])

    precip_raw = load_precip_slice(
        raw_f, "precipitation", idx_raw, raw_lat_idx, raw_lon_idx, raw_flip
    )
    precip_minmax = load_precip_slice(
        minmax_f, "precip", idx_minmax, minmax_lat_idx, minmax_lon_idx, minmax_flip
    )
    precip_minmax_ft = load_precip_slice(
        minmax_ft_f, "precip", idx_minmax_ft, minmax_ft_lat_idx, minmax_ft_lon_idx, minmax_ft_flip
    )
    precip_minmax_mlp = load_precip_slice(
        minmax_mlp_f, "precip", idx_minmax_mlp, minmax_mlp_lat_idx, minmax_mlp_lon_idx, minmax_mlp_flip
    )

    vmax = np.ceil(np.nanmax(precip_raw) / 5.0) * 5.0
    bounds = make_bounds_5(vmax)
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="max")

    panels = [
        (precip_raw, f"raw (0.1°)  |  {date}"),
        (precip_minmax, f"minmax  |  {date}"),
        (precip_minmax_ft, f"minmax_ft (linear)  |  {date}"),
        (precip_minmax_mlp, f"minmax_mlp  |  {date}"),
    ]

    for col, (precip, title) in enumerate(panels):
        plot_panel(axes[date_i, col], precip, raw_lon_sub, raw_lat_sub, title,
                   cmap, norm, bounds, "Precipitation (mm/day)")

plt.tight_layout()
plt.savefig("/home/inhye_yoo/deeplearning/inr/liif/inference_tools/adapter_diagnostics/mlp_hole_dates.png",
            dpi=150, bbox_inches="tight")
plt.show()

raw_f.close()
minmax_f.close()
minmax_ft_f.close()
minmax_mlp_f.close()
