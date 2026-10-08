"""
_climate_edsr_topo_80x100_topo_minmax_lte 예측 결과를 raw IMERG(0.1도) 와 비교해서
두 사례일(2019-07-20, 2018-08-21) 강수장을 그림으로 저장.

패널: raw (0.1°) | LTE pred | (pred - raw) 차이(+등고선 오버레이)
"""
import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

path_raw = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_01deg_1998_202504_noleaf.nc"
path_lte = (
    "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_minmax_lte/"
    "preds_nc/preds_check_epoch-best-psnr/pred_liif_lte_epoch-best-psnr_01deg_1998_2024.nc"
)
topo_path = "/home/inhye_yoo/data/level/NOAA_ETOPO1_topo/ETOPO1_Ice_g_gmt4_kor.nc"

out_png = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/precip_comparison/precip_raw_vs_lte.png"

lon_min, lon_max = 124, 132
lat_min, lat_max = 33, 43

target_dates = ["2019-07-20", "2018-08-26"]
vmax_by_date = {
    "2019-07-20": 100,
    "2018-08-26": 200,
}

raw_f = Dataset(path_raw)
lte_f = Dataset(path_lte)


# ============================================================
# 유틸
# ============================================================
def get_time_index(nc_f, target_date):
    time_var = nc_f.variables["time"]
    times = num2date(time_var[:], units=time_var.units,
                     calendar=getattr(time_var, "calendar", "standard"))
    time_strs = [t.strftime("%Y-%m-%d") for t in times]
    if target_date not in time_strs:
        raise ValueError(f"{target_date}가 시간축에 없습니다. "
                         f"사용 가능 범위: {time_strs[0]} ~ {time_strs[-1]}")
    return time_strs.index(target_date)


def get_date_at_index(nc_f, idx):
    time_var = nc_f.variables["time"]
    times = num2date(time_var[:], units=time_var.units,
                     calendar=getattr(time_var, "calendar", "standard"))
    return times[idx].strftime("%Y-%m-%d")


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


def load_and_regrid_elev(path, target_lat, target_lon):
    """ETOPO1 을 target 격자(0.1도)로 block-average 다운샘플링 (등고선 오버레이용)."""
    ds = Dataset(path)
    elev_lat = np.array(ds.variables["lat"][:])
    elev_lon = np.array(ds.variables["lon"][:])
    elev = np.array(ds.variables["elev"][0])
    ds.close()

    if elev_lat[0] > elev_lat[-1]:
        elev_lat = elev_lat[::-1]
        elev = elev[::-1, :]

    lat_step = target_lat[1] - target_lat[0]
    lon_step = target_lon[1] - target_lon[0]
    lat_edges = np.concatenate([target_lat - lat_step / 2, [target_lat[-1] + lat_step / 2]])
    lon_edges = np.concatenate([target_lon - lon_step / 2, [target_lon[-1] + lon_step / 2]])

    lat_bin = np.clip(np.digitize(elev_lat, lat_edges) - 1, 0, len(target_lat) - 1)
    lon_bin = np.clip(np.digitize(elev_lon, lon_edges) - 1, 0, len(target_lon) - 1)
    lat_bin_2d, lon_bin_2d = np.meshgrid(lat_bin, lon_bin, indexing="ij")

    sum_grid = np.zeros((len(target_lat), len(target_lon)), dtype=np.float64)
    cnt_grid = np.zeros((len(target_lat), len(target_lon)), dtype=np.float64)
    np.add.at(sum_grid, (lat_bin_2d, lon_bin_2d), elev)
    np.add.at(cnt_grid, (lat_bin_2d, lon_bin_2d), 1)
    with np.errstate(invalid="ignore"):
        elev_regrid = sum_grid / cnt_grid
    return elev_regrid.astype(np.float32)


def make_bounds_5(vmax):
    upper = int(np.ceil(vmax / 5.0) * 5)
    return list(range(0, upper + 5, 5))


# ============================================================
# 좌표 / flip 판단 (한 번만)
# ============================================================
raw_lat_name = "lat" if "lat" in raw_f.variables else "latitude"
raw_lon_name = "lon" if "lon" in raw_f.variables else "longitude"

raw_lat_idx, raw_lon_idx, raw_lat_sub, raw_lon_sub = get_lat_lon_idx(
    raw_f, raw_lat_name, raw_lon_name, [lon_min, lon_max], [lat_min, lat_max]
)
lte_lat_idx, lte_lon_idx, lte_lat_sub, lte_lon_sub = get_lat_lon_idx(
    lte_f, "lat", "lon", [lon_min, lon_max], [lat_min, lat_max]
)

assert len(lte_lat_idx) == len(raw_lat_idx) and len(lte_lon_idx) == len(raw_lon_idx), \
    (f"shape 불일치: LTE ({len(lte_lat_idx)},{len(lte_lon_idx)}) vs "
     f"raw ({len(raw_lat_idx)},{len(raw_lon_idx)})")

raw_flip = raw_lat_sub[0] > raw_lat_sub[-1]
lte_flip = lte_lat_sub[0] > lte_lat_sub[-1]
if raw_flip:
    raw_lat_sub = raw_lat_sub[::-1]
if lte_flip:
    lte_lat_sub = lte_lat_sub[::-1]

elev_grid = load_and_regrid_elev(topo_path, raw_lat_sub, raw_lon_sub)
elev_contour_levels = np.arange(300, np.nanmax(elev_grid) + 300, 300)

cmap = plt.get_cmap("turbo")
diff_cmap = plt.get_cmap("RdBu_r")


# ============================================================
# 플롯
# ============================================================
def plot_precip(ax, precip, lon, lat, title, panel_norm, bounds):
    ax.set_extent([raw_lon_sub.min(), raw_lon_sub.max(),
                   raw_lat_sub.min(), raw_lat_sub.max()], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    mesh = ax.pcolormesh(lon, lat, precip, transform=ccrs.PlateCarree(),
                         cmap=cmap, norm=panel_norm, shading="auto")
    ax.set_title(title, fontsize=11)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.8,
                        pad=0.05, ticks=bounds, extend="max")
    cbar.set_label("Precipitation (mm/day)")


def plot_diff(ax, diff, lon, lat, title, dmax):
    ax.set_extent([raw_lon_sub.min(), raw_lon_sub.max(),
                   raw_lat_sub.min(), raw_lat_sub.max()], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    mesh = ax.pcolormesh(lon, lat, diff, transform=ccrs.PlateCarree(),
                         cmap=diff_cmap, vmin=-dmax, vmax=dmax, shading="auto")
    ax.contour(raw_lon_sub, raw_lat_sub, elev_grid, levels=elev_contour_levels,
               colors="black", linewidths=0.4, alpha=0.6, transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=11)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.8,
                        pad=0.05, extend="both")
    cbar.set_label("LTE - raw (mm/day)")


fig, axes = plt.subplots(
    len(target_dates), 3,
    figsize=(18, 5 * len(target_dates)),
    subplot_kw={"projection": ccrs.PlateCarree()},
)
if axes.ndim == 1:
    axes = axes[np.newaxis, :]

for date_i, date in enumerate(target_dates):
    idx_raw = get_time_index(raw_f, date)
    idx_lte = get_time_index(lte_f, date)

    precip_raw = load_precip_slice(raw_f, "precipitation", idx_raw,
                                   raw_lat_idx, raw_lon_idx, raw_flip)
    precip_lte = load_precip_slice(lte_f, "precip", idx_lte,
                                   lte_lat_idx, lte_lon_idx, lte_flip)
    diff = precip_lte - precip_raw

    print(f"\n===== {date} =====")
    print("raw date :", get_date_at_index(raw_f, idx_raw),
          " lte date :", get_date_at_index(lte_f, idx_lte))
    print(f"raw  max {np.nanmax(precip_raw):.1f}  mean {np.nanmean(precip_raw):.2f}")
    print(f"lte  max {np.nanmax(precip_lte):.1f}  mean {np.nanmean(precip_lte):.2f}")
    print(f"MAE(lte, raw) = {np.nanmean(np.abs(diff)):.3f} mm/day")

    vmax = vmax_by_date[date]
    bounds = make_bounds_5(vmax)
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="max")
    dmax = float(np.nanpercentile(np.abs(diff), 99)) or 1.0

    plot_precip(axes[date_i, 0], precip_raw, raw_lon_sub, raw_lat_sub,
                f"raw IMERG (0.1°)  |  {date}", norm, bounds)
    plot_precip(axes[date_i, 1], precip_lte, raw_lon_sub, raw_lat_sub,
                f"LTE pred (best-psnr)  |  {date}", norm, bounds)
    plot_diff(axes[date_i, 2], diff, raw_lon_sub, raw_lat_sub,
              f"LTE - raw  |  {date}", dmax)

print("\n===== COORD CHECK =====")
print("RAW lat:", raw_lat_sub[0], raw_lat_sub[-1],
      "lon:", raw_lon_sub[0], raw_lon_sub[-1],
      "shape:", len(raw_lat_sub), len(raw_lon_sub))

plt.tight_layout()
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print("\nsaved:", out_png)

raw_f.close()
lte_f.close()
