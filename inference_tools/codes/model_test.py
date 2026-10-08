import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

path_raw = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_01deg_1998_202504_noleaf.nc"
path_minmax = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax/preds_nc/preds_not_merged_2019_2024/pred_liif_minmax_ft_01deg_1998_2024.nc"
path_minmax_ft = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax_ft/preds_nc/preds_not_merged_test/pred_liif_minmax_ft_01deg_1998_2024.nc"
path_minmax_mlp = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax_ft_mlp/preds_nc/preds_not_merged_test/pred_liif_minmax_ft_01deg_1998_2024.nc"
# path_before = "/home/inhye_yoo/inference_tools/imerg_liif/results/enc_ft/preds_not_merged/preds_daily_precip_IMERG_TOPO_100X100/pred_2017_2019_merged.nc"
topo_path = "/home/inhye_yoo/data/level/NOAA_ETOPO1_topo/ETOPO1_Ice_g_gmt4_kor.nc"

raw_f = Dataset(path_raw)
minmax_f = Dataset(path_minmax)
minmax_ft_f = Dataset(path_minmax_ft)
minmax_mlp_f = Dataset(path_minmax_mlp)
# before_f = Dataset(path_before)

lon_min, lon_max = 124, 132
lat_min, lat_max = 33, 43


def get_time_index(nc_f, target_date):
    """nc 파일의 time 변수에서 target_date에 해당하는 인덱스를 찾음"""
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
    times = num2date(
        time_var[:],
        units=time_var.units,
        calendar=getattr(time_var, "calendar", "standard")
    )
    return times[idx].strftime("%Y-%m-%d")


def get_lat_lon_idx(nc_f, lat_name, lon_name, lon_range, lat_range):
    """lon/lat 범위에 해당하는 인덱스 슬라이스를 구함 (lat은 원본 순서 그대로)"""
    lat_full = np.array(nc_f.variables[lat_name][:])
    lon_full = np.array(nc_f.variables[lon_name][:])

    lat_idx = np.where((lat_full >= lat_range[0]) & (lat_full <= lat_range[1]))[0]
    lon_idx = np.where((lon_full >= lon_range[0]) & (lon_full <= lon_range[1]))[0]

    lat_sub = lat_full[lat_idx[0]:lat_idx[-1] + 1]
    lon_sub = lon_full[lon_idx[0]:lon_idx[-1] + 1]

    return lat_idx, lon_idx, lat_sub, lon_sub


def load_precip_slice(nc_f, var_name, idx, lat_idx, lon_idx, flip_lat):
    """
    지정된 시간/영역의 강수장을 잘라오고, flip_lat이 True면
    오름차순(남→북)으로 데이터를 뒤집어서 반환.
    (flip 여부는 파일의 원본 lat 순서로 미리 한 번만 판단해서 넘겨받음 —
     날짜별로 반복 호출해도 항상 같은 파일 원본 순서 기준으로 뒤집힘)
    """
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
    """
    ETOPO1(고해상도 DEM)을 target 격자(0.1도) 해상도로 block-average 다운샘플링.
    elev_model_test.py에서 쓴 것과 동일한 로직 (등고선 오버레이용).
    """
    ds = Dataset(path)
    elev_lat = np.array(ds.variables["lat"][:])
    elev_lon = np.array(ds.variables["lon"][:])
    elev = np.array(ds.variables["elev"][0])  # (lat, lon)
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


# raw 파일은 lat/lon 변수명이 다를 수 있어 확인 후 자동 대응
raw_lat_name = "lat" if "lat" in raw_f.variables else "latitude"
raw_lon_name = "lon" if "lon" in raw_f.variables else "longitude"

# ---- 모든 파일 같은 lon/lat 범위로 인덱스만 구함 (좌표값은 raw 기준으로 통일해서 씀) ----
raw_lat_idx, raw_lon_idx, raw_lat_sub, raw_lon_sub = get_lat_lon_idx(
    raw_f, raw_lat_name, raw_lon_name, [lon_min, lon_max], [lat_min, lat_max]
)
minmax_lat_idx, minmax_lon_idx, minmax_lat_sub, minmax_lon_sub = get_lat_lon_idx(
    minmax_f, "lat", "lon", [lon_min, lon_max], [lat_min, lat_max]
)
# minmax_ft / minmax_mlp는 minmax와 같은 파이프라인/같은 grid라서 좌표 인덱스 그대로 재사용
minmax_ft_lat_idx, minmax_ft_lon_idx = minmax_lat_idx, minmax_lon_idx
minmax_mlp_lat_idx, minmax_mlp_lon_idx = minmax_lat_idx, minmax_lon_idx

# ---- shape이 raw와 같은지 확인 (다르면 단순 라벨 교체로 맞출 수 없음) ----
assert len(minmax_lat_idx) == len(raw_lat_idx) and len(minmax_lon_idx) == len(raw_lon_idx), \
    f"❌ minmax shape 불일치: ({len(minmax_lat_idx)},{len(minmax_lon_idx)}) vs raw ({len(raw_lat_idx)},{len(raw_lon_idx)})"

# ---- 각 파일의 원본 lat 순서가 내림차순인지 미리 한 번만 판단 (날짜 루프 안에서 매번 새로 판단하지 않음) ----
raw_flip = raw_lat_sub[0] > raw_lat_sub[-1]
minmax_flip = minmax_lat_sub[0] > minmax_lat_sub[-1]
minmax_ft_flip = minmax_flip     # 같은 grid라서 flip 여부도 동일
minmax_mlp_flip = minmax_flip

# ---- 좌표 배열 자체도 오름차순으로 한 번만 뒤집어 둠 (플롯에는 raw_lat_sub만 쓰지만 일관성을 위해) ----
if raw_flip:
    raw_lat_sub = raw_lat_sub[::-1]
if minmax_flip:
    minmax_lat_sub = minmax_lat_sub[::-1]

# ---- 등고선 오버레이용 고도(DEM), raw 격자에 맞춰 미리 한 번만 regrid ----
elev_grid = load_and_regrid_elev(topo_path, raw_lat_sub, raw_lon_sub)

target_dates = ["2019-07-20", "2018-08-26"]
vmax_by_date = {
    "2019-07-20": 100,
    "2018-08-26": 200,
}


# ---- 5 간격 bounds 자동 생성 함수 ----
def make_bounds_5(vmax):
    """0부터 vmax까지 5 간격 구간 리스트 생성 (vmax가 5의 배수가 아니면 올림 처리)"""
    upper = int(np.ceil(vmax / 5.0) * 5)
    return list(range(0, upper + 5, 5))


cmap = plt.get_cmap("turbo")

# ---- 등고선 레벨 (300m 간격) ----
elev_contour_levels = np.arange(300, np.nanmax(elev_grid) + 300, 300)

fig, axes = plt.subplots(
    len(target_dates), 4,
    figsize=(24, 5 * len(target_dates)),
    subplot_kw={"projection": ccrs.PlateCarree()}
)
if len(target_dates) == 1:
    axes = axes[np.newaxis, :] if axes.ndim == 1 else axes


def plot_panel(ax, precip, lon, lat, title, panel_cmap, panel_norm, panel_bounds, cbar_label, show_contour):
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

    # ---- 차이 패널에는 고도 등고선을 겹쳐서, 지형과 차이 패턴이 맞는지 눈으로 확인 ----
    if show_contour:
        ax.contour(
            raw_lon_sub, raw_lat_sub, elev_grid,
            levels=elev_contour_levels,
            colors="black", linewidths=0.4, alpha=0.6,
            transform=ccrs.PlateCarree()
        )

    ax.set_title(title, fontsize=11)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    if panel_bounds is not None:
        cbar = plt.colorbar(mesh, ax=ax, orientation="vertical",
                             shrink=0.8, pad=0.05,
                             ticks=panel_bounds, extend="max")
    else:
        cbar = plt.colorbar(mesh, ax=ax, orientation="vertical",
                             shrink=0.8, pad=0.05, extend="both")
    cbar.set_label(cbar_label)


for date_i, date in enumerate(target_dates):
    idx_raw = get_time_index(raw_f, date)
    idx_minmax = get_time_index(minmax_f, date)
    idx_minmax_ft = get_time_index(minmax_ft_f, date)
    idx_minmax_mlp = get_time_index(minmax_mlp_f, date)

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

    print(f"\n===== {date} =====")
    print("idx raw       :", idx_raw)
    print("idx minmax    :", idx_minmax)
    print("idx minmax_ft :", idx_minmax_ft)
    print("idx minmax_mlp:", idx_minmax_mlp)

    print("raw date       :", get_date_at_index(raw_f, idx_raw))
    print("minmax date    :", get_date_at_index(minmax_f, idx_minmax))
    print("minmax_ft date :", get_date_at_index(minmax_ft_f, idx_minmax_ft))
    print("minmax_mlp date:", get_date_at_index(minmax_mlp_f, idx_minmax_mlp))

    vmax = vmax_by_date[date]

    bounds = make_bounds_5(vmax)
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="max")

    abs_panels = [
        (precip_raw, raw_lon_sub, raw_lat_sub, f"raw (0.1°)  |  {date}"),
        (precip_minmax, raw_lon_sub, raw_lat_sub, f"minmax  |  {date}"),
        (precip_minmax_ft, raw_lon_sub, raw_lat_sub, f"minmax_ft (linear)  |  {date}"),
        (precip_minmax_mlp, raw_lon_sub, raw_lat_sub, f"minmax_mlp  |  {date}"),
    ]

    for col, (precip, lon, lat, title) in enumerate(abs_panels):
        plot_panel(axes[date_i, col], precip, lon, lat, title,
                   cmap, norm, bounds, "Precipitation (mm/day)", show_contour=False)

print("\n===== COORD CHECK (raw 기준으로 통일된 좌표) =====")

print(
    "RAW lat:",
    raw_lat_sub[0], raw_lat_sub[-1],
    "lon:", raw_lon_sub[0], raw_lon_sub[-1],
    "shape:", len(raw_lat_sub), len(raw_lon_sub)
)

plt.tight_layout()
plt.savefig("/home/inhye_yoo/deeplearning/inr/liif/inference_tools/precip_comparison/precip_raw_vs_minmax_ft_mlp.png",
            dpi=150, bbox_inches="tight")
plt.show()

raw_f.close()
minmax_f.close()
minmax_ft_f.close()
minmax_mlp_f.close()
