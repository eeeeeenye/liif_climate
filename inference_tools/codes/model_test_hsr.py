# =============================================================================
# HSR 레이더 SR 케이스 비교 (model_test.py 구조를 그대로 따르되, 서로 다른 두 모델을
# 각자의 케이스 시각에서 raw HSR 레이더와 나란히 그린다)
#
# model A: _climate_edsr_topo_80x100_topo_precip_minmax
#          -> hsr_radar_sr_2021jja/ : 2021 JJA, 하루 1장(일 누적, KST)
# model B: _climate_edsr_radar_hsr_2x_minmax_liif
#          -> preds_nc/case_sr/     : 2019-06-01~11, 1시간 1장(시간 누적, KST)
#
# 두 모델은 학습 기간/시간 해상도가 달라서 겹치는 날짜가 없다.
# -> 모델별로 "자기 데이터 안에서 가장 비가 많이 온 case"를 자동으로 골라
#    (raw HSR vs 모델 SR) 2단 그림을 각각 따로 그려서 따로 저장한다.
#
# raw HSR 쪽 정답은 다음처럼 만든다 (모델의 학습 시간 해상도에 맞춤):
#   - model A(일 누적) : 해당 KST 날짜의 24개 시간 슬라이스를 합산
#   - model B(시간 누적): 해당 KST 시각의 시간 슬라이스 그대로 사용
# =============================================================================

import glob
import os
import re

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date, date2index
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

RAW_PATH = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/HSR_2019_2021_JJA_merged_kst_hourly_2881x2305.nc"

MODEL_A_DIR = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax/hsr_radar_sr_2021jja"
MODEL_B_DIR = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif/preds_nc/case_sr"

SCAN_RES = "300m"   # 최대값(케이스) 스캔은 가벼운 300m로
PLOT_RES = "300m"   # 실제 그림에 쓸 해상도 (100m로 바꾸면 더 느려짐/메모리 커짐)

SAVE_DIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"

lon_min, lon_max = 124.0, 132.0
lat_min, lat_max = 33.0, 43.0


# =============================================================================
# 공통 유틸
# =============================================================================
def get_lat_lon_idx(nc_f, lat_name, lon_name, lon_range, lat_range):
    lat_full = np.array(nc_f.variables[lat_name][:])
    lon_full = np.array(nc_f.variables[lon_name][:])

    lat_idx = np.where((lat_full >= lat_range[0]) & (lat_full <= lat_range[1]))[0]
    lon_idx = np.where((lon_full >= lon_range[0]) & (lon_full <= lon_range[1]))[0]

    lat_sub = lat_full[lat_idx[0]:lat_idx[-1] + 1]
    lon_sub = lon_full[lon_idx[0]:lon_idx[-1] + 1]
    return lat_idx, lon_idx, lat_sub, lon_sub


def make_bounds_5(vmax):
    upper = max(5, int(np.ceil(vmax / 5.0) * 5))
    return list(range(0, upper + 5, 5))


def make_ticks(bounds, max_ticks=12):
    """bounds가 너무 촘촘하면(값 범위가 클 때) colorbar 눈금 개수를 줄여서 겹침 방지."""
    n = len(bounds)
    if n <= max_ticks:
        return bounds
    stride = int(np.ceil(n / max_ticks))
    return bounds[::stride]


def plot_panel(ax, precip, lon, lat, title, cmap, norm, bounds, cbar_label):
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")

    mesh = ax.pcolormesh(
        lon, lat, precip,
        transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, shading="auto"
    )
    ax.set_title(title, fontsize=11)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.8, pad=0.05,
                         ticks=make_ticks(bounds), extend="max")
    cbar.set_label(cbar_label)


# =============================================================================
# model A / model B 파일 목록 파싱
# =============================================================================
def list_model_a_files(model_dir, res):
    pattern = os.path.join(model_dir, f"hsr_liif_sr_{res}_*.nc")
    files = sorted(glob.glob(pattern))
    out = []
    for f in files:
        m = re.search(r"hsr_liif_sr_\d+m_(\d{4}-\d{2}-\d{2})\.nc$", f)
        if m:
            out.append((pd.Timestamp(m.group(1)), None, f))
    return out


def list_model_b_files(model_dir, res):
    pattern = os.path.join(model_dir, f"hsr2x_liif_sr_{res}_*.nc")
    files = sorted(glob.glob(pattern))
    out = []
    for f in files:
        m = re.search(r"hsr2x_liif_sr_\d+m_(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})kst\.nc$", f)
        if m:
            date = pd.Timestamp(m.group(1))
            hour = int(m.group(2))
            out.append((date, hour, f))
    return out


def pick_heaviest_case(file_list):
    """(date, hour, path) 목록에서 precip 최댓값이 가장 큰 파일을 고른다."""
    best = None
    for date, hour, path in file_list:
        nc = Dataset(path)
        vmax = float(np.nanmax(nc.variables["precip"][:]))
        nc.close()
        if best is None or vmax > best[0]:
            best = (vmax, date, hour, path)
    return best  # (vmax, date, hour, path)


def pick_heaviest_hour_on_date(file_list, date):
    """(date, hour, path) 목록에서 지정 날짜에 한정해 precip 최댓값이 가장 큰 시각을 고른다."""
    subset = [f for f in file_list if f[0] == date]
    if not subset:
        return None
    return pick_heaviest_case(subset)


# =============================================================================
# raw HSR에서 정답(daily sum 또는 hourly) 가져오기
# =============================================================================
def load_raw_daily_sum(raw_f, date, lat_idx, lon_idx):
    """date(KST) 하루치(00~23시) HSR을 합산."""
    time_var = raw_f.variables["time"]
    times = num2date(time_var[:], units=time_var.units,
                      calendar=getattr(time_var, "calendar", "standard"))
    times = pd.to_datetime([str(t) for t in times])

    day_mask = (times.date == date.date())
    idxs = np.where(day_mask)[0]
    if len(idxs) == 0:
        raise ValueError(f"raw HSR에 {date.date()} 시간 슬라이스가 없습니다.")

    acc = None
    for i in idxs:
        sl = raw_f.variables["precip"][i, lat_idx[0]:lat_idx[-1] + 1, lon_idx[0]:lon_idx[-1] + 1]
        sl = np.ma.filled(sl, 0.0)
        acc = sl.astype(np.float64) if acc is None else acc + sl
    print(f"  raw daily sum: {date.date()} 사용된 시간 슬라이스 {len(idxs)}개")
    return acc


def load_raw_hourly(raw_f, date, hour, lat_idx, lon_idx):
    target = date + pd.Timedelta(hours=hour)
    time_var = raw_f.variables["time"]
    idx = date2index(target.to_pydatetime(), time_var,
                      calendar=getattr(time_var, "calendar", "standard"),
                      select="nearest")
    sl = raw_f.variables["precip"][idx, lat_idx[0]:lat_idx[-1] + 1, lon_idx[0]:lon_idx[-1] + 1]
    sl = np.ma.filled(sl, np.nan)
    got = num2date(time_var[idx], units=time_var.units,
                    calendar=getattr(time_var, "calendar", "standard"))
    print(f"  raw hourly: 요청 {target} -> 실제 {got}")
    return sl


# =============================================================================
# 모델 하나(raw + SR 2단)를 그려서 저장
# =============================================================================
def run_one_model(model_name, model_dir, list_fn, is_daily, save_name,
                   forced_date=None, forced_hour=None, forced_vmax=None):
    print(f"\n===== {model_name} =====")
    files = list_fn(model_dir, SCAN_RES)
    if not files:
        print(f"  파일을 찾지 못했습니다: {model_dir}")
        return

    if forced_date is not None:
        target_date = pd.Timestamp(forced_date)
        if is_daily or forced_hour is not None:
            date, hour, scan_path = target_date, forced_hour, None
            print(f"  case 고정: date={date.date()}"
                  + (f" hour={hour:02d}KST" if hour is not None else " (일 누적)"))
        else:
            # 시간별 모델인데 시각을 안 정해줬으면, 그 날짜 안에서 가장 비가 많이 온 시각을 고름
            picked = pick_heaviest_hour_on_date(files, target_date)
            if picked is None:
                print(f"  {target_date.date()}에 해당하는 파일이 없습니다: {model_dir}")
                return
            vmax, date, hour, scan_path = picked
            print(f"  case 고정: date={date.date()}, 그 안에서 최댓값 시각 선택 "
                  f"hour={hour:02d}KST  max(precip)={vmax:.2f}")
    else:
        vmax, date, hour, scan_path = pick_heaviest_case(files)
        print(f"  가장 비가 많이 온 case 선택: date={date.date()}"
              + (f" hour={hour:02d}KST" if hour is not None else " (일 누적)")
              + f"  max(precip)={vmax:.2f}")

    # PLOT_RES로 실제 SR 파일 다시 찾기
    plot_files = list_fn(model_dir, PLOT_RES)
    sr_path = None
    for d, h, p in plot_files:
        if d == date and h == hour:
            sr_path = p
            break
    if sr_path is None:
        print(f"  PLOT_RES={PLOT_RES} 파일을 못 찾아 SCAN_RES 파일을 그대로 사용합니다: {scan_path}")
        sr_path = scan_path

    sr_f = Dataset(sr_path)
    sr_lat = np.array(sr_f.variables["lat"][:])
    sr_lon = np.array(sr_f.variables["lon"][:])
    sr_precip = np.array(sr_f.variables["precip"][:])
    sr_f.close()
    sr_flip = sr_lat[0] > sr_lat[-1]
    if sr_flip:
        sr_lat = sr_lat[::-1]
        sr_precip = sr_precip[::-1, :]

    raw_f = Dataset(RAW_PATH)
    raw_lat_idx, raw_lon_idx, raw_lat_sub, raw_lon_sub = get_lat_lon_idx(
        raw_f, "lat", "lon", [lon_min, lon_max], [lat_min, lat_max]
    )
    if is_daily:
        raw_precip = load_raw_daily_sum(raw_f, date, raw_lat_idx, raw_lon_idx)
        cbar_label = "Precipitation (mm/day)"
        date_label = f"{date.date()} (daily sum)"
    else:
        raw_precip = load_raw_hourly(raw_f, date, hour, raw_lat_idx, raw_lon_idx)
        cbar_label = "Precipitation (mm/hr)"
        date_label = f"{date.date()} {hour:02d}:00 KST"
    raw_f.close()

    vmax_plot = forced_vmax if forced_vmax is not None else max(np.nanmax(raw_precip), np.nanmax(sr_precip))
    bounds = make_bounds_5(vmax_plot)
    cmap = plt.get_cmap("turbo")
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="max")

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    plot_panel(axes[0], raw_precip, raw_lon_sub, raw_lat_sub,
               f"raw HSR (native)  |  {date_label}", cmap, norm, bounds, cbar_label)
    plot_panel(axes[1], sr_precip, sr_lon, sr_lat,
               f"{model_name} SR ({PLOT_RES})  |  {date_label}", cmap, norm, bounds, cbar_label)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장 완료: {save_path}")


if __name__ == "__main__":
    run_one_model(
        "model A (_climate_edsr_topo_80x100_topo_precip_minmax)",
        MODEL_A_DIR, list_model_a_files, is_daily=True,
        save_name="case_hsr_modelA_topo80x100_vs_raw.png",
        forced_date="2021-08-23", forced_vmax=300,
    )
    run_one_model(
        "model B (_climate_edsr_radar_hsr_2x_minmax_liif)",
        MODEL_B_DIR, list_model_b_files, is_daily=False,
        save_name="case_hsr_modelB_radarhsr2x_vs_raw.png",
        forced_date="2021-08-23", forced_vmax=300,
    )
