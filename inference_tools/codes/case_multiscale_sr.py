"""
raw HSR(native ~500m, 2021-08-23)을 인위적으로 10km / 1km로 다운샘플링한 뒤
model A(_climate_edsr_topo_80x100_topo_precip_minmax)로 다시 500m로 SR해서
"입력 해상도가 다르면 복원이 얼마나 달라지는가"를 보는 실험.
추가로 model B(_climate_edsr_radar_hsr_2x_minmax_liif)로 raw HSR(500m, 같은 날 최댓값
시각)을 100m로 SR하는 것도 같이 그린다 (500m -> 100m).

3개 케이스, 3개 그림으로 각각 저장:
  1) 10km -> 500m  (model A, out-of-distribution 입력)
  2) 1km  -> 500m  (model A, out-of-distribution 입력)
  3) 500m -> 100m  (model B, 원래 학습 스케일)

env: /home/inhye_yoo/anaconda3/envs/liif/bin/python
"""
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset, num2date, date2index
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
from models.models import make
from datasets.image_folder import TopoInMemory
from utils import make_coord

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HSR_PATH = (
    "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/"
    "HSR_2019_2021_JJA_merged_kst_hourly_2881x2305.nc"
)
TOPO_PATH = "/home/inhye_yoo/data/level/NOAA_ETOPO1_topo/ETOPO1_Ice_g_gmt4_kor.nc"

MODEL_A_PATH = ("/home/inhye_yoo/deeplearning/inr/liif/save/"
                 "_climate_edsr_topo_80x100_topo_precip_minmax/epoch-best-psnr.pth")
MODEL_B_PATH = ("/home/inhye_yoo/deeplearning/inr/liif/save/"
                 "_climate_edsr_radar_hsr_2x_minmax_liif/epoch-best-psnr.pth")

LAT_MIN, LAT_MAX = 33.0, 43.0
LON_MIN, LON_MAX = 124.0, 132.0

# model B(500m->100m)는 feat_unfold가 전체 입력 도메인 크기에 비례해서 GPU 메모리를
# 아주 많이 먹기 때문에(전체 도메인 기준 ~25GB), 스톰 core 주변으로 좁혀서 돌린다.
MODEL_B_LAT_MIN, MODEL_B_LAT_MAX = 33.0, 37.5
MODEL_B_LON_MIN, MODEL_B_LON_MAX = 129.0, 132.0

TARGET_DATE = "2021-08-08"
TARGET_HOUR = 11  # KST, model B용 (해당일 raw 최댓값 시각, 사전에 스캔해서 확인)
CASE_TAG = "2021-08-08"

SAVE_DIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/case_studies/multiscale"

MODEL_A_MAXV = 506.01995849609375  # mm/day 기준 (infer_hsr_radar_sr.py와 동일)
MODEL_B_PMIN, MODEL_B_PMAX = 0.0, 74.8784  # mm/hr 기준 (infer_hsr_radar_2x_sr.py와 동일)

BSIZE = 500_000


# =============================================================================
# 공통 유틸
# =============================================================================
def crop_idx(coord_full, vmin, vmax):
    idx = np.where((coord_full >= vmin) & (coord_full <= vmax))[0]
    if len(idx) == 0:
        raise ValueError(f"[{vmin}, {vmax}] 범위에 해당하는 격자가 없습니다.")
    return idx


def load_hsr_daily(hsr_path, date, lat_min, lat_max, lon_min, lon_max):
    ds = Dataset(hsr_path)
    time_var = ds.variables["time"]
    time_arr = pd.to_datetime(
        num2date(time_var[:], time_var.units,
                 calendar=getattr(time_var, "calendar", "standard"),
                 only_use_cftime_datetimes=False)
    )
    lat_full = np.asarray(ds.variables["lat"][:], float)
    lon_full = np.asarray(ds.variables["lon"][:], float)
    lat_idx = crop_idx(lat_full, lat_min, lat_max)
    lon_idx = crop_idx(lon_full, lon_min, lon_max)
    lat_sub = lat_full[lat_idx[0]:lat_idx[-1] + 1]
    lon_sub = lon_full[lon_idx[0]:lon_idx[-1] + 1]

    d = pd.Timestamp(date).normalize()
    t_idx = np.where(time_arr.normalize() == d)[0]
    if len(t_idx) == 0:
        ds.close()
        raise ValueError(f"{date} (KST) 시간 데이터를 HSR 파일에서 찾지 못했습니다.")

    acc = np.zeros((len(lat_sub), len(lon_sub)), dtype=np.float64)
    cnt = np.zeros_like(acc)
    for t in t_idx:
        s = np.asarray(
            ds.variables["precip"][t, lat_idx[0]:lat_idx[-1] + 1, lon_idx[0]:lon_idx[-1] + 1],
            float,
        )
        s[s < -1e8] = np.nan
        m = np.isfinite(s) & (s >= 0)
        acc[m] += s[m]
        cnt[m] += 1
    ds.close()
    daily = np.where(cnt > 0, acc, np.nan).astype(np.float32)
    daily = np.nan_to_num(daily, nan=0.0)
    return daily, lat_sub, lon_sub


def load_hsr_hourly(hsr_path, date, hour, lat_min, lat_max, lon_min, lon_max):
    ds = Dataset(hsr_path)
    lat_full = np.asarray(ds.variables["lat"][:], float)
    lon_full = np.asarray(ds.variables["lon"][:], float)
    lat_idx = crop_idx(lat_full, lat_min, lat_max)
    lon_idx = crop_idx(lon_full, lon_min, lon_max)
    lat_sub = lat_full[lat_idx[0]:lat_idx[-1] + 1]
    lon_sub = lon_full[lon_idx[0]:lon_idx[-1] + 1]

    target = pd.Timestamp(date) + pd.Timedelta(hours=hour)
    time_var = ds.variables["time"]
    t_idx = date2index(target.to_pydatetime(), time_var,
                        calendar=getattr(time_var, "calendar", "standard"),
                        select="nearest")
    s = np.asarray(
        ds.variables["precip"][t_idx, lat_idx[0]:lat_idx[-1] + 1, lon_idx[0]:lon_idx[-1] + 1],
        float,
    )
    s[s < -1e8] = np.nan
    s = np.nan_to_num(s, nan=0.0).astype(np.float32)
    ds.close()
    return s, lat_sub, lon_sub


def out_grid_for_resolution(res_m, lat_min, lat_max, lon_min, lon_max):
    mean_lat_rad = np.deg2rad((lat_min + lat_max) / 2.0)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(mean_lat_rad)
    step_lat = res_m / m_per_deg_lat
    step_lon = res_m / m_per_deg_lon
    H = max(1, round((lat_max - lat_min) / step_lat))
    W = max(1, round((lon_max - lon_min) / step_lon))
    step_lat = (lat_max - lat_min) / H
    step_lon = (lon_max - lon_min) / W
    lat_out = lat_min + step_lat * (np.arange(H) + 0.5)
    lon_out = lon_min + step_lon * (np.arange(W) + 0.5)
    return H, W, lat_out, lon_out


def block_average_downsample(data, src_lat, src_lon, tgt_lat, tgt_lon):
    """src(촘촘)를 tgt(성긴) 격자로 block-average. src/tgt 모두 오름차순 가정."""
    lat_step = tgt_lat[1] - tgt_lat[0]
    lon_step = tgt_lon[1] - tgt_lon[0]
    lat_edges = np.concatenate([tgt_lat - lat_step / 2, [tgt_lat[-1] + lat_step / 2]])
    lon_edges = np.concatenate([tgt_lon - lon_step / 2, [tgt_lon[-1] + lon_step / 2]])

    lat_bin = np.clip(np.digitize(src_lat, lat_edges) - 1, 0, len(tgt_lat) - 1)
    lon_bin = np.clip(np.digitize(src_lon, lon_edges) - 1, 0, len(tgt_lon) - 1)
    lat_bin_2d, lon_bin_2d = np.meshgrid(lat_bin, lon_bin, indexing="ij")

    sum_grid = np.zeros((len(tgt_lat), len(tgt_lon)), dtype=np.float64)
    cnt_grid = np.zeros((len(tgt_lat), len(tgt_lon)), dtype=np.float64)
    np.add.at(sum_grid, (lat_bin_2d, lon_bin_2d), data)
    np.add.at(cnt_grid, (lat_bin_2d, lon_bin_2d), 1)
    with np.errstate(invalid="ignore"):
        out = sum_grid / cnt_grid
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def build_and_load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_spec = copy.deepcopy(ckpt["model"])
    print("checkpoint model name:", model_spec["name"])
    model = make({"name": model_spec["name"], "args": model_spec["args"]}).to(device)
    model.load_state_dict(model_spec["sd"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def chunked_query(model, coord, cell, bsize):
    n = coord.shape[1]
    preds = []
    for ql in range(0, n, bsize):
        qr = min(ql + bsize, n)
        pred = model.query_rgb(coord[:, ql:qr, :], cell[:, ql:qr, :])
        preds.append(pred)
    return torch.cat(preds, dim=1)


@torch.no_grad()
def run_sr(model, inp_2d, maxv, pmin, out_res_m, device,
           lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX):
    """inp_2d: (H,W) 물리값(mm). maxv/pmin으로 정규화 후 SR, 다시 mm로 복원."""
    x_norm = (inp_2d.astype(np.float32) - pmin) / (maxv - pmin)
    inp_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    model.gen_feat(inp_tensor)

    H, W, lat_out, lon_out = out_grid_for_resolution(out_res_m, lat_min, lat_max, lon_min, lon_max)
    coord = make_coord(shape=(H, W)).unsqueeze(0).to(device)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2.0 / H
    cell[:, :, 1] *= 2.0 / W

    pred = chunked_query(model, coord, cell, BSIZE)
    pred = pred.view(H, W).clamp(0.0, 1.0)
    pred_mm = (pred * (maxv - pmin) + pmin).clamp(min=0.0).detach().cpu().numpy()

    del inp_tensor, coord, cell, pred
    torch.cuda.empty_cache()
    return pred_mm, lat_out, lon_out


# =============================================================================
# 플롯 유틸
# =============================================================================
def make_bounds_5(vmax):
    upper = max(5, int(np.ceil(vmax / 5.0) * 5))
    return list(range(0, upper + 5, 5))


def make_ticks(bounds, max_ticks=12):
    n = len(bounds)
    if n <= max_ticks:
        return bounds
    stride = int(np.ceil(n / max_ticks))
    return bounds[::stride]


def plot_panel(ax, precip, lon, lat, title, cmap, norm, bounds, cbar_label):
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    mesh = ax.pcolormesh(lon, lat, precip, transform=ccrs.PlateCarree(),
                          cmap=cmap, norm=norm, shading="auto")
    ax.set_title(title, fontsize=11)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.8, pad=0.05,
                         ticks=make_ticks(bounds), extend="max")
    cbar.set_label(cbar_label)


def save_3panel(panels, save_path, vmax):
    """panels: [(data, lon, lat, title), ...] 길이 2 또는 3."""
    bounds = make_bounds_5(vmax)
    cmap = plt.get_cmap("turbo")
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="max")

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 6),
                              subplot_kw={"projection": ccrs.PlateCarree()})
    if len(panels) == 1:
        axes = [axes]
    for ax, (data, lon, lat, title) in zip(axes, panels):
        plot_panel(ax, data, lon, lat, title, cmap, norm, bounds, "Precipitation (mm)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료: {save_path}")


NPZ_PATH = os.path.join(SAVE_DIR, f"_case_multiscale_intermediate_{CASE_TAG}.npz")
STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"  # "a" | "b" | "plot" | "all"


def load_npz_dict():
    if os.path.exists(NPZ_PATH):
        d = dict(np.load(NPZ_PATH))
    else:
        d = {}
    return d


def save_npz_dict(d):
    np.savez(NPZ_PATH, **d)
    print(f"중간 결과 저장: {NPZ_PATH}")


# =============================================================================
# 1) raw 로드 + 인위적 다운샘플 (GPU 불필요, 항상 실행)
# =============================================================================
print("raw HSR daily sum 로드 중...")
raw_daily, raw_lat, raw_lon = load_hsr_daily(HSR_PATH, TARGET_DATE, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
print(f"  raw daily shape={raw_daily.shape}, max={raw_daily.max():.1f} mm/day")

_, _, lat_10km, lon_10km = out_grid_for_resolution(10_000, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
_, _, lat_1km, lon_1km = out_grid_for_resolution(1_000, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

input_10km = block_average_downsample(raw_daily, raw_lat, raw_lon, lat_10km, lon_10km)
input_1km = block_average_downsample(raw_daily, raw_lat, raw_lon, lat_1km, lon_1km)
print(f"  10km 입력 shape={input_10km.shape}, max={input_10km.max():.1f}")
print(f"  1km 입력 shape={input_1km.shape}, max={input_1km.max():.1f}")

print("raw HSR hourly(model B용, 좁은 도메인) 로드 중...")
raw_hourly, raw_hourly_lat, raw_hourly_lon = load_hsr_hourly(
    HSR_PATH, TARGET_DATE, TARGET_HOUR,
    MODEL_B_LAT_MIN, MODEL_B_LAT_MAX, MODEL_B_LON_MIN, MODEL_B_LON_MAX
)
print(f"  raw hourly shape={raw_hourly.shape}, max={raw_hourly.max():.1f} mm/hr")

# =============================================================================
# 2) model A: 10km -> 500m, 1km -> 500m
# =============================================================================
if STAGE in ("a", "all"):
    print("\nmodel A 로드 중...")
    model_a = build_and_load_model(MODEL_A_PATH, DEVICE)
    topo_ds = TopoInMemory(path_1=TOPO_PATH, lat_range=[LAT_MIN, LAT_MAX], lon_range=[LON_MIN, LON_MAX])
    topo = topo_ds[0]
    if isinstance(topo, np.ndarray):
        topo = torch.from_numpy(topo).float()
    else:
        topo = topo.float()
    if topo.ndim == 2:
        topo = topo.unsqueeze(0)
    topo = topo[:1, ...].unsqueeze(0).to(DEVICE)
    model_a.topo = topo

    print("  10km -> 500m SR 실행...")
    sr_10km_500m, lat_500m, lon_500m = run_sr(model_a, input_10km, MODEL_A_MAXV, 0.0, 500, DEVICE)
    print(f"    출력 shape={sr_10km_500m.shape}, max={sr_10km_500m.max():.1f}")

    print("  1km -> 500m SR 실행...")
    sr_1km_500m, _, _ = run_sr(model_a, input_1km, MODEL_A_MAXV, 0.0, 500, DEVICE)
    print(f"    출력 shape={sr_1km_500m.shape}, max={sr_1km_500m.max():.1f}")

    del model_a
    torch.cuda.empty_cache()

    d = load_npz_dict()
    d.update(dict(
        raw_daily=raw_daily, raw_lat=raw_lat, raw_lon=raw_lon,
        input_10km=input_10km, lat_10km=lat_10km, lon_10km=lon_10km,
        input_1km=input_1km, lat_1km=lat_1km, lon_1km=lon_1km,
        sr_10km_500m=sr_10km_500m, sr_1km_500m=sr_1km_500m,
        lat_500m=lat_500m, lon_500m=lon_500m,
    ))
    save_npz_dict(d)

# =============================================================================
# 3) model B: 500m -> 100m
# =============================================================================
if STAGE in ("b", "all"):
    print("\nmodel B 로드 중...")
    model_b = build_and_load_model(MODEL_B_PATH, DEVICE)
    model_b.topo = None

    print("  500m -> 100m SR 실행 (좁은 도메인)...")
    sr_500m_100m, lat_100m, lon_100m = run_sr(
        model_b, raw_hourly, MODEL_B_PMAX, MODEL_B_PMIN, 100, DEVICE,
        lat_min=MODEL_B_LAT_MIN, lat_max=MODEL_B_LAT_MAX,
        lon_min=MODEL_B_LON_MIN, lon_max=MODEL_B_LON_MAX,
    )
    print(f"    출력 shape={sr_500m_100m.shape}, max={sr_500m_100m.max():.1f}")

    del model_b
    torch.cuda.empty_cache()

    d = load_npz_dict()
    d.update(dict(
        raw_hourly=raw_hourly, raw_hourly_lat=raw_hourly_lat, raw_hourly_lon=raw_hourly_lon,
        sr_500m_100m=sr_500m_100m, lat_100m=lat_100m, lon_100m=lon_100m,
    ))
    save_npz_dict(d)

if STAGE in ("a", "b"):
    print(f"\nSTAGE={STAGE} 완료. 전부 끝나면 `python case_multiscale_sr.py plot`으로 그림을 그리세요.")
    sys.exit(0)

# =============================================================================
# 4) 플롯 3장 저장 (STAGE == 'plot' or 'all')
# =============================================================================
if STAGE == "plot":
    d = load_npz_dict()
    raw_daily, raw_lat, raw_lon = d["raw_daily"], d["raw_lat"], d["raw_lon"]
    input_10km, lat_10km, lon_10km = d["input_10km"], d["lat_10km"], d["lon_10km"]
    input_1km, lat_1km, lon_1km = d["input_1km"], d["lat_1km"], d["lon_1km"]
    sr_10km_500m, sr_1km_500m = d["sr_10km_500m"], d["sr_1km_500m"]
    lat_500m, lon_500m = d["lat_500m"], d["lon_500m"]
    raw_hourly, raw_hourly_lat, raw_hourly_lon = d["raw_hourly"], d["raw_hourly_lat"], d["raw_hourly_lon"]
    sr_500m_100m, lat_100m, lon_100m = d["sr_500m_100m"], d["lat_100m"], d["lon_100m"]

vmax_daily = max(raw_daily.max(), sr_10km_500m.max(), sr_1km_500m.max())
vmax_hourly = max(raw_hourly.max(), sr_500m_100m.max())

save_3panel(
    [
        (input_10km, lon_10km, lat_10km, f"input (10km, block-avg)  |  {TARGET_DATE}"),
        (sr_10km_500m, lon_500m, lat_500m, f"model A SR (500m)  |  {TARGET_DATE}"),
        (raw_daily, raw_lon, raw_lat, f"raw HSR truth (native ~500m)  |  {TARGET_DATE}"),
    ],
    os.path.join(SAVE_DIR, f"case_multiscale_10km_to_500m_{CASE_TAG}.png"),
    vmax_daily,
)

save_3panel(
    [
        (input_1km, lon_1km, lat_1km, f"input (1km, block-avg)  |  {TARGET_DATE}"),
        (sr_1km_500m, lon_500m, lat_500m, f"model A SR (500m)  |  {TARGET_DATE}"),
        (raw_daily, raw_lon, raw_lat, f"raw HSR truth (native ~500m)  |  {TARGET_DATE}"),
    ],
    os.path.join(SAVE_DIR, f"case_multiscale_1km_to_500m_{CASE_TAG}.png"),
    vmax_daily,
)

save_3panel(
    [
        (raw_hourly, raw_hourly_lon, raw_hourly_lat,
         f"raw HSR input (native ~500m)  |  {TARGET_DATE} {TARGET_HOUR:02d}:00 KST"),
        (sr_500m_100m, lon_100m, lat_100m,
         f"model B SR (100m)  |  {TARGET_DATE} {TARGET_HOUR:02d}:00 KST"),
    ],
    os.path.join(SAVE_DIR, f"case_multiscale_500m_to_100m_{CASE_TAG}.png"),
    vmax_hourly,
)

print("\n모두 완료.")
