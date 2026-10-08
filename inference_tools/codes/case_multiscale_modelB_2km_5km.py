"""
model B(_climate_edsr_radar_hsr_2x_minmax_liif)로 raw HSR(native ~500m)을 인위적으로
2km / 5km로 다운샘플링한 뒤 다시 500m로 SR하는 실험.
case_multiscale_sr.py의 10km/1km->500m(model A) 실험과 같은 계열이지만, model B로
같은 걸 해보는 별도 파일 (model B 요청이라 케이스 파일 자체를 분리).

입력이 2km/5km로 이미 성긴 상태라 encoder feat map이 작아서(native 500m 입력 때와
달리) GPU 메모리 문제 없이 전체 도메인(33-43N, 124-132E)을 그대로 쓴다.

각 단계 소요시간을 stdout과 로그 파일(case_multiscale_modelB_2km5km_timing.log)에
같이 남긴다.

env: /home/inhye_yoo/anaconda3/envs/liif/bin/python
"""
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset, date2index
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
from models.models import make
from utils import make_coord

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HSR_PATH = (
    "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/"
    "HSR_2019_2021_JJA_merged_kst_hourly_2881x2305.nc"
)
MODEL_B_PATH = ("/home/inhye_yoo/deeplearning/inr/liif/save/"
                 "_climate_edsr_radar_hsr_2x_minmax_liif/epoch-best-psnr.pth")

LAT_MIN, LAT_MAX = 33.0, 43.0
LON_MIN, LON_MAX = 124.0, 132.0

TARGET_DATE = "2021-08-23"
TARGET_HOUR = 16  # KST (해당일 raw 최댓값 시각, 이전 케이스와 동일)
CASE_TAG = "2021-08-23"

SAVE_DIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/case_studies/multiscale"
LOG_PATH = os.path.join(SAVE_DIR, "case_multiscale_modelB_2km5km_timing.log")

MODEL_B_PMIN, MODEL_B_PMAX = 0.0, 74.8784  # mm/hr 기준 (infer_hsr_radar_2x_sr.py와 동일)
BSIZE = 500_000


# =============================================================================
# 타이밍 로그
# =============================================================================
def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        log(f"시작: {self.label}")
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.t0
        log(f"완료: {self.label}  ({elapsed:.1f}초)")


# =============================================================================
# 공통 유틸
# =============================================================================
def crop_idx(coord_full, vmin, vmax):
    idx = np.where((coord_full >= vmin) & (coord_full <= vmax))[0]
    if len(idx) == 0:
        raise ValueError(f"[{vmin}, {vmax}] 범위에 해당하는 격자가 없습니다.")
    return idx


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
    log(f"checkpoint model name: {model_spec['name']}")
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
    """panels: [(data, lon, lat, title), ...]"""
    bounds = make_bounds_5(vmax)
    cmap = plt.get_cmap("turbo")
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="max")

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 6),
                              subplot_kw={"projection": ccrs.PlateCarree()})
    if len(panels) == 1:
        axes = [axes]
    for ax, (data, lon, lat, title) in zip(axes, panels):
        plot_panel(ax, data, lon, lat, title, cmap, norm, bounds, "Precipitation (mm/hr)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"저장 완료: {save_path}")


# =============================================================================
# 실행
# =============================================================================
script_t0 = time.perf_counter()
log(f"===== case_multiscale_modelB_2km_5km.py 시작 (date={TARGET_DATE} {TARGET_HOUR:02d}:00 KST) =====")

with Timer("raw HSR hourly 로드 (전체 도메인)"):
    raw_hourly, raw_lat, raw_lon = load_hsr_hourly(
        HSR_PATH, TARGET_DATE, TARGET_HOUR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    )
    log(f"  raw hourly shape={raw_hourly.shape}, max={raw_hourly.max():.1f} mm/hr")

with Timer("2km / 5km block-average 다운샘플"):
    _, _, lat_2km, lon_2km = out_grid_for_resolution(2_000, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    _, _, lat_5km, lon_5km = out_grid_for_resolution(5_000, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    input_2km = block_average_downsample(raw_hourly, raw_lat, raw_lon, lat_2km, lon_2km)
    input_5km = block_average_downsample(raw_hourly, raw_lat, raw_lon, lat_5km, lon_5km)
    log(f"  2km 입력 shape={input_2km.shape}, max={input_2km.max():.1f}")
    log(f"  5km 입력 shape={input_5km.shape}, max={input_5km.max():.1f}")

with Timer("model B 체크포인트 로드"):
    model_b = build_and_load_model(MODEL_B_PATH, DEVICE)
    model_b.topo = None

with Timer("2km -> 500m SR"):
    sr_2km_500m, lat_500m, lon_500m = run_sr(model_b, input_2km, MODEL_B_PMAX, MODEL_B_PMIN, 500, DEVICE)
    log(f"  출력 shape={sr_2km_500m.shape}, max={sr_2km_500m.max():.1f}")

with Timer("5km -> 500m SR"):
    sr_5km_500m, _, _ = run_sr(model_b, input_5km, MODEL_B_PMAX, MODEL_B_PMIN, 500, DEVICE)
    log(f"  출력 shape={sr_5km_500m.shape}, max={sr_5km_500m.max():.1f}")

del model_b
torch.cuda.empty_cache()

vmax = max(raw_hourly.max(), sr_2km_500m.max(), sr_5km_500m.max())

with Timer("플롯 저장 (2km -> 500m)"):
    save_3panel(
        [
            (input_2km, lon_2km, lat_2km, f"input (2km, block-avg)  |  {TARGET_DATE} {TARGET_HOUR:02d}:00 KST"),
            (sr_2km_500m, lon_500m, lat_500m, f"model B SR (500m)  |  {TARGET_DATE} {TARGET_HOUR:02d}:00 KST"),
            (raw_hourly, raw_lon, raw_lat, f"raw HSR truth (native ~500m)  |  {TARGET_DATE} {TARGET_HOUR:02d}:00 KST"),
        ],
        os.path.join(SAVE_DIR, f"case_multiscale_modelB_2km_to_500m_{CASE_TAG}.png"),
        vmax,
    )

with Timer("플롯 저장 (5km -> 500m)"):
    save_3panel(
        [
            (input_5km, lon_5km, lat_5km, f"input (5km, block-avg)  |  {TARGET_DATE} {TARGET_HOUR:02d}:00 KST"),
            (sr_5km_500m, lon_500m, lat_500m, f"model B SR (500m)  |  {TARGET_DATE} {TARGET_HOUR:02d}:00 KST"),
            (raw_hourly, raw_lon, raw_lat, f"raw HSR truth (native ~500m)  |  {TARGET_DATE} {TARGET_HOUR:02d}:00 KST"),
        ],
        os.path.join(SAVE_DIR, f"case_multiscale_modelB_5km_to_500m_{CASE_TAG}.png"),
        vmax,
    )

log(f"===== 전체 완료. 총 소요시간 {time.perf_counter() - script_t0:.1f}초 =====")
