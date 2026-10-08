"""
model B(_climate_edsr_radar_hsr_2x_minmax_liif)로, raw HSR(native ~500m)을 인위적으로
2km / 5km로 다운샘플링한 입력을 다시 500m로 SR하는 실험을 2021 JJA 전체 시간(2208시간)에
대해 돌린다.

case_multiscale_modelB_2km_5km.py(한 시각만)의 전기간 버전.
- 입력이 2km/5km로 이미 성긴 상태라 encoder feat map이 작아서 GPU 메모리 문제 없이
  전체 도메인(33-43N, 124-132E)을 그대로 씀.
- infer_hsr_radar_2x_sr.py와 같은 패턴으로 SKIP_EXISTING 지원 -> 중간에 끊겨도 이어서 실행 가능.
- 시간이 오래 걸리는 작업이라(추정 수 시간) 시간별 진행상황 + 소요시간을 stdout과
  로그 파일(모두 SAVE_DIR/timing.log)에 같이 남김.

실행:
  conda activate liif
  python infer_hsr_radar_2x_sr_2km_5km_2021jja.py          # 2km, 5km 둘 다
  python infer_hsr_radar_2x_sr_2km_5km_2021jja.py 2km      # 2km만
  python infer_hsr_radar_2x_sr_2km_5km_2021jja.py 5km      # 5km만

env: /home/inhye_yoo/anaconda3/envs/liif/bin/python
"""
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset, num2date
from tqdm import tqdm

sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
from models.models import make
from utils import make_coord

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

HSR_PATH = (
    "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/"
    "HSR_2019_2021_JJA_merged_kst_hourly_2881x2305.nc"
)

CKPT_TAG = "epoch-best-psnr"
_RUN_DIR = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif"
MODEL_PATH = f"{_RUN_DIR}/{CKPT_TAG}.pth"

SAVE_DIR = f"{_RUN_DIR}/preds_nc/multiscale_2km5km_500m/"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_PATH = os.path.join(SAVE_DIR, "timing.log")

LAT_MIN, LAT_MAX = 33.0, 43.0
LON_MIN, LON_MAX = 124.0, 132.0

YEARS = [2021]  # HSR merged 파일에 있는 연도(2019-2021 JJA) 중 처리할 연도

# 인위적 입력 다운샘플 해상도(m) -> 출력은 항상 500m
INPUT_RES_M = ["2km", "5km"]
INPUT_RES_TO_M = {"2km": 2_000, "5km": 5_000}
if len(sys.argv) > 1:
    INPUT_RES_M = sys.argv[1:]

OUT_RES_M = 500

BSIZE = 500_000
SKIP_EXISTING = True

PMIN = 0.0
PMAX = 74.8784  # mm/hr 기준 (infer_hsr_radar_2x_sr.py와 동일 physical_norm)

CLAMP_01 = True
CLAMP_MM_ZERO = True


# ============================================================
# 타이밍 로그
# ============================================================
def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ============================================================
# 유틸 함수
# ============================================================
def crop_idx(coord_full, vmin, vmax):
    idx = np.where((coord_full >= vmin) & (coord_full <= vmax))[0]
    if len(idx) == 0:
        raise ValueError(f"[{vmin}, {vmax}] 범위에 해당하는 격자가 없습니다.")
    return idx


def open_hsr(hsr_path, lat_min, lat_max, lon_min, lon_max, years=None):
    ds = Dataset(hsr_path)
    time_var = ds.variables["time"]
    time_arr = pd.to_datetime(
        num2date(time_var[:], time_var.units,
                 calendar=getattr(time_var, "calendar", "standard"),
                 only_use_cftime_datetimes=False)
    )
    lat_full = np.asarray(ds.variables["lat"][:], float)
    lon_full = np.asarray(ds.variables["lon"][:], float)
    assert lat_full[0] < lat_full[-1] and lon_full[0] < lon_full[-1], \
        "HSR lat/lon이 오름차순이 아닙니다."

    lat_idx = crop_idx(lat_full, lat_min, lat_max)
    lon_idx = crop_idx(lon_full, lon_min, lon_max)
    lat_sub = lat_full[lat_idx[0]:lat_idx[-1] + 1]
    lon_sub = lon_full[lon_idx[0]:lon_idx[-1] + 1]

    if years is None:
        t_idx_all = np.arange(len(time_arr))
    else:
        t_idx_all = np.where(time_arr.year.isin(years))[0]

    return ds, time_arr, lat_idx, lon_idx, lat_sub, lon_sub, t_idx_all


def get_hour_array(ds, t_idx, lat_idx, lon_idx):
    arr = np.asarray(
        ds.variables["precip"][t_idx, lat_idx[0]:lat_idx[-1] + 1, lon_idx[0]:lon_idx[-1] + 1],
        float,
    )
    arr[arr < -1e8] = np.nan
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    return arr


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
    if "model" not in ckpt:
        raise KeyError(f"checkpoint에 model 정보가 없음 keys={list(ckpt.keys())}")
    model_spec = copy.deepcopy(ckpt["model"])
    log(f"checkpoint model name: {model_spec['name']}")
    model = make({"name": model_spec["name"], "args": model_spec["args"]}).to(device)
    model.load_state_dict(model_spec["sd"], strict=True)
    model.eval()
    log("Model loaded successfully")
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


def save_pred_nc(path, precip, lat_asc, lon_asc, used_ts, in_res_tag, out_res_m):
    ds = Dataset(path, "w", format="NETCDF4")
    ds.createDimension("lat", len(lat_asc))
    ds.createDimension("lon", len(lon_asc))
    v_lat = ds.createVariable("lat", "f4", ("lat",))
    v_lon = ds.createVariable("lon", "f4", ("lon",))
    v_p = ds.createVariable("precip", "f4", ("lat", "lon"), zlib=True, complevel=4)
    v_lat[:] = lat_asc
    v_lon[:] = lon_asc
    v_p[:, :] = precip
    v_p.units = "mm/hour"
    ds.title = f"LIIF(HSR-native 2x) SR, input artificially degraded to {in_res_tag}, {used_ts} KST, target {out_res_m}m"
    ds.source = HSR_PATH
    ds.model_ckpt = MODEL_PATH
    ds.close()


# ============================================================
# 모델 로드 (한 번만)
# ============================================================
script_t0 = time.perf_counter()
log(f"===== infer_hsr_radar_2x_sr_2km_5km_2021jja.py 시작 (입력 해상도={INPUT_RES_M}) =====")

model = build_and_load_model(MODEL_PATH, device)
model.topo = None

H, W, out_lat, out_lon = out_grid_for_resolution(OUT_RES_M, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
log(f"target {OUT_RES_M}m -> out grid {H} x {W} = {H * W:,} query points")

in_grids = {}
for tag in INPUT_RES_M:
    res_m = INPUT_RES_TO_M[tag]
    _, _, in_lat, in_lon = out_grid_for_resolution(res_m, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    in_grids[tag] = (in_lat, in_lon)

ds, time_arr, lat_idx, lon_idx, raw_lat, raw_lon, t_idx_all = open_hsr(
    HSR_PATH, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, years=YEARS
)
log(f"처리 대상 시간 스텝: {len(t_idx_all)}개 "
    f"({time_arr[t_idx_all[0]]} ~ {time_arr[t_idx_all[-1]]} KST, YEARS={YEARS})")

n_done, n_skip = 0, 0
per_hour_times = []

for t_idx in tqdm(t_idx_all, desc="hours"):
    hour_t0 = time.perf_counter()
    used_ts = time_arr[t_idx]
    ts_tag = pd.Timestamp(used_ts).strftime("%Y-%m-%d_%H%M")

    todo = []
    for tag in INPUT_RES_M:
        out_path = os.path.join(SAVE_DIR, f"hsr2x_liif_sr_in{tag}_{OUT_RES_M}m_{ts_tag}kst.nc")
        if SKIP_EXISTING and os.path.exists(out_path):
            n_skip += 1
        else:
            todo.append((tag, out_path))
    if not todo:
        continue

    raw_snap = get_hour_array(ds, t_idx, lat_idx, lon_idx)

    for tag, out_path in todo:
        in_lat, in_lon = in_grids[tag]
        inp_2d = block_average_downsample(raw_snap, raw_lat, raw_lon, in_lat, in_lon)

        x_norm = (inp_2d.astype(np.float32) - PMIN) / (PMAX - PMIN)
        inp_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            model.gen_feat(inp_tensor)

            coord = make_coord(shape=(H, W)).unsqueeze(0).to(device)
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2.0 / H
            cell[:, :, 1] *= 2.0 / W

            pred = chunked_query(model, coord, cell, BSIZE)
            pred = pred.view(H, W)

            if CLAMP_01:
                pred = pred.clamp(0.0, 1.0)
            pred_mm = pred * (PMAX - PMIN) + PMIN
            if CLAMP_MM_ZERO:
                pred_mm = pred_mm.clamp(min=0.0)
            pred_mm = pred_mm.detach().cpu().numpy()

            del coord, cell, pred
            torch.cuda.empty_cache()

        save_pred_nc(out_path, pred_mm, out_lat, out_lon, used_ts, tag, OUT_RES_M)
        n_done += 1

    del inp_tensor
    torch.cuda.empty_cache()

    hour_elapsed = time.perf_counter() - hour_t0
    per_hour_times.append(hour_elapsed)
    if len(per_hour_times) % 50 == 0:
        avg = np.mean(per_hour_times[-50:])
        remain = len(t_idx_all) - (n_done + n_skip) / max(len(todo), 1)
        eta_min = avg * remain / 60.0
        log(f"  진행: {ts_tag}  최근50개 평균 {avg:.2f}초/시간  누적저장={n_done}  스킵={n_skip}  ETA~{eta_min:.1f}분")

ds.close()
total_elapsed = time.perf_counter() - script_t0
log(f"\ndone. saved={n_done}  skipped(existing)={n_skip}  총 소요시간={total_elapsed/60:.1f}분")
