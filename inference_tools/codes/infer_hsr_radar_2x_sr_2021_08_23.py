"""
infer_hsr_radar_2x_sr.py의 단발성 버전: model B(_climate_edsr_radar_hsr_2x_minmax_liif)로
2021-08-23 하루(24시간, KST)만, 300m 해상도로 추론해서 case_sr/에 저장.

model A(_climate_edsr_topo_80x100_topo_precip_minmax)와 같은 날짜로 case 비교 그림을
그리기 위해 필요한 최소 실행분만 돌린다 (전체 2021 JJA를 다 돌리는 원본 스크립트는
GPU가 이미 다른 작업으로 사용 중이라 너무 오래 걸림).

env: /home/inhye_yoo/anaconda3/envs/liif/bin/python
"""
import copy
import os
import sys

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

SAVE_DIR = f"{_RUN_DIR}/preds_nc/case_sr/"
os.makedirs(SAVE_DIR, exist_ok=True)

LAT_MIN, LAT_MAX = 33.0, 43.0
LON_MIN, LON_MAX = 124.0, 132.0

TARGET_DATE = "2021-08-23"
TARGET_RES_M = [300]

BSIZE = 500_000
SKIP_EXISTING = True

PMIN = 0.0
PMAX = 74.8784

CLAMP_01 = True
CLAMP_MM_ZERO = True


def crop_idx(coord_full, vmin, vmax):
    idx = np.where((coord_full >= vmin) & (coord_full <= vmax))[0]
    if len(idx) == 0:
        raise ValueError(f"[{vmin}, {vmax}] 범위에 해당하는 격자가 없습니다.")
    return idx


def get_hour_array(ds, t_idx, lat_idx, lon_idx):
    arr = np.asarray(
        ds.variables["precip"][t_idx, lat_idx[0]:lat_idx[-1] + 1, lon_idx[0]:lon_idx[-1] + 1],
        float,
    )
    arr[arr < -1e8] = np.nan
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    return arr


def build_and_load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_spec = copy.deepcopy(ckpt["model"])
    print("checkpoint model name:", model_spec["name"])
    model = make({"name": model_spec["name"], "args": model_spec["args"]}).to(device)
    model.load_state_dict(model_spec["sd"], strict=True)
    model.eval()
    print("Model loaded successfully")
    return model


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


@torch.no_grad()
def chunked_query(model, coord, cell, bsize):
    n = coord.shape[1]
    preds = []
    for ql in tqdm(range(0, n, bsize), desc="    query", leave=False):
        qr = min(ql + bsize, n)
        pred = model.query_rgb(coord[:, ql:qr, :], cell[:, ql:qr, :])
        preds.append(pred)
    return torch.cat(preds, dim=1)


def save_pred_nc(path, precip, lat_asc, lon_asc, used_ts, res_m):
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
    ds.title = f"LIIF(HSR-native 2x) SR, {used_ts} KST, target {res_m}m"
    ds.source = HSR_PATH
    ds.model_ckpt = MODEL_PATH
    ds.close()


model = build_and_load_model(MODEL_PATH, device)

out_grids = {res_m: out_grid_for_resolution(res_m, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
             for res_m in TARGET_RES_M}
for res_m, (H, W, _, _) in out_grids.items():
    print(f"target {res_m}m -> out grid {H} x {W} = {H * W:,} query points")

ds = Dataset(HSR_PATH)
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

lat_idx = crop_idx(lat_full, LAT_MIN, LAT_MAX)
lon_idx = crop_idx(lon_full, LON_MIN, LON_MAX)

t_idx_all = np.where(pd.to_datetime(time_arr).strftime("%Y-%m-%d") == TARGET_DATE)[0]
if len(t_idx_all) == 0:
    raise ValueError(f"HSR merged 파일에 {TARGET_DATE} 시간 슬라이스가 없습니다.")
print(f"처리 대상: {TARGET_DATE}, 시간 스텝 {len(t_idx_all)}개")

n_done, n_skip = 0, 0
for t_idx in tqdm(t_idx_all, desc="hours"):
    used_ts = time_arr[t_idx]
    ts_tag = pd.Timestamp(used_ts).strftime("%Y-%m-%d_%H%M")

    todo_res = []
    for res_m in TARGET_RES_M:
        out_path = os.path.join(SAVE_DIR, f"hsr2x_liif_sr_{res_m}m_{ts_tag}kst.nc")
        if SKIP_EXISTING and os.path.exists(out_path):
            n_skip += 1
        else:
            todo_res.append((res_m, out_path))
    if not todo_res:
        continue

    snap = get_hour_array(ds, t_idx, lat_idx, lon_idx)
    x_norm = (snap.astype(np.float32) - PMIN) / (PMAX - PMIN)
    inp_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        model.gen_feat(inp_tensor)
        model.topo = None

        for res_m, out_path in todo_res:
            H, W, lat_out, lon_out = out_grids[res_m]

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

            save_pred_nc(out_path, pred_mm, lat_out, lon_out, used_ts, res_m)
            n_done += 1

    del inp_tensor
    torch.cuda.empty_cache()

ds.close()
print(f"\ndone. saved={n_done}  skipped(existing)={n_skip}")
