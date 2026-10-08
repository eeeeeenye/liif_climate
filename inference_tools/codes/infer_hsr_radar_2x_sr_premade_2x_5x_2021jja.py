"""
model B(_climate_edsr_radar_hsr_2x_minmax_liif)로, 학습 때와 똑같은 방식(cdo remapbil)으로
미리 만들어둔 HSR_2021_norm_2x.nc(≈1km, 2x 다운그레이드) / HSR_2021_norm_5x.nc(≈2.5km, 5x
다운그레이드)를 그대로 입력으로 넣어 2021 JJA 전체를 500m로 SR한다.

case_multiscale_modelB_2km_5km.py / infer_hsr_radar_2x_sr_2km_5km_2021jja.py와 다른 점:
- 입력을 내가 block-average로 즉석에서 만드는 게 아니라, 학습 파이프라인과 동일한
  cdo remapbil로 미리 만들어둔 nc를 그대로 읽음 (train/inference 다운샘플 방식 일치).
- 이미 정규화된 값(precip_norm = precip/74.8784)이 들어있어서 다시 나눌 필요 없음.
- 입력 파일의 위경도 범위(HSR merged 전체 도메인, ~119.46-132.9E/30.3-43.57N)를 그대로
  따라가서 output도 같은 물리 영역에 놓이도록 함 (33-43/124-132로 임의로 자르지 않음).

시간이 오래 걸리는 작업이라 stdout + 로그 파일(SAVE_DIR/timing.log)에 진행상황/소요시간 기록.
SKIP_EXISTING으로 중간에 끊겨도 이어서 실행 가능.

실행:
  conda activate liif
  python infer_hsr_radar_2x_sr_premade_2x_5x_2021jja.py         # 2x, 5x 둘 다
  python infer_hsr_radar_2x_sr_premade_2x_5x_2021jja.py 2x      # 2x만
  python infer_hsr_radar_2x_sr_premade_2x_5x_2021jja.py 5x      # 5x만

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

INPUT_PATHS = {
    "2x": "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/HSR_2021_norm_2x.nc",
    "5x": "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/HSR_2021_norm_5x.nc",
}

CKPT_TAG = "epoch-best-psnr"
_RUN_DIR = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif"
MODEL_PATH = f"{_RUN_DIR}/{CKPT_TAG}.pth"

SAVE_DIR = f"{_RUN_DIR}/preds_nc/premade_2x5x_500m/"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_PATH = os.path.join(SAVE_DIR, "timing.log")

OUT_RES_M = 500

TAGS = ["2x", "5x"]
if len(sys.argv) > 1:
    TAGS = sys.argv[1:]

BSIZE = 500_000
SKIP_EXISTING = True

PMIN, PMAX = 0.0, 74.8784  # 이미 이 값으로 정규화된 입력이지만, 출력 복원에 필요
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
# 유틸
# ============================================================
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


def build_and_load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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


def save_pred_nc(path, precip, lat_asc, lon_asc, used_ts, in_tag, out_res_m):
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
    ds.title = f"LIIF(HSR-native 2x) SR, premade {in_tag} input(cdo remapbil), {used_ts} KST, target {out_res_m}m"
    ds.source = INPUT_PATHS[in_tag]
    ds.model_ckpt = MODEL_PATH
    ds.close()


# ============================================================
# 실행
# ============================================================
script_t0 = time.perf_counter()
log(f"===== infer_hsr_radar_2x_sr_premade_2x_5x_2021jja.py 시작 (TAGS={TAGS}) =====")

model = build_and_load_model(MODEL_PATH, device)
model.topo = None

for tag in TAGS:
    in_path = INPUT_PATHS[tag]
    tag_save_dir = os.path.join(SAVE_DIR, tag)
    os.makedirs(tag_save_dir, exist_ok=True)

    t_stage0 = time.perf_counter()
    log(f"\n----- {tag} 입력 처리 시작: {in_path} -----")

    ds = Dataset(in_path)
    time_var = ds.variables["time"]
    time_arr = pd.to_datetime(
        num2date(time_var[:], time_var.units,
                 calendar=getattr(time_var, "calendar", "standard"),
                 only_use_cftime_datetimes=False)
    )
    in_lat = np.asarray(ds.variables["lat"][:], float)
    in_lon = np.asarray(ds.variables["lon"][:], float)
    assert in_lat[0] < in_lat[-1] and in_lon[0] < in_lon[-1], "입력 lat/lon이 오름차순이 아닙니다."

    lat_min, lat_max = float(in_lat.min()), float(in_lat.max())
    lon_min, lon_max = float(in_lon.min()), float(in_lon.max())
    log(f"  입력 시간 스텝: {len(time_arr)}개 ({time_arr[0]} ~ {time_arr[-1]} KST)")
    log(f"  입력 도메인: lat {lat_min:.3f}-{lat_max:.3f}, lon {lon_min:.3f}-{lon_max:.3f}, "
        f"shape={in_lat.shape[0]}x{in_lon.shape[0]}")

    H, W, out_lat, out_lon = out_grid_for_resolution(OUT_RES_M, lat_min, lat_max, lon_min, lon_max)
    log(f"  target {OUT_RES_M}m -> out grid {H} x {W} = {H * W:,} query points")

    n_done, n_skip = 0, 0
    per_hour_times = []

    for i, t_idx in enumerate(tqdm(range(len(time_arr)), desc=f"{tag} hours")):
        hour_t0 = time.perf_counter()
        used_ts = time_arr[t_idx]
        ts_tag = pd.Timestamp(used_ts).strftime("%Y-%m-%d_%H%M")
        out_path = os.path.join(tag_save_dir, f"hsr2x_liif_sr_premade{tag}_{OUT_RES_M}m_{ts_tag}kst.nc")

        if SKIP_EXISTING and os.path.exists(out_path):
            n_skip += 1
            continue

        inp_2d = np.asarray(ds.variables["precip_norm"][t_idx, :, :], np.float32)
        # raw HSR의 결측 sentinel(-999000000 등)이 정규화돼도 그대로 남아있어서, NaN이
        # 아니라 매우 큰 음수로 존재함 (다른 infer 스크립트의 `arr < -1e8` 필터와 동일 기준).
        inp_2d[inp_2d < -1e8] = 0.0
        inp_2d = np.nan_to_num(inp_2d, nan=0.0)
        inp_tensor = torch.from_numpy(inp_2d).float().unsqueeze(0).unsqueeze(0).to(device)

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
            remain = len(time_arr) - (i + 1)
            eta_min = avg * remain / 60.0
            log(f"  [{tag}] 진행: {ts_tag}  최근50개 평균 {avg:.2f}초/시간  "
                f"누적저장={n_done}  스킵={n_skip}  ETA~{eta_min:.1f}분")

    ds.close()
    stage_elapsed = time.perf_counter() - t_stage0
    log(f"----- {tag} 완료. saved={n_done} skipped={n_skip} 소요시간={stage_elapsed/60:.1f}분 -----")

total_elapsed = time.perf_counter() - script_t0
log(f"\n전체 완료. 총 소요시간={total_elapsed/60:.1f}분")
