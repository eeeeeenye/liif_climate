"""
HSR 레이더 native 2x SR 모델(_climate_edsr_radar_hsr_2x_minmax_liif)로 전기간(JJA) 인퍼런스.

infer_hsr_radar_sr.py(IMERG로 학습된 모델에 HSR을 억지로 태우는 버전) 와는 모델이 다름:

- 이 체크포인트는 HSR 500m(정답) <- HSR 1000m(2x 다운그레이드, 입력) 쌍으로 직접 학습됨
  (config: climate_edsr_radar_hsr_2x_minmax_liif.yaml). topo 채널 없음(extra_in_dim=0)
  -> model.topo 를 반드시 None으로 둬야 함 (topo를 넣으면 imnet 입력 차원이 안 맞아 에러).
- 정규화 기준도 다름: physical_norm.max = 74.8784 (IMERG 모델의 506.02가 아님!),
  train(2019-2020 JJA)만으로 계산한 값이라 min-max scale이 훨씬 작음.
- 이 모델은 **시간별(hourly)** HSR 필드로 학습됐음(HSR_train_500m_norm.nc의 time=4416=
  2019~2020 JJA 매시간). 그래서 여기서는 하루 합산이 아니라 HSR merged 파일의 매 시간
  스텝을 그대로 입력으로 씀.
- encoder(EDSR no_upsampling, n_resblocks=32/n_feats=256)는 infer_hsr_radar_sr.py와 완전히
  동일 구조라 gen_feat는 반드시 torch.no_grad() 안에서 호출해야 OOM이 안 남.

전기간 모드:
- YEARS = None -> HSR merged 파일에 있는 전체 기간(2019-2021 JJA, 6624시간) 처리.
  너무 오래 걸리면 YEARS = [2019] 로 바꿔서 2019 JJA(약 2208시간)만 먼저 돌릴 것.
- 100m는 시간당 ~7,800만 쿼리포인트라 전기간(6624시간) 기준 파일만 수십 TB.
  `python infer_hsr_radar_2x_sr.py 300` 처럼 인자를 줘서 300m만 먼저 끝내는 걸 권장.
- 중간에 끊겨도 이어서 돌릴 수 있도록, 이미 저장된 파일은 건너뜀(SKIP_EXISTING).

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

# ============================================================
# 1. 기본 설정
# ============================================================

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

HSR_PATH = (
    "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/"
    "HSR_2019_2021_JJA_merged_kst_hourly_2881x2305.nc"
)

CKPT_TAG = "epoch-best-psnr"  # "epoch-best-psnr" | "epoch-best-loss" | "epoch-last"
_RUN_DIR = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif"
MODEL_PATH = f"{_RUN_DIR}/{CKPT_TAG}.pth"

SAVE_DIR = f"{_RUN_DIR}/preds_nc/case_sr/"
os.makedirs(SAVE_DIR, exist_ok=True)

# 이전 infer_hsr_radar_sr.py와 같은 crop(비교용). 이 모델은 topo가 없어서 도메인이 학습과
# 무관하게 자유로움 -> 원하면 그냥 넓혀도 됨.
LAT_MIN, LAT_MAX = 33.0, 43.0
LON_MIN, LON_MAX = 124.0, 132.0

# 처리할 연도(KST). None이면 HSR merged 파일에 있는 전체 기간(2019-2021 JJA) 처리.
# 오래 걸리면 [2019] 로 바꿔서 2019 JJA만 먼저 돌릴 것.
YEARS = [2021]

# 위 YEARS 범위 안에서, 아래 날짜들을 먼저 처리(저장)하고 나머지는 그 뒤 원래 순서로 처리.
# - MANDATORY_DATES: 무조건 먼저 저장할 날짜
#   (2021-08-23: HSR 일누적 도메인최댓값 1위. 2021-07-05: HSR 기준 top15엔 안 들지만
#    ASOS 관측 기준으로는 99퍼센타일 초과 관측소 수/단일관측소 최댓값 모두 2021년 1위라
#    - 이 날 강한 대류성 강수를 레이더가 과소평가했을 가능성 있어 수동으로 추가)
# - EXTREME_TOP_N: MANDATORY_DATES 외에, PRIORITY_YEAR의 남한 도메인(LAT/LON 위 crop)
#   일 누적강수량 기준 상위 top_n일도 자동으로 찾아서 같이 먼저 저장
MANDATORY_DATES = ["2021-08-23", "2021-07-05"]
EXTREME_TOP_N = 15
PRIORITY_YEAR = 2021

# 출력 목표 물리 해상도 (m). 한 번에 하나씩만 돌리고 싶으면
# `python infer_hsr_radar_2x_sr.py 300` 처럼 인자로 넘길 것 (GPU 메모리/디스크 절약,
# 300m부터 먼저 끝내는 걸 권장).
TARGET_RES_M = [100, 300]
if len(sys.argv) > 1:
    TARGET_RES_M = [int(a) for a in sys.argv[1:]]

BSIZE = 500_000

# 이미 저장된 (시간, 해상도) 조합은 다시 안 돌리고 건너뜀 -> 중간에 끊겨도 이어서 실행 가능
SKIP_EXISTING = True

# physical_norm.min/max (config.yaml 그대로, train(2019-2020 JJA)만으로 계산된 값)
PMIN = 0.0
PMAX = 74.8784

CLAMP_01 = True
CLAMP_MM_ZERO = True


# ============================================================
# 2. 유틸 함수
# ============================================================

def crop_idx(coord_full, vmin, vmax):
    idx = np.where((coord_full >= vmin) & (coord_full <= vmax))[0]
    if len(idx) == 0:
        raise ValueError(f"[{vmin}, {vmax}] 범위에 해당하는 격자가 없습니다.")
    return idx


def open_hsr(hsr_path, lat_min, lat_max, lon_min, lon_max, years=None):
    """
    HSR merged nc를 한 번만 열고, crop 인덱스와 처리할 시간 index 목록을 반환.
    (전기간 루프에서 매 시간마다 파일을 새로 열지 않기 위해 ds를 열어둔 채로 돌려줌 ->
    호출부에서 다 쓴 뒤 ds.close() 할 것.)
    """
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
    """지정 시간 index 하나의 crop된 (H,W) 강수 배열."""
    arr = np.asarray(
        ds.variables["precip"][t_idx, lat_idx[0]:lat_idx[-1] + 1, lon_idx[0]:lon_idx[-1] + 1],
        float,
    )
    arr[arr < -1e8] = np.nan
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    return arr


def find_extreme_dates(ds, time_arr, lat_idx, lon_idx, year, top_n):
    """
    지정 연도의 날짜별로 (LAT/LON crop 도메인 내) 격자별 24시간 누적강수량을 구하고,
    그 도메인 최댓값이 큰 순서로 top_n개 날짜를 반환.
    """
    l0, l1 = lat_idx[0], lat_idx[-1] + 1
    o0, o1 = lon_idx[0], lon_idx[-1] + 1
    year_t_idx = np.where(time_arr.year == year)[0]

    daily_accum = {}
    for ti in tqdm(year_t_idx, desc="    scanning extreme days", leave=False):
        arr = np.asarray(ds.variables["precip"][ti, l0:l1, o0:o1], float)
        arr[arr < -1e8] = np.nan
        arr = np.nan_to_num(arr, nan=0.0)
        d = time_arr[ti].date()
        if d not in daily_accum:
            daily_accum[d] = np.zeros_like(arr)
        daily_accum[d] += arr

    ranked = sorted(daily_accum.items(), key=lambda kv: kv[1].max(), reverse=True)
    return [d.strftime("%Y-%m-%d") for d, _ in ranked[:top_n]]


def build_and_load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"checkpoint에 model 정보가 없음 keys={list(ckpt.keys())}")
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


# ============================================================
# 3. 모델 로드 (날짜/해상도와 무관하게 한 번만). topo 없음.
# ============================================================
model = build_and_load_model(MODEL_PATH, device)

out_grids = {res_m: out_grid_for_resolution(res_m, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
             for res_m in TARGET_RES_M}
for res_m, (H, W, _, _) in out_grids.items():
    print(f"target {res_m}m -> out grid {H} x {W} = {H * W:,} query points")

# ============================================================
# 4. 전기간 추론
# ============================================================
ds, time_arr, lat_idx, lon_idx, hsr_lat, hsr_lon, t_idx_all = open_hsr(
    HSR_PATH, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, years=YEARS
)
print(f"\n처리 대상 시간 스텝: {len(t_idx_all)}개 "
      f"({time_arr[t_idx_all[0]]} ~ {time_arr[t_idx_all[-1]]} KST, YEARS={YEARS})")

# 지정 날짜 + 극한강수일을 먼저 처리하도록 t_idx_all 순서 재배열
extreme_dates = find_extreme_dates(ds, time_arr, lat_idx, lon_idx, PRIORITY_YEAR, EXTREME_TOP_N)
priority_dates = list(dict.fromkeys(MANDATORY_DATES + extreme_dates))  # 순서 유지, 중복 제거
print(f"우선 처리 날짜 {len(priority_dates)}개 (지정 {len(MANDATORY_DATES)}개 + "
      f"{PRIORITY_YEAR} 극한강수 상위 {EXTREME_TOP_N}개): {priority_dates}")

date_strs = pd.to_datetime(time_arr[t_idx_all]).strftime("%Y-%m-%d").values
is_priority = np.isin(date_strs, priority_dates)
t_idx_all = np.concatenate([t_idx_all[is_priority], t_idx_all[~is_priority]])

n_done, n_skip = 0, 0
for t_idx in tqdm(t_idx_all, desc="hours"):
    used_ts = time_arr[t_idx]
    ts_tag = pd.Timestamp(used_ts).strftime("%Y-%m-%d_%H%M")

    # 이 시간에 필요한 해상도들 중 이미 저장 안 된 게 있는지 먼저 확인 (있으면 read+encode 자체를 건너뜀)
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
        model.topo = None  # 이 모델은 topo 채널 없이 학습됨

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
