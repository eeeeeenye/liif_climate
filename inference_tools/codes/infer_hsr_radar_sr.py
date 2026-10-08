"""
HSR 레이더(500m) 를 LIIF 강수 SR 모델(IMERG 0.2->0.1deg 로 학습됨)에 입력으로 넣어
100m / 300m 해상도로 초고해상도화하는 인퍼런스. 2021 JJA(6-8월) 전체를 일 단위로 처리.

- 모델    : _climate_edsr_topo_80x100_topo_precip_minmax (adapter/correction_layer 없음,
            layer_spec 미설정 -> correction_layer는 identity)
- 입력    : HSR_2019_2021_JJA_merged_kst_hourly_2881x2305.nc (KST, 시간별 정확도 약 500m,
            lat/lon 모두 오름차순) 를 도메인으로 crop 후 KST 하루(00~23시) 합산 -> 일 강수량.
            네트워크는 fully-convolutional encoder(EDSR, no_upsampling)라 학습 때(40x50)와
            전혀 다른 입력 해상도를 그대로 넣어도 forward는 되지만, 학습분포 밖 입력이라는
            점은 감안할 것.
- topo    : ETOPO1_Ice_g_gmt4_kor.nc (make_preds.py 와 동일 recipe, lat/lon 오름차순)
- 정규화  : minmax, mm/day = value * 506.01996 (min=0), data_norm 없음 -> train_liif.py 기본
            항등변환과 동일 파이프라인 (train_imerg80x100_meta.json 기준 전역 min/max)
- 출력    : LIIF는 좌표 기반 continuous decoder라 encoder는 한 번만 돌리고(gen_feat), 해상도만
            다른 query 좌표(coord/cell)를 여러 번 넣어 100m/300m 두 해상도를 동시에 뽑음.
            query 개수가 도메인 전체 기준 수천만~수억 개로 커질 수 있어 bsize 단위로 나눠서
            query_rgb를 호출(batched query). feat_unfold(3x3)는 query_rgb 호출마다 다시
            계산되므로 bsize를 너무 작게 잡으면 같은 unfold를 불필요하게 반복하게 됨 -> 너무
            느리면 BSIZE를 늘리거나 CROP 영역을 좁혀서 쓸 것.
- 좌표계  : HSR/topo/저장 출력 모두 lat 오름차순(남->북)으로 통일. make_coord 의 좌표 index 0 은
            항상 각 텐서의 row index 0 에 대응하므로, 세 텐서(입력/topo/출력)가 같은 lat 방향
            convention을 쓰지 않으면 남북이 뒤집힌 채로 섞이게 된다.

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
from datasets.image_folder import TopoInMemory
from utils import make_coord

# ============================================================
# 1. 기본 설정
# ============================================================

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

HSR_PATH = (
    "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/"
    "HSR_2019_2021_JJA_merged_kst_hourly_2881x2305.nc"
)
TOPO_PATH = "/home/inhye_yoo/data/level/NOAA_ETOPO1_topo/ETOPO1_Ice_g_gmt4_kor.nc"

CKPT_TAG = "epoch-best-psnr"  # "epoch-best-psnr" | "epoch-best-loss" | "epoch-last"
_RUN_DIR = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax"
MODEL_PATH = f"{_RUN_DIR}/{CKPT_TAG}.pth"

SAVE_DIR = f"{_RUN_DIR}/hsr_radar_sr_2021jja/"
os.makedirs(SAVE_DIR, exist_ok=True)

# 학습 도메인과 동일 (topo/모델이 이 범위로 검증됨). 필요하면 좁혀서 테스트할 것.
LAT_MIN, LAT_MAX = 33.0, 43.0
LON_MIN, LON_MAX = 124.0, 132.0

# 추론할 날짜(day, KST). 2021 JJA 전체(6/1~8/31, 92일). HSR merged 파일이 2019-2021 JJA만
# 있음. 다른 연도로 바꾸려면 DATE_RANGE를 수정할 것.
DATE_RANGE = ("2021-06-01", "2021-08-31")
DATES = [d.strftime("%Y-%m-%d") for d in pd.date_range(*DATE_RANGE, freq="D")]

# 출력 목표 물리 해상도 (m). 한 번에 하나씩만 돌리고 싶으면
# `python infer_hsr_radar_sr.py 300` 처럼 인자로 넘길 것 (GPU 메모리/디스크 절약,
# 300m부터 먼저 끝내는 걸 권장).
TARGET_RES_M = [100, 300]
if len(sys.argv) > 1:
    TARGET_RES_M = [int(a) for a in sys.argv[1:]]

# 이미 저장된 (날짜, 해상도) 조합은 다시 안 돌리고 건너뜀 -> 중간에 끊겨도 이어서 실행 가능
SKIP_EXISTING = True

# query 를 한 번에 몇 개씩 처리할지 (GPU 메모리에 맞춰 조절. feat_unfold가 query_rgb
# 호출마다 매번 다시 계산되므로 너무 작으면 같은 unfold를 불필요하게 반복하게 됨)
BSIZE = 500_000

MAXV = 506.01995849609375  # train_imerg80x100_meta.json 기준 전역 min-max (min=0)

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


def load_hsr_daily(hsr_path, date, lat_min, lat_max, lon_min, lon_max):
    """
    HSR merged nc(KST, lat/lon 오름차순)에서 지정한 KST 날짜의 00~23시를 도메인으로
    crop 후 합산해 일 강수량(오름차순 lat/lon)을 반환.
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
    assert lat_full[0] < lat_full[-1], "HSR lat이 오름차순이 아닙니다. crop/저장 로직을 다시 확인할 것."
    assert lon_full[0] < lon_full[-1], "HSR lon이 오름차순이 아닙니다. crop/저장 로직을 다시 확인할 것."

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

    print(f"  [{date}] HSR: {len(t_idx)}/24 KST hourly slots")
    daily = np.where(cnt > 0, acc, np.nan).astype(np.float32)
    daily = np.nan_to_num(daily, nan=0.0)
    return daily, lat_sub, lon_sub


def load_topo_tensor(topo_path, lat_min, lat_max, lon_min, lon_max, device):
    """make_preds.py 와 동일 recipe (TopoInMemory, lat 오름차순으로 정렬돼 반환됨)."""
    topo_ds = TopoInMemory(path_1=topo_path, lat_range=[lat_min, lat_max], lon_range=[lon_min, lon_max])
    topo = topo_ds[0]
    if isinstance(topo, np.ndarray):
        topo = torch.from_numpy(topo).float()
    else:
        topo = topo.float()
    if topo.ndim == 2:
        topo = topo.unsqueeze(0)
    if topo.ndim == 3:
        topo = topo[:1, ...]
    return topo.to(device)


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
    """목표 물리 해상도(m)를 도메인 중앙위도 기준으로 격자 크기(H,W)와 오름차순
    lat/lon(pixel-center) 로 변환."""
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
    """coord/cell: (1, Q, 2). query_rgb를 bsize 단위로 나눠 호출 후 이어붙임."""
    n = coord.shape[1]
    preds = []
    for ql in tqdm(range(0, n, bsize), desc="    query", leave=False):
        qr = min(ql + bsize, n)
        pred = model.query_rgb(coord[:, ql:qr, :], cell[:, ql:qr, :])
        preds.append(pred)
    return torch.cat(preds, dim=1)


def save_pred_nc(path, precip, lat_asc, lon_asc, date, res_m):
    ds = Dataset(path, "w", format="NETCDF4")
    ds.createDimension("lat", len(lat_asc))
    ds.createDimension("lon", len(lon_asc))
    v_lat = ds.createVariable("lat", "f4", ("lat",))
    v_lon = ds.createVariable("lon", "f4", ("lon",))
    v_p = ds.createVariable("precip", "f4", ("lat", "lon"), zlib=True, complevel=4)
    v_lat[:] = lat_asc
    v_lon[:] = lon_asc
    v_p[:, :] = precip
    v_p.units = "mm"
    ds.title = f"LIIF SR of HSR radar, {date} KST, target {res_m}m"
    ds.source = HSR_PATH
    ds.model_ckpt = MODEL_PATH
    ds.close()


# ============================================================
# 3. 모델/토포 로드 (날짜/해상도와 무관하게 한 번만)
# ============================================================
model = build_and_load_model(MODEL_PATH, device)
topo_tensor = load_topo_tensor(TOPO_PATH, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, device)
if topo_tensor.ndim == 3:
    topo_tensor = topo_tensor.unsqueeze(0)
model.topo = topo_tensor

out_grids = {res_m: out_grid_for_resolution(res_m, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
             for res_m in TARGET_RES_M}
for res_m, (H, W, _, _) in out_grids.items():
    print(f"target {res_m}m -> out grid {H} x {W} = {H * W:,} query points")

# ============================================================
# 4. 전기간(2021 JJA) 추론
# ============================================================
print(f"\n처리 대상: {len(DATES)}일 ({DATES[0]} ~ {DATES[-1]}, 해상도={TARGET_RES_M})")

n_done, n_skip = 0, 0
for date in tqdm(DATES, desc="days"):
    todo_res = []
    for res_m in TARGET_RES_M:
        out_path = os.path.join(SAVE_DIR, f"hsr_liif_sr_{res_m}m_{date}.nc")
        if SKIP_EXISTING and os.path.exists(out_path):
            n_skip += 1
        else:
            todo_res.append((res_m, out_path))
    if not todo_res:
        continue

    daily, hsr_lat, hsr_lon = load_hsr_daily(HSR_PATH, date, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

    x_norm = (daily.astype(np.float32) - 0.0) / MAXV
    inp_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        model.gen_feat(inp_tensor)

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
            pred_mm = pred * MAXV
            if CLAMP_MM_ZERO:
                pred_mm = pred_mm.clamp(min=0.0)
            pred_mm = pred_mm.detach().cpu().numpy()

            del coord, cell, pred
            torch.cuda.empty_cache()

            save_pred_nc(out_path, pred_mm, lat_out, lon_out, date, res_m)
            n_done += 1

    del inp_tensor
    torch.cuda.empty_cache()

print(f"\ndone. saved={n_done}  skipped(existing)={n_skip}")
