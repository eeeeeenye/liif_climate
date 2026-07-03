import os
import sys
import copy
import json
from collections import OrderedDict
 
import numpy as np
import pandas as pd
import torch
import xarray as xr
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from omegaconf import OmegaConf
 
sys.path.append("/home/inhye_yoo/deeplearning/inr/liif")
from models.models import make
from datasets.image_folder import TopoInMemory
from utils import make_coord
 
 
# ============================================================
# 1. 기본 설정
# ============================================================

# 사용할 device 설정
# CUDA 사용 가능하면 cuda:2 사용, 아니면 CPU 사용
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# IMERG 강수 데이터 경로
imerg_path = "/home/inhye_yoo/data/IMERG/Imerg_raw/imerg_merged_precip.nc"

# IMERG 격자에 맞춘 지형 데이터 경로
topo_path  = "/home/inhye_yoo/deeplearning/inr/liif/data/ETOPO1_on_IMERG_Land.nc"

# 학습된 모델 checkpoint 경로
model_path = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_imerg_topo_chmeta_final/epoch-best.pth"

# 학습 당시 config yaml 경로
yaml_path  = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_imerg_topo_chmeta_final/config.yaml"

# 예측 결과 저장 디렉토리
save_dir = "./enc_ft/preds_not_merged/preds_daily_precip_IMERG_TOPO_800X800_periodminmax"

# 저장 디렉토리가 없으면 생성
os.makedirs(save_dir, exist_ok=True)

# 추론할 영역 설정: 한반도 주변
lat_min, lat_max = 33, 43
lon_min, lon_max = 124, 134

# 추론 기간 설정
start_date = "2017-01-01"
end_date   = "2019-12-31"

# LIIF 입력 저해상도 크기
H_lr, W_lr = 50, 50

# LIIF 출력 고해상도 크기
H_hr, W_hr = 800, 800

# 예측 결과를 0~1 범위로 제한할지 여부
CLAMP_01 = True

# mm/day 변환 후 음수 강수량을 0으로 제한할지 여부
CLAMP_MM_ZERO = True


# ============================================================
# 2. 유틸 함수
# ============================================================

def make_in_transform(in_mean, in_std):
    """
    입력 이미지를 Tensor로 변환하고,
    학습 시 사용한 mean/std로 정규화하는 transform 생성
    """

    return transforms.Compose([
        # PIL image 또는 numpy image를 torch Tensor로 변환
        # 값 범위는 [0, 1]
        transforms.ToTensor(),

        # 학습 때 사용한 입력 정규화 적용
        transforms.Normalize([in_mean], [in_std])
    ])


def ds_slice(xr_ds, date, lat_min, lat_max, lon_min, lon_max):
    """
    xarray Dataset에서 특정 날짜와 특정 영역의 IMERG 강수장을 추출
    """

    # 강수 변수명 지정
    var = "precipitation"

    # 강수 DataArray 선택
    da = xr_ds[var]

    # 위도 좌표 이름 자동 확인
    # 데이터에 따라 latitude 또는 lat일 수 있음
    lat_name = "latitude" if "latitude" in da.coords else ("lat" if "lat" in da.coords else None)

    # 경도 좌표 이름 자동 확인
    # 데이터에 따라 longitude 또는 lon일 수 있음
    lon_name = "longitude" if "longitude" in da.coords else ("lon" if "lon" in da.coords else None)

    # 위도/경도 좌표가 없으면 에러 발생
    if lat_name is None or lon_name is None:
        raise KeyError(f"lat/lon coord not found. coords={list(da.coords)}")

    # 지정한 위경도 영역만 추출
    sub = da.where(
        (da[lat_name] >= lat_min) & (da[lat_name] <= lat_max) &
        (da[lon_name] >= lon_min) & (da[lon_name] <= lon_max),
        drop=True
    )

    # 입력 date를 pandas Timestamp로 변환
    d = pd.Timestamp(date)

    try:
        # 해당 날짜의 자료 선택
        sub_t = sub.sel(time=d)

    except Exception:
        # 정확히 같은 날짜가 없으면 하루 이내에서 가장 가까운 시간 선택
        sub_t = sub.sel(time=d, method="nearest", tolerance=np.timedelta64(1, "D"))

    # 위도를 북쪽에서 남쪽 방향으로 정렬
    sub_t = sub_t.sortby(lat_name, ascending=False)

    # 마지막 두 차원이 반드시 lat, lon 순서가 되도록 변환
    if tuple(sub_t.dims[-2:]) != (lat_name, lon_name):
        sub_t = sub_t.transpose(..., lat_name, lon_name)

    # NaN은 0으로 바꾸고 float32 배열로 변환
    arr = np.nan_to_num(sub_t.values, nan=0.0).astype(np.float32)

    # 추출된 배열이 비어 있으면 에러
    if arr.size == 0:
        raise ValueError(f"Empty slice at {date}")

    return arr


def compute_period_minmax(imerg_path, start_date, end_date,
                          lat_min, lat_max, lon_min, lon_max):
    """
    추론 기간과 영역에 해당하는 IMERG 강수의 최소/최대값 계산

    이 min/max는 입력을 0~255 이미지 스케일로 변환하거나,
    모델 출력을 다시 mm/day로 복원할 때 사용됨
    """

    print(f"Computing period min/max ({start_date} ~ {end_date}, "
          f"lat {lat_min}-{lat_max}, lon {lon_min}-{lon_max}) ...")

    # IMERG NetCDF 파일 열기
    ds = xr.open_dataset(imerg_path)

    # 강수 변수 선택
    da = ds["precipitation"]

    # 위도 좌표 이름 확인
    lat_name = "latitude" if "latitude" in da.coords else "lat"

    # 경도 좌표 이름 확인
    lon_name = "longitude" if "longitude" in da.coords else "lon"

    # 지정 영역과 기간에 해당하는 데이터만 선택
    sub = da.where(
        (da[lat_name] >= lat_min) & (da[lat_name] <= lat_max) &
        (da[lon_name] >= lon_min) & (da[lon_name] <= lon_max),
        drop=True
    ).sel(time=slice(start_date, end_date))

    # numpy 배열로 변환
    vals = sub.values

    # NaN 제외 최소값 계산
    pmin = float(np.nanmin(vals))

    # NaN 제외 최대값 계산
    pmax = float(np.nanmax(vals))

    print(f"  period min: {pmin:.4f}  max: {pmax:.4f}")

    return pmin, pmax


def lr_to_input_tensor(arr, in_transform, pmin, pmax, out_hw):
    """
    원본 강수 배열을 LIIF 입력 tensor로 변환

    과정:
    1. mm/day 강수값을 period min/max 기준으로 0~255로 변환
    2. uint8 grayscale image로 변환
    3. 50x50으로 resize
    4. Tensor 변환 및 정규화
    """

    # 입력 배열을 float32로 변환하고 NaN은 0으로 처리
    x = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0)

    # period min/max 기준으로 0~255 스케일 변환
    x255 = (x - pmin) / (pmax - pmin + 1e-12) * 255.0

    # 0~255 범위 밖 값 제거
    x255 = np.clip(x255, 0, 255)

    # uint8 이미지로 변환
    img_u8 = np.round(x255).astype(np.uint8)

    # PIL grayscale 이미지로 변환
    pil = Image.fromarray(img_u8).convert("L")

    # LIIF encoder 입력 크기인 50x50으로 resize
    pil = pil.resize(out_hw, resample=Image.BILINEAR)

    # Tensor 변환, 정규화, batch 차원 추가
    return in_transform(pil).unsqueeze(0)


def load_topo_tensor(topo_path, lat_min, lat_max, lon_min, lon_max, device):
    """
    지형 데이터를 불러와 모델 입력에 맞는 Tensor 형태로 변환
    """

    # 지정한 영역의 topo 데이터를 메모리에 로드
    topo_ds = TopoInMemory(
        path_1=topo_path,
        lat_range=[lat_min, lat_max],
        lon_range=[lon_min, lon_max]
    )

    # 첫 번째 topo 데이터 추출
    raw_topo = topo_ds[0]

    # numpy 배열이면 torch Tensor로 변환
    if isinstance(raw_topo, np.ndarray):
        topo = torch.from_numpy(raw_topo).float()

    # 이미 torch Tensor이면 float 타입으로 변환
    elif isinstance(raw_topo, torch.Tensor):
        topo = raw_topo.float()

    # 지원하지 않는 타입이면 에러
    else:
        raise TypeError(f"Unsupported topo type: {type(raw_topo)}")

    # topo가 2차원일 경우 channel 차원 추가
    if topo.ndim == 2:
        topo = topo.unsqueeze(0)

    # topo가 3차원일 경우 첫 번째 채널만 사용
    if topo.ndim == 3:
        topo = topo[:1, ...]

    # batch 차원 추가 후 device로 이동
    return topo.unsqueeze(0).to(device)


def build_and_load_model(ckpt_path, yaml_path, device):
    """
    학습된 LIIF 모델을 생성하고 checkpoint weight를 로드
    """

    # checkpoint 파일 로드
    ckpt = torch.load(ckpt_path, map_location=device)

    # checkpoint 안의 모델 state_dict
    model_sd = ckpt["model"]["sd"]

    # checkpoint 안의 모델 생성 argument
    args = copy.deepcopy(ckpt["model"]["args"])

    # encoder_spec에 freeze 옵션이 있으면 제거
    # 추론 시 현재 모델 생성 함수와 충돌 방지 목적
    if "encoder_spec" in args and "args" in args["encoder_spec"]:
        args["encoder_spec"]["args"].pop("freeze", None)

    # LIIF 모델 생성
    model = make({"name": "liif", "args": args}).to(device)

    # 현재 모델의 state_dict
    new_sd = model.state_dict()

    # shape이 맞는 parameter만 저장할 OrderedDict
    compat = OrderedDict()

    # shape이 맞지 않아 건너뛴 key 저장
    skipped = []

    # 현재 모델 key 기준으로 checkpoint weight와 비교
    for k in new_sd:

        # checkpoint에 같은 key가 있고 shape도 같으면 로드 대상
        if k in model_sd and model_sd[k].shape == new_sd[k].shape:
            compat[k] = model_sd[k]

        # 없거나 shape이 다르면 skip
        else:
            skipped.append(k)

    # 호환되는 weight만 로드
    model.load_state_dict(compat, strict=False)

    # 추론 모드로 전환
    model.eval()

    print(f"Loaded: {len(compat)}  Skipped: {len(skipped)}")

    return model


def make_cell_for_liif_coord(coord, H, W):
    """
    LIIF query_rgb에 필요한 cell tensor 생성

    cell은 각 query 좌표가 대표하는 pixel 영역 크기 정보를 의미함
    """

    # coord와 같은 shape의 tensor 생성
    cell = torch.ones_like(coord)

    # normalized coordinate [-1, 1] 기준 세로 방향 cell size
    cell[:, :, 0] *= 2.0 / H

    # normalized coordinate [-1, 1] 기준 가로 방향 cell size
    cell[:, :, 1] *= 2.0 / W

    return cell


def make_pixel_center_latlon(lat_min, lat_max, lon_min, lon_max, H, W):
    """
    800x800 출력 격자의 pixel center 기준 위도/경도 좌표 생성
    """

    # 위도 해상도
    lat_res = (lat_max - lat_min) / H

    # 경도 해상도
    lon_res = (lon_max - lon_min) / W

    # 각 pixel center의 위도
    # lat_max에서 시작해서 남쪽으로 내려감
    lat = lat_max - (np.arange(H) + 0.5) * lat_res

    # 각 pixel center의 경도
    # lon_min에서 시작해서 동쪽으로 증가
    lon = lon_min + (np.arange(W) + 0.5) * lon_res

    return lat, lon


@torch.no_grad()
def run_liif(model, inp_tensor, coord, cell, topo_tensor,
             gt_mean, gt_std, pmin, pmax, device, H, W,
             clamp_01=True, clamp_mm_zero=True):
    """
    LIIF 모델을 사용해 하나의 날짜에 대한 800x800 고해상도 강수장을 예측
    """

    # 입력 tensor를 device로 이동
    inp_tensor = inp_tensor.to(device, non_blocking=True)

    # query 좌표를 device로 이동
    coord = coord.to(device)

    # cell 정보를 device로 이동
    cell = cell.to(device)

    # topo tensor를 device로 이동
    topo_tensor = topo_tensor.to(device)

    # topo tensor가 2차원이면 batch, channel 차원 추가
    if topo_tensor.ndim == 2:
        topo_tensor = topo_tensor.unsqueeze(0).unsqueeze(0)

    # topo tensor가 3차원이면 channel 차원 추가
    elif topo_tensor.ndim == 3:
        topo_tensor = topo_tensor.unsqueeze(1)

    # 모델 내부에 topo 정보 저장
    model.topo = topo_tensor

    # encoder를 통해 입력 강수장의 feature 생성
    model.gen_feat(inp_tensor)

    # LIIF decoder에 고해상도 좌표를 query하여 예측 수행
    # 출력 shape: (1, H*W, 1)
    pred = model.query_rgb(coord, cell)

    # 예측 결과를 2D grid 형태로 변환
    # shape: (H, W, C)
    pred = pred.view(H, W, -1)

    # 학습 시 gt normalization을 되돌려 0~1 image scale로 복원
    pred01 = pred * gt_std + gt_mean

    # 예측값을 0~1 범위로 제한
    if clamp_01:
        pred01 = pred01.clamp(0.0, 1.0)

    # 0~1 scale을 실제 강수량 mm/day로 복원
    pred_mm = pred01 * (pmax - pmin) + pmin

    # GPU tensor를 CPU numpy 배열로 변환
    pred_mm = pred_mm.detach().cpu().numpy()

    # 마지막 channel 차원 제거
    pred_mm = pred_mm.squeeze()

    # 음수 강수량을 0으로 처리
    if clamp_mm_zero:
        pred_mm = np.maximum(pred_mm, 0.0)

    return pred_mm


# ============================================================
# 3. 기간 min/max 계산
# ============================================================

# 추론 기간과 영역에 해당하는 강수량 최소/최대값 계산
pmin, pmax = compute_period_minmax(
    imerg_path,
    start_date,
    end_date,
    lat_min,
    lat_max,
    lon_min,
    lon_max
)


# ============================================================
# 4. 모델 및 기타 데이터 로드
# ============================================================

# 학습 당시 config 파일 로드
config = OmegaConf.load(yaml_path)

# 입력 데이터 정규화에 사용한 mean
in_mean = float(config.data_norm.inp.sub[0])

# 입력 데이터 정규화에 사용한 std
in_std = float(config.data_norm.inp.div[0])

# GT 데이터 정규화에 사용한 mean
gt_mean = float(config.data_norm.gt.sub[0])

# GT 데이터 정규화에 사용한 std
gt_std = float(config.data_norm.gt.div[0])

# 입력 transform 생성
in_transform = make_in_transform(in_mean, in_std)

# IMERG Dataset 열기
ds_imerg = xr.open_dataset(imerg_path)

# topo tensor 로드
topo_tensor = load_topo_tensor(
    topo_path,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    device
)

# 학습된 LIIF 모델 로드
model = build_and_load_model(model_path, yaml_path, device)

# 정규화 정보 출력
print(f"input norm : mean={in_mean}, std={in_std}")
print(f"gt norm    : mean={gt_mean}, std={gt_std}")
print(f"period min/max : {pmin:.4f} / {pmax:.4f}  (replaces global gmin/gmax)")


# ============================================================
# 5. coord / cell / lat-lon 생성
# ============================================================

# 800x800 고해상도 query 좌표 생성
# shape: (1, H_hr*W_hr, 2)
coord = make_coord(shape=(H_hr, W_hr)).unsqueeze(0)

# 각 query 좌표의 cell size 생성
cell = make_cell_for_liif_coord(coord, H_hr, W_hr)

# 출력 NetCDF에 저장할 고해상도 위도/경도 좌표 생성
lat_hr, lon_hr = make_pixel_center_latlon(
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    H_hr,
    W_hr
)

# 추론할 날짜 목록 생성
dates = pd.date_range(start_date, end_date, freq="D")

# 좌표 tensor shape 확인
print(f"coord: {coord.shape},  cell: {cell.shape}")


# ============================================================
# 6. 전체 기간 예측 저장
# ============================================================

# 날짜별로 반복하면서 예측 수행
for date in tqdm(dates):

    # 날짜를 문자열로 변환
    date_label = date.strftime("%Y-%m-%d")

    # 해당 날짜의 IMERG 저해상도 강수장 추출
    arr = ds_slice(
        ds_imerg,
        date,
        lat_min,
        lat_max,
        lon_min,
        lon_max
    )

    # 강수 배열을 LIIF 입력 tensor로 변환
    t_inp = lr_to_input_tensor(
        arr,
        in_transform,
        pmin,
        pmax,
        out_hw=(W_lr, H_lr)
    )

    # LIIF 모델로 800x800 고해상도 강수장 예측
    pred_mm = run_liif(
        model=model,
        inp_tensor=t_inp,
        coord=coord,
        cell=cell,
        topo_tensor=topo_tensor,
        gt_mean=gt_mean,
        gt_std=gt_std,
        pmin=pmin,
        pmax=pmax,
        device=device,
        H=H_hr,
        W=W_hr,
        clamp_01=CLAMP_01,
        clamp_mm_zero=CLAMP_MM_ZERO,
    )

    # 예측 결과를 xarray DataArray로 변환
    da = xr.DataArray(
        pred_mm[None, :, :],
        dims=("time", "lat", "lon"),
        coords={
            "time": [date],
            "lat": lat_hr,
            "lon": lon_hr
        },
        name="precip"
    )

    # DataArray를 Dataset으로 변환
    ds_out = xr.Dataset({"precip": da})

    # 사용한 min/max 정규화 정보 기록
    ds_out.attrs["norm_info"] = f"period_min={pmin:.4f}, period_max={pmax:.4f}"

    # 날짜별 NetCDF 파일로 저장
    ds_out.to_netcdf(
        os.path.join(save_dir, f"pred_{date_label}.nc")
    )