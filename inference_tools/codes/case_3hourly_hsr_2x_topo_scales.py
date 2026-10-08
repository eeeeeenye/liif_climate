"""
HSR 2x LIIF + GMTED2010 지형(_climate_edsr_radar_hsr_2x_minmax_liif_topo, 학습 입력 1km -> 500m) 의 3시간 누적 케이스 (val 2021 JJA, KST 하루 -> 3시간 8구간).
  열 = Raw HSR(500m) | Input(1km, 2x LR) | 모델 출력 500m (x2, 학습 배율) | 모델 출력 200m (x5, 외삽)
  plot_case_3hourly_8x_scales.py 와 호환 (같은 npz 키). 8x(4km 입력) 그림의 topo 버전 대응용이나, topo 모델은 2x 뿐이라 입력이 1km.
  ASOS 3시간합(원) 은 case_3hourly_hsr_8x_scales.py 와 같은 lag1h obs csv.

지형 처리 (학습과 동일해야 함): TopoInMemory 는 lon 119~134/lat 30~44 (= HSR 전 도메인 2881x2305) 전체의 평균/표준편차로 z-score+tanh.
  추론 상자는 obscrop(1326x1234) 이므로 "전 도메인 통계로 정규화 -> obscrop 상자로 자름" 순서. 지형 격자 = HR 500m 격자와 동일 -> grid_sample(nearest) 이 200m 질의에도 500m 지형을 그대로 씀.

사용:  python case_3hourly_hsr_2x_topo_scales.py 2021-08-23      (CKPT=<ckpt> TAG=<접미사>)
출력:  inference_tools/case_studies/3hourly/case_3h_<날짜>_2x_topo_scales.npz
env: liif, 물리 GPU 2 (CUDA_VISIBLE_DEVICES=2 -> cuda:0)
"""
import os, sys, copy
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm
import models
from utils import make_coord

dev = torch.device("cuda:0")
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge"
LR_NC = f"{M}/regrid/HSR_val_norm_2x_obscrop.nc"        # 1km (HR obscrop 상자와 같은 범위, 2x 파이프라인 입력과 동일 파일)
HR_NC = f"{M}/regrid/HSR_val_500m_norm_obscrop.nc"
OBS = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv"
CKPT = os.environ.get("CKPT", "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif_topo/epoch-last.pth")
TAG = os.environ.get("TAG", "2x_topo_scales")               # 출력 npz 접미사
TOPO_NC = "/home/inhye_yoo/data/level/GMTED2010_korea/GMTED2010_mea075_hsr500m.nc"
OUTDIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
PMAX = 74.8784
BSIZE = 500_000
FINE = 2.5                                              # 500m -> 200m
dates = sys.argv[1:] or ["2021-07-06", "2021-08-23"]

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
spec = copy.deepcopy(ck["model"])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
print("2x LIIF topo epoch", ck.get("epoch"))


@torch.no_grad()
def predict(lr_norm, coord, cell, H, W):
    model.gen_feat(lr_norm)
    out = []
    for s in range(0, coord.shape[1], BSIZE):
        out.append(model.query_rgb(coord[:, s:s + BSIZE], cell[:, s:s + BSIZE]))
    return (torch.cat(out, 1).view(H, W).clamp(0, 1) * PMAX).cpu().numpy()


hr_ds = xr.open_dataset(HR_NC)
lat, lon = hr_ds.lat.values, hr_ds.lon.values
lr_full = xr.open_dataset(LR_NC)
lr_ds = lr_full                                                  # 이미 obscrop (663x617). HR 상자로 다시 자르면 가장자리 1~2픽셀이 빠짐
times = pd.to_datetime(hr_ds.time.values)
assert np.array_equal(times, pd.to_datetime(lr_ds.time.values))
H, W = hr_ds.sizes["lat"], hr_ds.sizes["lon"]
hL, wL = lr_ds.sizes["lat"], lr_ds.sizes["lon"]

# 지형: 학습 TopoInMemory(normalize=standard_tanh, lon 119~134 / lat 30~44) 와 같은 정규화 -> obscrop 상자만 잘라 사용
tds = xr.open_dataset(TOPO_NC).elev.sel(lat=slice(30.0, 44.0), lon=slice(119.0, 134.0)).fillna(0)
tz = tds.values.astype(np.float32)
assert tds.lat.values[0] < tds.lat.values[-1]                                     # 위도 오름차순 (학습 코드도 뒤집지 않음)
telev = np.tanh((tz - tz.mean()) / (tz.std() + 1e-6)).astype(np.float32)
i0 = int(np.abs(tds.lat.values - lat[0]).argmin()); j0 = int(np.abs(tds.lon.values - lon[0]).argmin())
telev = telev[i0:i0 + H, j0:j0 + W]
assert telev.shape == (H, W), (telev.shape, H, W)
assert abs(tds.lat.values[i0] - lat[0]) < 1e-4 and abs(tds.lon.values[j0] - lon[0]) < 1e-4
topo_t = torch.from_numpy(telev)[None, None].to(dev)                              # [1,1,H,W]
model.topo = topo_t
print(f"topo: full {tz.shape} mean {tz.mean():.1f} std {tz.std():.1f} m -> crop {telev.shape}, tanh range {telev.min():.2f}~{telev.max():.2f}")
print(f"HR {H}x{W} (500m) | LR {hL}x{wL} (1km, x{H / hL:.2f} / x{W / wL:.2f})")

# 200m 질의 격자: HR 셀 가장자리 상자를 FINE 배 촘촘히 (중심 좌표)
H2, W2 = int(round(H * FINE)), int(round(W * FINE))
dlat, dlon = np.diff(lat).mean(), np.diff(lon).mean()
lat2 = lat[0] - dlat / 2 + (np.arange(H2) + 0.5) * (H * dlat / H2)
lon2 = lon[0] - dlon / 2 + (np.arange(W2) + 0.5) * (W * dlon / W2)
print(f"200m 격자 {H2}x{W2}, 간격 lat {H * dlat / H2:.5f} lon {W * dlon / W2:.5f} deg")

coord1 = make_coord((H, W)).unsqueeze(0).to(dev)
cell1 = torch.ones_like(coord1); cell1[:, :, 0] *= 2.0 / H; cell1[:, :, 1] *= 2.0 / W
coord2 = make_coord((H2, W2)).unsqueeze(0).to(dev)
cell2 = torch.ones_like(coord2); cell2[:, :, 0] *= 2.0 / H2; cell2[:, :, 1] *= 2.0 / W2

obs = pd.read_csv(OBS, encoding="utf-8-sig"); obs["t"] = pd.to_datetime(obs["time"])


def at(field, flat, flon, slat, slon):
    r = np.abs(flat[None, :] - slat[:, None]).argmin(1); c = np.abs(flon[None, :] - slon[:, None]).argmin(1)
    return field[r, c]


for date in dates:
    npz = f"{OUTDIR}/case_studies/3hourly/case_3h_{date}_{TAG}.npz"
    if os.path.exists(npz):
        print("이미 있음, 건너뜀:", npz); continue
    tidx = np.where(times.strftime("%Y-%m-%d") == date)[0]
    assert len(tidx) == 24, (date, len(tidx))
    raw = np.zeros((8, H, W), np.float32); inp = np.zeros((8, hL, wL), np.float32)
    m500 = np.zeros((8, H, W), np.float32); m200 = np.zeros((8, H2, W2), np.float32); valid = np.zeros((H, W), bool)
    for k, t in enumerate(tqdm(tidx, desc=date)):
        b3 = k // 3
        hr = hr_ds.precip_norm.isel(time=int(t)).values; lr = lr_ds.precip_norm.isel(time=int(t)).values
        valid |= np.isfinite(hr)
        raw[b3] += np.nan_to_num(hr) * PMAX; inp[b3] += np.nan_to_num(lr) * PMAX
        x = torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev)
        m500[b3] += np.clip(predict(x, coord1, cell1, H, W), 0, None)
        m200[b3] += np.clip(predict(x, coord2, cell2, H2, W2), 0, None)
        torch.cuda.empty_cache()

    o = obs[(obs.t.dt.strftime("%Y-%m-%d") == date) & obs.v.notna()].copy()
    o["b3"] = o.t.dt.hour // 3
    st = o.groupby(["x", "y", "b3"]).v.sum().unstack("b3").reindex(columns=range(8))
    st_lon = st.index.get_level_values("x").values; st_lat = st.index.get_level_values("y").values; st_asos = st.values.T
    # 관측소 위치의 값 (전정밀도; 그림은 통계를 여기서 읽음)
    sv = {name: np.stack([at(f[b], fl, fo, st_lat, st_lon) for b in range(8)]) for name, f, fl, fo in
          [("raw", raw, lat, lon), ("inp", inp, lr_ds.lat.values, lr_ds.lon.values),
           ("m500", m500, lat, lon), ("m200", m200, lat2, lon2)]}
    np.savez_compressed(npz, raw=raw, inp=inp, m500=m500, m200=m200.astype(np.float16), valid=valid,
                        lat=lat, lon=lon, lat2=lat2, lon2=lon2, lr_lat=lr_ds.lat.values, lr_lon=lr_ds.lon.values,
                        st_lat=st_lat, st_lon=st_lon, st_asos=st_asos, ep=ck.get("epoch"),
                        **{f"sv_{k}": v for k, v in sv.items()})
    print("saved", npz)
