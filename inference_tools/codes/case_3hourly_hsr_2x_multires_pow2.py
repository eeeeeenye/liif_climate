"""
case_3hourly_hsr_2x_multires.py 와 같은 모델/입력/창(호우 창, case_multires_<날짜>.npz 재사용), 단 배율을 200m/100m(x5/x10) 대신
1km LR 기준 x2/x4/x8/x16 (= 500/250/125/62.5m, 전부 500m 격자의 정수배라 valid mask 도 kron 하나로 처리) 로 확장.
  Raw HSR 500m / Input 1km / LIIF x2(500m, 학습 배율) / x4(250m) / x8(125m) / x16(62.5m, 전부 외삽 배율은 x4 이상).
ASOS 3시간합 = case_3hourly_hsr_2x_multires.py 와 같은 방식 (lag1h obs csv).
사용: python case_3hourly_hsr_2x_multires_pow2.py 2021-07-06        (CKPT=... 로 pretrain/ft 체크포인트 지정, TAG=... 로 출력 접미사)
출력: inference_tools/case_studies/3hourly/case_3h_<날짜>_2x_multires_pow2<TAG>.npz
env: liif, 물리 GPU 2
"""
import os, sys, copy, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.ndimage import uniform_filter
import models
from utils import make_coord

dev = torch.device("cuda:0")
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid"
LR_NC = f"{M}/HSR_val_norm_2x_obscrop.nc"
HR_NC = f"{M}/HSR_val_500m_norm_obscrop.nc"
OBS = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv"
CKPT = os.environ.get("CKPT", "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth")   # 기본 = 사전학습 2x. ft 모델은 어댑터가 query_rgb 안에서 자동 적용됨
TAG = os.environ.get("TAG", "")                                                                                                  # 출력 접미사 (예: _ftlag1h)
OUTDIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
PMAX = 74.8784
BSIZE = int(os.environ.get("BSIZE", "500000"))
REG_H, REG_W = 350, 342
# (열 이름, HR(500m) 격자 기준 배율 k, 1km LR 기준 배율 태그)  k 는 전부 정수 -> valid mask 는 np.kron 으로 그대로 업샘플
SCALES = [("500", 1, "x2"), ("250", 2, "x4"), ("125", 4, "x8"), ("62.5", 8, "x16")]
dates = sys.argv[1:] or ["2021-07-06"]

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
spec = copy.deepcopy(ck["model"])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
print("2x LIIF epoch", ck.get("epoch"))

hr_ds = xr.open_dataset(HR_NC); lr_ds = xr.open_dataset(LR_NC)
lat, lon = hr_ds.lat.values, hr_ds.lon.values
times = pd.to_datetime(hr_ds.time.values)
assert np.array_equal(times, pd.to_datetime(lr_ds.time.values))
H, W = hr_ds.sizes["lat"], hr_ds.sizes["lon"]
hL, wL = lr_ds.sizes["lat"], lr_ds.sizes["lon"]
dlat, dlon = np.diff(lat).mean(), np.diff(lon).mean()
obs = pd.read_csv(OBS, encoding="utf-8-sig"); obs["t"] = pd.to_datetime(obs["time"])


@torch.no_grad()
def predict(coord, cell, shape):
    out = [model.query_rgb(coord[:, s:s + BSIZE], cell[:, s:s + BSIZE]) for s in range(0, coord.shape[1], BSIZE)]
    return (torch.cat(out, 1).view(*shape).clamp(0, 1) * PMAX).clamp(min=0).cpu().numpy()


def query_grid(r0, r1, c0, c1, k):
    """500m 픽셀 [r0,r1) x [c0,c1) 창을 k 배 촘촘히 (정수). 전 격자 (H*k, W*k) 의 부분집합, [-1,1] 규약은 학습과 동일."""
    Hk, Wk = H * k, W * k
    ri = np.arange(r0 * k, r1 * k); ci = np.arange(c0 * k, c1 * k)
    yy, xx = -1 + (2 * ri + 1) / Hk, -1 + (2 * ci + 1) / Wk
    coord = torch.from_numpy(np.stack(np.meshgrid(yy, xx, indexing="ij"), -1).reshape(-1, 2).astype(np.float32)).unsqueeze(0).to(dev)
    cell = torch.empty_like(coord); cell[:, :, 0] = 2.0 / Hk; cell[:, :, 1] = 2.0 / Wk
    la = lat[0] - dlat / 2 + (ri + 0.5) * (H * dlat / Hk); lo = lon[0] - dlon / 2 + (ci + 0.5) * (W * dlon / Wk)
    return coord, cell, (len(ri), len(ci)), la, lo


for date in dates:
    npz = f"{OUTDIR}/case_studies/3hourly/case_3h_{date}_2x_multires_pow2{TAG}.npz"
    if os.path.exists(npz):
        print("이미 있음, 건너뜀:", npz); continue
    tidx = np.where(times.strftime("%Y-%m-%d") == date)[0]
    assert len(tidx) == 24, (date, len(tidx))
    prev = f"{OUTDIR}/case_studies/multires/case_multires_{date}.npz"
    if os.path.exists(prev):
        r0, r1, c0, c1 = [int(v) for v in np.load(prev)["region"]]
    else:
        gt_d = np.stack([np.nan_to_num(hr_ds.precip_norm.isel(time=int(t)).values) * PMAX for t in tidx]).sum(0)
        cr, cc = np.unravel_index(uniform_filter(gt_d, 40, mode="constant").argmax(), gt_d.shape)
        r0 = int(np.clip(cr - REG_H // 2, 0, H - REG_H)) // 2 * 2; c0 = int(np.clip(cc - REG_W // 2, 0, W - REG_W)) // 2 * 2
        r1, c1 = r0 + REG_H, c0 + REG_W
    print(f"{date}: 창 rows {r0}:{r1} cols {c0}:{c1} -> lat {lat[r0]:.2f}~{lat[r1 - 1]:.2f}, lon {lon[c0]:.2f}~{lon[c1 - 1]:.2f}")
    grids = {name: query_grid(r0, r1, c0, c1, k) for name, k, _ in SCALES}

    raw = np.zeros((8, REG_H, REG_W), np.float32); inp = np.zeros((8, REG_H // 2, REG_W // 2), np.float32)
    preds = {name: np.zeros((8,) + grids[name][2], np.float32) for name, _, _ in SCALES}
    valid = np.zeros((REG_H, REG_W), bool); t0 = time.time()
    for k, t in enumerate(tidx):
        b = k // 3
        hr = hr_ds.precip_norm.isel(time=int(t)).values; lr = lr_ds.precip_norm.isel(time=int(t)).values
        hr = hr[r0:r1, c0:c1]; lr_w = lr[r0 // 2:r1 // 2, c0 // 2:c1 // 2]
        valid |= np.isfinite(hr)
        raw[b] += np.nan_to_num(hr) * PMAX; inp[b] += np.nan_to_num(lr_w) * PMAX
        x = torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev)      # 인코더 입력은 전체 LR
        with torch.no_grad():
            model.gen_feat(x)
        for name, _, _ in SCALES:
            preds[name][b] += predict(grids[name][0], grids[name][1], grids[name][2])
        torch.cuda.empty_cache()
        print(f"  {k + 1}/24h  {time.time() - t0:.0f}s", flush=True)

    o = obs[(obs.t.dt.strftime("%Y-%m-%d") == date) & obs.v.notna()].copy()
    o["b3"] = o.t.dt.hour // 3
    st = o.groupby(["x", "y", "b3"]).v.sum().unstack("b3").reindex(columns=range(8))
    save = dict(raw=raw, inp=inp, valid=valid, region=np.array([r0, r1, c0, c1]),
                lat=lat[r0:r1], lon=lon[c0:c1], lr_lat=lr_ds.lat.values[r0 // 2:r1 // 2], lr_lon=lr_ds.lon.values[c0 // 2:c1 // 2],
                st_lat=st.index.get_level_values("y").values, st_lon=st.index.get_level_values("x").values, st_asos=st.values.T,
                ep=ck.get("epoch"))
    for name, _, _ in SCALES:
        save[f"m{name}"] = preds[name]; save[f"lat{name}"] = grids[name][3]; save[f"lon{name}"] = grids[name][4]
    np.savez_compressed(npz, **save)
    print("saved", npz, f"({time.time() - t0:.0f}s)")
