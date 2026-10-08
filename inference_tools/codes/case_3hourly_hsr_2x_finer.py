"""
HSR 2x LIIF (MLP ft lag1h) 의 출력 배율 케이스: 입력은 1km 고정, 출력 해상도만 x2 / x4 / x8 / x16 = 500m / 250m / 125m / 62.5m.
  배율은 1km 입력 기준. 학습 배율은 x2(1km->500m) 뿐이라 x4/x8/x16 은 LIIF 연속 표현으로 외삽. 500m 보다 세밀한 GT 는 없음(HSR 원자료 500m).
  (코드 변수명 m500/m250/m125/m62 = x2/x4/x8/x16, g* 는 500m HR 격자 대비 k=1/2/4/8 배 촘촘한 질의 격자)
  전 도메인 62.5m 는 시간당 1억 질의라 불가 -> 창(REG_H x REG_W 500m 픽셀)에서만. 창 = 그날 ASOS 지점이 5개 이상 든 곳 중 일강수 평활 최대 (CENTER 로 덮어쓰기: "lat,lon").
  ASOS 3시간합 = lag1h obs csv (라벨 = HSR 시각 -> ASOS[T]<->HSR[T-1h] 반영).
사용: python case_3hourly_hsr_2x_finer.py 2021-08-23
      전 도메인(한국 전체): REG_H=1326 REG_W=1234 TAG=_full NOCOMPRESS=1 python ...   (x16 은 시간당 ~1e8 질의 -> 무거움, m62 는 float16 저장)
출력: inference_tools/case_studies/3hourly/case_3h_<날짜>_2x_finer.npz
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
CKPT = os.environ.get("CKPT", "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h/epoch-best-loss.pth")   # MLP ft lag1h (ep 393). 어댑터는 query_rgb 안에서 자동 적용됨
TAG = os.environ.get("TAG", "")
OUTDIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
PMAX = 74.8784
BSIZE = int(os.environ.get("BSIZE", "500000"))
REG_H, REG_W = int(os.environ.get("REG_H", 200)), int(os.environ.get("REG_W", 200))      # 500m 픽셀 (짝수)
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
    """500m 픽셀 [r0,r1) x [c0,c1) 창을 k 배 촘촘히 (k=1, 2.5, 5). 전 격자 (H*k, W*k) 의 부분집합, [-1,1] 규약은 학습과 동일."""
    Hk, Wk = int(round(H * k)), int(round(W * k))
    ri = np.arange(int(round(r0 * k)), int(round(r1 * k))); ci = np.arange(int(round(c0 * k)), int(round(c1 * k)))
    yy, xx = -1 + (2 * ri + 1) / Hk, -1 + (2 * ci + 1) / Wk
    coord = torch.from_numpy(np.stack(np.meshgrid(yy, xx, indexing="ij"), -1).reshape(-1, 2).astype(np.float32)).unsqueeze(0).to(dev)
    cell = torch.empty_like(coord); cell[:, :, 0] = 2.0 / Hk; cell[:, :, 1] = 2.0 / Wk
    la = lat[0] - dlat / 2 + (ri + 0.5) * (H * dlat / Hk); lo = lon[0] - dlon / 2 + (ci + 0.5) * (W * dlon / Wk)
    return coord, cell, (len(ri), len(ci)), la, lo


for date in dates:
    npz = f"{OUTDIR}/case_studies/3hourly/case_3h_{date}_2x_finer{TAG}.npz"
    if os.path.exists(npz):
        print("이미 있음, 건너뜀:", npz); continue
    tidx = np.where(times.strftime("%Y-%m-%d") == date)[0]
    assert len(tidx) == 24, (date, len(tidx))
    gt_d = np.stack([np.nan_to_num(hr_ds.precip_norm.isel(time=int(t)).values) * PMAX for t in tidx]).sum(0)
    o = obs[(obs.t.dt.strftime("%Y-%m-%d") == date) & obs.v.notna()]
    sxy = o.groupby(["x", "y"]).size().reset_index()[["x", "y"]].values                       # 그날 기록 있는 지점 (lon, lat)
    sr = np.abs(lat[None, :] - sxy[:, 1:2]).argmin(1); sc = np.abs(lon[None, :] - sxy[:, 0:1]).argmin(1)
    if os.environ.get("CENTER"):
        clat, clon = (float(v) for v in os.environ["CENTER"].split(","))
        cr, cc = np.abs(lat - clat).argmin(), np.abs(lon - clon).argmin()
    else:
        sm = uniform_filter(gt_d, 90, mode="constant")                                        # ~45km 평활
        best, cr, cc = -1, None, None
        for r_, c_ in zip(sr, sc):
            n_in = int(((np.abs(sr - r_) < REG_H // 2) & (np.abs(sc - c_) < REG_W // 2)).sum())
            if n_in >= 5 and sm[r_, c_] > best:
                best, cr, cc = sm[r_, c_], r_, c_
    r0 = int(np.clip(cr - REG_H // 2, 0, H - REG_H)) // 2 * 2; c0 = int(np.clip(cc - REG_W // 2, 0, W - REG_W)) // 2 * 2
    r1, c1 = r0 + REG_H, c0 + REG_W
    print(f"{date}: 창 rows {r0}:{r1} cols {c0}:{c1} -> lat {lat[r0]:.2f}~{lat[r1 - 1]:.2f}, lon {lon[c0]:.2f}~{lon[c1 - 1]:.2f}")
    g500 = query_grid(r0, r1, c0, c1, 1.0); g250 = query_grid(r0, r1, c0, c1, 2.0); g125 = query_grid(r0, r1, c0, c1, 4.0); g62 = query_grid(r0, r1, c0, c1, 8.0)

    raw = np.zeros((8, REG_H, REG_W), np.float32); inp = np.zeros((8, REG_H // 2, REG_W // 2), np.float32); m500 = np.zeros((8,) + g500[2], np.float32)
    m250 = np.zeros((8,) + g250[2], np.float32); m125 = np.zeros((8,) + g125[2], np.float32); m62 = np.zeros((8,) + g62[2], np.float32)
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
        m500[b] += predict(g500[0], g500[1], g500[2])
        m250[b] += predict(g250[0], g250[1], g250[2]); m125[b] += predict(g125[0], g125[1], g125[2]); m62[b] += predict(g62[0], g62[1], g62[2])
        torch.cuda.empty_cache()
        print(f"  {k + 1}/24h  {time.time() - t0:.0f}s", flush=True)

    o = obs[(obs.t.dt.strftime("%Y-%m-%d") == date) & obs.v.notna()].copy()
    o["b3"] = o.t.dt.hour // 3
    st = o.groupby(["x", "y", "b3"]).v.sum().unstack("b3").reindex(columns=range(8))
    (np.savez if os.environ.get("NOCOMPRESS") else np.savez_compressed)(npz, raw=raw, inp=inp, m500=m500, m250=m250, m125=m125, m62=m62.astype(np.float16), valid=valid, region=np.array([r0, r1, c0, c1]),
                        lat=lat[r0:r1], lon=lon[c0:c1], lr_lat=lr_ds.lat.values[r0 // 2:r1 // 2], lr_lon=lr_ds.lon.values[c0 // 2:c1 // 2],
                        lat250=g250[3], lon250=g250[4], lat125=g125[3], lon125=g125[4], lat62=g62[3], lon62=g62[4],
                        st_lat=st.index.get_level_values("y").values, st_lon=st.index.get_level_values("x").values, st_asos=st.values.T,
                        ep=ck.get("epoch"))
    print("saved", npz, f"({time.time() - t0:.0f}s)")
