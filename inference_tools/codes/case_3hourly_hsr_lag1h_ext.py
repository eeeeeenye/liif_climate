"""
case_3hourly_hsr_lag1h_new.py 와 같은 추론(3시간합 raw/inp/pre/prd + ASOS 3시간합)을 val(2021 JJA) 밖 날짜에 대해 수행.
입력 = .bin -> 시간별 -> 정규화/2x 로 새로 만든 날짜별 nc/csv (build_hourly.py, make_norm_lr.py, make_obs.py; DATA 디렉토리).
사용:  python case_3hourly_hsr_lag1h_ext.py 2022-08-08 2023-07-15
출력:  inference_tools/case_studies/3hourly/case_3h_<날짜>_lag1h_ext.npz
env: liif, 물리 GPU 2
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
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid"
DATA = "/tmp/claude-1018/-home-inhye-yoo/fbf3d9d8-ae9f-4c73-a347-49b598bfcb85/scratchpad/norm"
S = "/home/inhye_yoo/deeplearning/inr/liif/save"
BASE_CKPT = f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth"
FT_CKPT = os.environ.get("FT_CKPT", "/tmp/claude-1018/-home-inhye-yoo/80214217-fd09-4b86-ac2a-f405eaea02bd/scratchpad/ft_mse_new_best_loss.pth")
OUTDIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
PMAX = 74.8784
BSIZE = 500_000
dates = sys.argv[1:]


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    spec = copy.deepcopy(ck["model"])
    m = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
    m.load_state_dict(spec["sd"], strict=True); m.eval(); m.topo = None
    return m, ck.get("epoch")


@torch.no_grad()
def predict(model, lr_norm, coord, cell, H, W, adapter):
    model.gen_feat(lr_norm)
    out = []
    for s in range(0, coord.shape[1], BSIZE):
        p = model.query_rgb(coord[:, s:s + BSIZE], cell[:, s:s + BSIZE])
        if adapter:
            p = model.correction_layer(p)
        out.append(p)
    p = torch.cat(out, 1).view(H, W).clamp(0, 1) * PMAX
    return p.clamp(min=0).cpu().numpy()


base, ep_b = load_model(BASE_CKPT)
ft, ep_f = load_model(FT_CKPT)
print(f"pretrain epoch {ep_b}, ft epoch {ep_f}")
for date in dates:
    lr_ds, hr_ds = xr.open_dataset(f"{DATA}/{date}_lr.nc"), xr.open_dataset(f"{DATA}/{date}_hr.nc")
    times = pd.to_datetime(hr_ds.time.values)
    H, W = hr_ds.sizes["lat"], hr_ds.sizes["lon"]
    coord = make_coord((H, W)).unsqueeze(0).to(dev)
    cell = torch.ones_like(coord); cell[:, :, 0] *= 2.0 / H; cell[:, :, 1] *= 2.0 / W
    lat, lon = hr_ds.lat.values, hr_ds.lon.values
    obs = pd.read_csv(f"{DATA}/{date}_obs_lag1h.csv", encoding="utf-8-sig"); obs["t"] = pd.to_datetime(obs["time"])
    npz = f"{OUTDIR}/case_studies/3hourly/case_3h_{date}_lag1h_ext.npz"
    if os.path.exists(npz):
        print("이미 있음, 건너뜀:", npz); continue
    tidx = np.where(times.strftime("%Y-%m-%d") == date)[0]
    assert len(tidx) == 24, (date, len(tidx))
    hL, wL = lr_ds.sizes["lat"], lr_ds.sizes["lon"]
    raw = np.zeros((8, H, W), np.float32); inp = np.zeros((8, hL, wL), np.float32)
    pre = np.zeros((8, H, W), np.float32); prd = np.zeros((8, H, W), np.float32); valid = np.zeros((H, W), bool)
    for k, t in enumerate(tqdm(tidx, desc=date)):
        b3 = k // 3                                        # 0..7  (KST 시 k 가 속한 3시간 구간)
        hr = hr_ds.precip_norm.isel(time=int(t)).values; lr = lr_ds.precip_norm.isel(time=int(t)).values
        valid |= np.isfinite(hr)
        raw[b3] += np.nan_to_num(hr) * PMAX; inp[b3] += np.nan_to_num(lr) * PMAX
        x = torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev)
        pre[b3] += predict(base, x, coord, cell, H, W, adapter=False)
        prd[b3] += predict(ft, x, coord, cell, H, W, adapter=False)   # 재학습 모델: adapter 는 query_rgb 안에서 방향별 1회 적용됨 -> 뒤에 또 부르면 2회
        torch.cuda.empty_cache()
    # ASOS 3시간합 (lag1h csv 라벨 = HSR 시각): 구간 b3 = 라벨 시 3*b3 .. 3*b3+2.  기록이 하나도 없는 지점은 그 구간에서 제외(NaN)
    o = obs[(obs.t.dt.strftime("%Y-%m-%d") == date) & obs.v.notna()].copy()
    o["b3"] = o.t.dt.hour // 3
    st = o.groupby(["x", "y", "b3"]).v.sum().unstack("b3").reindex(columns=range(8))
    st_lon = st.index.get_level_values("x").values; st_lat = st.index.get_level_values("y").values; st_asos = st.values.T   # (8, n_st)
    np.savez_compressed(npz, raw=raw, inp=inp, pre=pre, prd=prd, valid=valid, lat=lat, lon=lon,
             lr_lat=lr_ds.lat.values, lr_lon=lr_ds.lon.values, st_lat=st_lat, st_lon=st_lon,
             st_asos=st_asos, ep_ft=ep_f, ep_pre=ep_b)
    print("saved", npz)
