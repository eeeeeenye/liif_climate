"""
8x LTE(_climate_edsr_radar_hsr_8x_minmax_lte, 학습 입력 4km -> 여기선 1km 입력 사용, 4x LTE 에서 resume 해 학습 중, best epoch = epoch-best-loss) 의 3시간 누적 케이스.
case_3hourly_hsr_lag1h_ext.py 와 같은 구조/출력 키(raw/inp/pre/prd/...), 단
  inp = 1km (2x 파이프라인의 <날짜>_lr.nc, ext 2x 그림과 동일 입력),  pre = 8x LIIF(epoch-last, 비교 기준),  prd = 8x LTE(best).
ASOS 3시간합 = lag1h csv (라벨 = HSR 시각).
사용: python case_3hourly_hsr_8x_lte_ext.py 2023-07-15        (LTE_CKPT 환경변수로 체크포인트 지정, 학습 중 덮어쓰기 방지용 복사본 권장)
출력: inference_tools/case_studies/3hourly/case_3h_<날짜>_8x_lte_1km_ext.npz
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
NORM = "/tmp/claude-1018/-home-inhye-yoo/fbf3d9d8-ae9f-4c73-a347-49b598bfcb85/scratchpad/norm"                                   # <날짜>_hr.nc (500m obscrop), <날짜>_obs_lag1h.csv
S = "/home/inhye_yoo/deeplearning/inr/liif/save"
LIIF_CKPT = f"{S}/_climate_edsr_radar_hsr_8x_minmax_liif/epoch-last.pth"
LTE_CKPT = os.environ.get("LTE_CKPT", f"{S}/_climate_edsr_radar_hsr_8x_minmax_lte/epoch-best-loss.pth")
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
def predict(model, lr_norm, coord, cell, H, W):
    model.gen_feat(lr_norm)
    out = [model.query_rgb(coord[:, s:s + BSIZE], cell[:, s:s + BSIZE]) for s in range(0, coord.shape[1], BSIZE)]
    return (torch.cat(out, 1).view(H, W).clamp(0, 1) * PMAX).clamp(min=0).cpu().numpy()


base, ep_b = load_model(LIIF_CKPT)
lte, ep_l = load_model(LTE_CKPT)
print(f"8x LIIF epoch {ep_b}, 8x LTE epoch {ep_l} ({LTE_CKPT})")
for date in dates:
    npz = f"{OUTDIR}/case_studies/3hourly/case_3h_{date}_8x_lte_1km_ext.npz"
    if os.path.exists(npz):
        print("이미 있음, 건너뜀:", npz); continue
    lr_ds, hr_ds = xr.open_dataset(f"{NORM}/{date}_lr.nc"), xr.open_dataset(f"{NORM}/{date}_hr.nc")
    times = pd.to_datetime(hr_ds.time.values)
    H, W = hr_ds.sizes["lat"], hr_ds.sizes["lon"]; hL, wL = lr_ds.sizes["lat"], lr_ds.sizes["lon"]
    assert (hL, wL) == (H // 2, W // 2), (hL, wL)     # 1km = 500m 의 정확히 1/2
    coord = make_coord((H, W)).unsqueeze(0).to(dev)
    cell = torch.ones_like(coord); cell[:, :, 0] *= 2.0 / H; cell[:, :, 1] *= 2.0 / W
    obs = pd.read_csv(f"{NORM}/{date}_obs_lag1h.csv", encoding="utf-8-sig"); obs["t"] = pd.to_datetime(obs["time"])
    tidx = np.where(times.strftime("%Y-%m-%d") == date)[0]
    assert len(tidx) == 24, (date, len(tidx))
    raw = np.zeros((8, H, W), np.float32); inp = np.zeros((8, hL, wL), np.float32)
    pre = np.zeros((8, H, W), np.float32); prd = np.zeros((8, H, W), np.float32); valid = np.zeros((H, W), bool)
    for k, t in enumerate(tqdm(tidx, desc=date)):
        b3 = k // 3
        hr = hr_ds.precip_norm.isel(time=int(t)).values; lr = lr_ds.precip_norm.isel(time=int(t)).values
        valid |= np.isfinite(hr)
        raw[b3] += np.nan_to_num(hr) * PMAX; inp[b3] += np.nan_to_num(lr) * PMAX
        x = torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev)
        pre[b3] += predict(base, x, coord, cell, H, W); prd[b3] += predict(lte, x, coord, cell, H, W)
        torch.cuda.empty_cache()
    o = obs[(obs.t.dt.strftime("%Y-%m-%d") == date) & obs.v.notna()].copy()
    o["b3"] = o.t.dt.hour // 3
    st = o.groupby(["x", "y", "b3"]).v.sum().unstack("b3").reindex(columns=range(8))
    np.savez_compressed(npz, raw=raw, inp=inp, pre=pre, prd=prd, valid=valid, lat=hr_ds.lat.values, lon=hr_ds.lon.values,
                        lr_lat=lr_ds.lat.values, lr_lon=lr_ds.lon.values, st_lat=st.index.get_level_values("y").values,
                        st_lon=st.index.get_level_values("x").values, st_asos=st.values.T, ep_ft=ep_l, ep_pre=ep_b)
    print("saved", npz)
