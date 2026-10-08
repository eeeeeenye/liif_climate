"""
HSR 8x LTE: 사전학습(base) vs Weighted-MAE 파인튜닝(wmae) 3시간 누적 케이스 (val 2021 JJA).
입력 = HSR_val_norm_8x_obscrop.nc (4km), GT = 500m. raw/valid/ASOS 는 case_3hourly_hsr_lag1h_new.py 산출 npz 재사용.
adapter 는 query_rgb 안에서 적용되므로 따로 부르지 않음 (이중 적용 주의).
사용:  python case_3hourly_hsr_8x_lte_wmae.py 2021-07-06
출력:  case_studies/3hourly/case_3h_<날짜>_8x_lte_wmae.npz  (raw/inp/pre/prd (8,H,W) 3시간합 mm)
env: liif, GPU 2
"""
import os, sys, copy
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
import numpy as np, pandas as pd, torch, xarray as xr
from tqdm import tqdm
import models
from utils import make_coord

dev = torch.device("cuda:0")
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid"
LR_NC, HR_NC = f"{M}/HSR_val_norm_8x_obscrop.nc", f"{M}/HSR_val_500m_norm_obscrop.nc"
S = "/home/inhye_yoo/deeplearning/inr/liif/save"
BASE = f"{S}/_climate_edsr_radar_hsr_8x_minmax_lte/epoch-last.pth"
FT = os.environ.get("FT_CKPT", f"{S}/_climate_edsr_radar_hsr_8x_minmax_lte_ft_mlp_1e-3_lag1h_2019_2020_wmae/epoch-best-loss.pth")
OUT = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/case_studies/3hourly"
PMAX, BSIZE = 74.8784, 500_000
date = sys.argv[1]


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False); sp = copy.deepcopy(ck["model"])
    m = models.make({"name": sp["name"], "args": sp["args"]}).to(dev)
    m.load_state_dict(sp["sd"], strict=True); m.eval(); m.topo = None
    return m, ck.get("epoch")


@torch.no_grad()
def predict(m, x, coord, cell, H, W):
    m.gen_feat(x)
    o = [m.query_rgb(coord[:, s:s + BSIZE], cell[:, s:s + BSIZE]) for s in range(0, coord.shape[1], BSIZE)]
    return (torch.cat(o, 1).view(H, W).clamp(0, 1) * PMAX).clamp(min=0).cpu().numpy()


base, ep_b = load(BASE); ft, ep_f = load(FT); print("base ep", ep_b, "| wmae ep", ep_f)
lr_ds, hr_ds = xr.open_dataset(LR_NC), xr.open_dataset(HR_NC)
times = pd.to_datetime(hr_ds.time.values); assert np.array_equal(times, pd.to_datetime(lr_ds.time.values))
H, W = hr_ds.sizes["lat"], hr_ds.sizes["lon"]; hL, wL = lr_ds.sizes["lat"], lr_ds.sizes["lon"]
coord = make_coord((H, W)).unsqueeze(0).to(dev)
cell = torch.ones_like(coord); cell[:, :, 0] *= 2.0 / H; cell[:, :, 1] *= 2.0 / W
tidx = np.where(times.strftime("%Y-%m-%d") == date)[0]; assert len(tidx) == 24
pre = np.zeros((8, H, W), np.float32); prd = np.zeros_like(pre); inp = np.zeros((8, hL, wL), np.float32)
for k, t in enumerate(tqdm(tidx)):
    b = k // 3
    lr = np.nan_to_num(lr_ds.precip_norm.isel(time=int(t)).values).astype(np.float32)
    inp[b] += lr * PMAX
    x = torch.from_numpy(lr)[None, None].to(dev)
    pre[b] += predict(base, x, coord, cell, H, W); prd[b] += predict(ft, x, coord, cell, H, W)
    torch.cuda.empty_cache()
src = np.load(f"{OUT}/case_3h_{date}_lag1h_new.npz")
np.savez_compressed(f"{OUT}/case_3h_{date}_8x_lte_wmae.npz", raw=src["raw"], inp=inp, pre=pre, prd=prd, valid=src["valid"],
                    lat=src["lat"], lon=src["lon"], lr_lat=lr_ds.lat.values, lr_lon=lr_ds.lon.values,
                    st_lat=src["st_lat"], st_lon=src["st_lon"], st_asos=src["st_asos"], ep_pre=ep_b, ep_ft=ep_f)
print("saved")
