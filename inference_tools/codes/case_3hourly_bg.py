"""
BG(Bernoulli-Gamma) 파인튜닝 모델의 3시간 누적 지도용 npz 생성 (case_3hourly_hsr_lag1h.py 의 BG 버전).

  시간별로 격자 전체(1326x1234)에서 (p, alpha, beta) 를 예측 -> 조건부 평균 mu = p*alpha*beta (mm/h)
  를 3시간씩 합산 (= 3시간 누적 강수의 기대값).  raw/inp/pre/prd 는 기존 case_3h_<date>_lag1h.npz 에 있음.

  adapter 는 LIIF 안에서 local ensemble 전에 방향별로 1회 적용되므로 query_rgb 출력이 이미 raw [.,3] 로짓이다.
  기존 MLP ft 처럼 뒤에 correction_layer 를 또 부르면 안 된다 (3채널이라 adapter 입력 shape 불일치).

사용:  BG_CKPT=... python case_3hourly_bg.py 2021-07-06 2021-08-23
출력:  inference_tools/case_studies/3hourly/case_3h_<날짜>_bg.npz   (bg, bg_p: (8,H,W)  3시간합 mu(mm) / 3시간 평균 p,  ep_bg)
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
import models.bg_adapter          # noqa: F401
from losses_bernoulli_gamma import bg_params
from utils import make_coord

dev = torch.device("cuda:0")
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid"
LR_NC = f"{M}/HSR_val_norm_2x_obscrop.nc"
HR_NC = f"{M}/HSR_val_500m_norm_obscrop.nc"
S = "/home/inhye_yoo/deeplearning/inr/liif/save"
BG_CKPT = os.environ.get("BG_CKPT", f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif_ft_bg_mlp_lag1h/epoch-best-nll.pth")
OUTDIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
BSIZE = 500_000
dates = sys.argv[1:] or ["2021-07-06", "2021-08-23"]

ck = torch.load(BG_CKPT, map_location="cpu", weights_only=False)
spec = copy.deepcopy(ck["model"])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
ep = int(ck.get("epoch"))
print(f"BG ckpt {BG_CKPT}  epoch {ep}")


@torch.no_grad()
def predict(lr_norm, coord, cell, H, W):
    model.gen_feat(lr_norm)
    mu, pp = [], []
    for s in range(0, coord.shape[1], BSIZE):
        raw = model.query_rgb(coord[:, s:s + BSIZE], cell[:, s:s + BSIZE])       # [1,n,3] (adapter 는 내부에서 방향별 1회)
        p, a, b = bg_params(raw)
        mu.append(p * a * b); pp.append(p)
    return torch.cat(mu, 1).view(H, W).cpu().numpy(), torch.cat(pp, 1).view(H, W).cpu().numpy()


lr_ds, hr_ds = xr.open_dataset(LR_NC), xr.open_dataset(HR_NC)
times = pd.to_datetime(hr_ds.time.values)
assert np.array_equal(times, pd.to_datetime(lr_ds.time.values))
H, W = hr_ds.sizes["lat"], hr_ds.sizes["lon"]
coord = make_coord((H, W)).unsqueeze(0).to(dev)
cell = torch.ones_like(coord); cell[:, :, 0] *= 2.0 / H; cell[:, :, 1] *= 2.0 / W

for date in dates:
    tidx = np.where(times.strftime("%Y-%m-%d") == date)[0]
    assert len(tidx) == 24, (date, len(tidx))
    bg = np.zeros((8, H, W), np.float32); bg_p = np.zeros((8, H, W), np.float32)
    for k, t in enumerate(tqdm(tidx, desc=date)):
        lr = lr_ds.precip_norm.isel(time=int(t)).values
        x = torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev)
        mu, p = predict(x, coord, cell, H, W)
        bg[k // 3] += mu; bg_p[k // 3] += p / 3.0
        torch.cuda.empty_cache()
    out = f"{OUTDIR}/case_studies/3hourly/case_3h_{date}_bg.npz"
    np.savez_compressed(out, bg=bg, bg_p=bg_p, ep_bg=ep)
    print("saved", out)
