"""
집중호우 case 3시간 누적 (val 2021 JJA): 사전학습 LTE vs LoRA(r16, ASOS finetune) 예측.
raw/inp/valid/ASOS 는 case_3hourly_hsr_lag1h_new.py 산출 npz(case_3h_<날짜>_lag1h_new.npz) 를 재사용하고,
LTE base / LoRA 체크포인트들의 3시간합 예측만 새로 계산해 case_3h_<날짜>_lte_lora_r16.npz 로 저장.

사용:  python case_3hourly_lte_lora.py <날짜> <name1>=<lora_ckpt1> [<name2>=<lora_ckpt2> ...]
       (체크포인트는 학습 중 덮어쓰기 방지를 위해 복사본을 넘길 것)
env: liif, 물리 GPU 2 (CUDA_VISIBLE_DEVICES=2 -> cuda:0)
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm
import models
from models.lora import build_lora_lte
from utils import make_coord

dev = torch.device("cuda:0")
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid"
LR_NC = f"{M}/HSR_val_norm_2x_obscrop.nc"
HR_NC = f"{M}/HSR_val_500m_norm_obscrop.nc"
OUTDIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
PMAX = 74.8784
BSIZE = 500_000
date = sys.argv[1]
ft_specs = [a.split("=", 1) for a in sys.argv[2:]]


@torch.no_grad()
def predict(model, lr_norm, coord, cell, H, W):
    model.gen_feat(lr_norm)
    out = [model.query_rgb(coord[:, s:s + BSIZE], cell[:, s:s + BSIZE]) for s in range(0, coord.shape[1], BSIZE)]
    p = torch.cat(out, 1).view(H, W).clamp(0, 1) * PMAX
    return p.clamp(min=0).cpu().numpy()


cks = {n: torch.load(p, map_location="cpu", weights_only=False) for n, p in ft_specs}
base_ckpt = next(iter(cks.values()))["base_ckpt"]
sv = torch.load(base_ckpt, map_location="cpu", weights_only=False)
base = models.make(sv["model"], load_sd=True).to(dev).eval(); base.topo = None
mods = {"lte": base}
for n, ck in cks.items():
    m, _ = build_lora_lte(ck["base_ckpt"], ck["rank"], ck["layers"], ck["lora"], device=dev)
    m.eval(); m.topo = None
    mods[n] = m
    print(n, "epoch", ck["epoch"])

lr_ds, hr_ds = xr.open_dataset(LR_NC), xr.open_dataset(HR_NC)
times = pd.to_datetime(hr_ds.time.values)
H, W = hr_ds.sizes["lat"], hr_ds.sizes["lon"]
coord = make_coord((H, W)).unsqueeze(0).to(dev)
cell = torch.ones_like(coord); cell[:, :, 0] *= 2.0 / H; cell[:, :, 1] *= 2.0 / W

tidx = np.where(times.strftime("%Y-%m-%d") == date)[0]
assert len(tidx) == 24, (date, len(tidx))
acc = {n: np.zeros((8, H, W), np.float32) for n in mods}
for k, t in enumerate(tqdm(tidx, desc=date)):
    lr = lr_ds.precip_norm.isel(time=int(t)).values
    x = torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev)
    for n, m in mods.items():
        acc[n][k // 3] += predict(m, x, coord, cell, H, W)
    torch.cuda.empty_cache()

src = np.load(f"{OUTDIR}/case_studies/3hourly/case_3h_{date}_lag1h_new.npz")
keep = {k: src[k] for k in ["raw", "inp", "valid", "lat", "lon", "lr_lat", "lr_lon", "st_lat", "st_lon", "st_asos"]}
out = f"{OUTDIR}/case_studies/3hourly/case_3h_{date}_lte_lora_r16.npz"
np.savez_compressed(out, **keep, **{f"pred_{n}": v for n, v in acc.items()},
                    names=np.array(list(mods)), epochs=np.array([0] + [int(c["epoch"]) for c in cks.values()]))
print("saved", out)
