"""
슬라이드 왼쪽 표 "Seasonal Evaluation for Extreme Rainfall Events (GT > 30 mm/day)" 의 HSR 버전.
  - 방법: Bicubic (LR 1km -> 500m, F.interpolate bicubic) vs LIIF (2x, epoch-last).  둘 다 같은 1km 입력, 500m 출력.
  - 기준: 일누적(KST 달력일, 24h 합) GT(500m raw HSR) > 30 mm/day 인 픽셀-일 만 풀링해서 MAE / RMSE / PCC.
  - 데이터: 2021 JJA (val, out-of-sample) 92일.  HSR 시계열이 JJA 뿐이라 DJF/MAM/SON 은 없음. 월별(6/7/8) 행을 추가로 냄.
  - 속도: 일합 GT > 30 인 픽셀만 시간별로 LIIF 질의 (나머지 픽셀은 평가에 안 쓰임). 하루 = 24h 질의 합산 -> 일합.
    시간별로 각각 clamp(>=0) 후 합 (bicubic 도 동일).
사용: python table_hsr_2x_extreme_seasonal.py [--thr 30]     (일별 캐시: <scratch>/cache_thr30/, 끊겨도 이어서 실행)
출력: 표준출력 표 + inference_tools/station_eval/tables/table_hsr_2x_extreme_thr30.{csv,npz}
env: liif, 물리 GPU 2
"""
import os, sys, copy, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
import models
from utils import make_coord

THR = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 30.0
dev = torch.device("cuda:0")
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid"
CKPT = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth"
OUTDIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
CACHE = f"/tmp/claude-1018/-home-inhye-yoo/360583e4-d141-44c9-9858-78608ac036d0/scratchpad/cache_thr{int(THR)}"
os.makedirs(CACHE, exist_ok=True)
PMAX = 74.8784
BSIZE = 500_000

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
spec = copy.deepcopy(ck["model"])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
print("2x LIIF epoch", ck.get("epoch"))

hr_ds = xr.open_dataset(f"{M}/HSR_val_500m_norm_obscrop.nc"); lr_ds = xr.open_dataset(f"{M}/HSR_val_norm_2x_obscrop.nc")
times = pd.to_datetime(hr_ds.time.values)
assert np.array_equal(times, pd.to_datetime(lr_ds.time.values))
H, W = hr_ds.sizes["lat"], hr_ds.sizes["lon"]
coord_all = make_coord((H, W)).to(dev)                                    # (H*W, 2), row-major (row0 = 남쪽)
days = sorted(set(times.strftime("%Y-%m-%d")))
print(f"{len(days)}일 ({days[0]} ~ {days[-1]}), 임계 GT > {THR} mm/day")

t0 = time.time()
for n, day in enumerate(days):
    cf = f"{CACHE}/{day}.npz"
    if os.path.exists(cf):
        continue
    tidx = np.where(times.strftime("%Y-%m-%d") == day)[0]
    assert len(tidx) == 24, (day, len(tidx))
    gt_h = np.nan_to_num(hr_ds.precip_norm.isel(time=slice(int(tidx[0]), int(tidx[-1]) + 1)).values) * PMAX      # (24,H,W)
    lr_h = np.nan_to_num(lr_ds.precip_norm.isel(time=slice(int(tidx[0]), int(tidx[-1]) + 1)).values).astype(np.float32)
    gt_d = gt_h.sum(0)
    sel = np.flatnonzero((gt_d > THR).ravel())
    if len(sel) == 0:
        np.savez(cf, gt=np.zeros(0, np.float32), liif=np.zeros(0, np.float32), bic=np.zeros(0, np.float32), sel=sel); continue
    sel_t = torch.from_numpy(sel).to(dev)
    coord = coord_all[sel_t].unsqueeze(0)
    cell = torch.ones_like(coord); cell[:, :, 0] *= 2.0 / H; cell[:, :, 1] *= 2.0 / W
    liif = np.zeros(len(sel), np.float32); bic = np.zeros(len(sel), np.float32)
    with torch.no_grad():
        for k in range(24):
            x = torch.from_numpy(lr_h[k])[None, None].to(dev)
            model.gen_feat(x)
            p = torch.cat([model.query_rgb(coord[:, s:s + BSIZE], cell[:, s:s + BSIZE]) for s in range(0, coord.shape[1], BSIZE)], 1)[0, :, 0]
            liif += (p.clamp(0, 1) * PMAX).clamp(min=0).cpu().numpy()
            b = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False)[0, 0].reshape(-1)[sel_t]
            bic += (b * PMAX).clamp(min=0).cpu().numpy()
    np.savez(cf, gt=gt_d.ravel()[sel].astype(np.float32), liif=liif, bic=bic, sel=sel)
    print(f"  {day}: 극한 픽셀 {len(sel):,}  ({n + 1}/{len(days)}, {time.time() - t0:.0f}s)", flush=True)

# 집계
rows = []
parts = {"gt": [], "liif": [], "bic": [], "month": []}
for day in days:
    z = np.load(f"{CACHE}/{day}.npz")
    for k in ("gt", "liif", "bic"):
        parts[k].append(z[k])
    parts["month"].append(np.full(len(z["gt"]), int(day[5:7])))
A = {k: np.concatenate(v) for k, v in parts.items()}


def metrics(p, g):
    e = p - g
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), PCC=np.corrcoef(p, g)[0, 1], bias=e.mean())


groups = [("JJA (2021)", np.ones(len(A["gt"]), bool)), ("Jun", A["month"] == 6), ("Jul", A["month"] == 7), ("Aug", A["month"] == 8)]
for season, m in groups:
    for method, key in [("Bicubic", "bic"), ("LIIF", "liif")]:
        r = metrics(A[key][m], A["gt"][m]); r.update(Season=season, Method=method, n_pixel_days=int(m.sum())); rows.append(r)
df = pd.DataFrame(rows)[["Season", "Method", "MAE", "RMSE", "PCC", "bias", "n_pixel_days"]]
print(f"\nSeasonal Evaluation for Extreme Rainfall Events (GT > {THR:g} mm/day), HSR 2021, 1 km -> 500 m, daily sum (KST)")
print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
df.to_csv(f"{OUTDIR}/station_eval/tables/table_hsr_2x_extreme_thr{int(THR)}.csv", index=False)
np.savez_compressed(f"{OUTDIR}/station_eval/tables/table_hsr_2x_extreme_thr{int(THR)}.npz", **A)
