"""
2021 JJA(검증 기간, 2208시간) 시간강수 90/99th percentile 지도용 집계: GT(raw HSR 500m) vs 8x LIIF(base, epoch-last) vs 8x MLP ft(adapter, epoch-best-loss).
  전 도메인(1326x1234) 매 시각 모델 추론 -> 픽셀별 시계열 90/99th percentile.  GT 결측 시각은 세 필드(gt/base/ft) 모두 제외(np.nanpercentile).
  픽셀이 대부분 시각(>50%) 결측이면(해양 등) valid=False 로 표시, 지도에서 제외.
캐시: 하루(24h)씩 <CACHE>/YYYY-MM-DD.npz 로 저장, 끊겨도 이어서 실행 가능(이미 있는 날은 건너뜀).
사용: python quantile_hsr_8x_mlp_ft_2021jja.py            (--days N 으로 앞 N일만 테스트)
출력: inference_tools/station_eval/preds_npz/quantile_hsr_8x_mlp_ft_2021jja.npz  (q90_gt/q90_base/q90_ft, q99_gt/q99_base/q99_ft, valid, lat, lon, ep_base, ep_ft, n_hours)
소요: 시간당 추론 base+ft 합쳐 약 7~8초 x 2208시간 -> 약 4~5시간 (GPU 2)
env: liif, 물리 GPU 2
"""
import os, sys, copy, time, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import pandas as pd
import torch
import xarray as xr
import models
from utils import make_coord

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=None, help="앞 N일만 처리 (테스트용)")
args = ap.parse_args()

dev = torch.device("cuda:0")
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid"
S = "/home/inhye_yoo/deeplearning/inr/liif/save"
LIIF_CKPT = f"{S}/_climate_edsr_radar_hsr_8x_minmax_liif/epoch-last.pth"
FT_CKPT = f"{S}/_climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h/epoch-best-loss.pth"
OUTDIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
CACHE = f"{OUTDIR}/station_eval/preds_npz/cache_quantile_8x_2021jja"
os.makedirs(CACHE, exist_ok=True)
PMAX = 74.8784
BSIZE = 500_000


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


base, ep_b = load_model(LIIF_CKPT)
ft, ep_f = load_model(FT_CKPT)
print(f"8x LIIF epoch {ep_b}, 8x MLP ft epoch {ep_f}")

hr_ds = xr.open_dataset(f"{M}/HSR_val_500m_norm_obscrop.nc")
lr_ds = xr.open_dataset(f"{M}/HSR_val_norm_8x_obscrop.nc")
lat, lon = hr_ds.lat.values, hr_ds.lon.values
times = pd.to_datetime(hr_ds.time.values)
assert np.array_equal(times, pd.to_datetime(lr_ds.time.values))
H, W = hr_ds.sizes["lat"], hr_ds.sizes["lon"]
coord = make_coord((H, W)).unsqueeze(0).to(dev)
cell = torch.ones_like(coord); cell[:, :, 0] *= 2.0 / H; cell[:, :, 1] *= 2.0 / W

print("전체 시계열 메모리 적재 중...")
HR_ALL = hr_ds.precip_norm.values          # (2208, H, W), NaN = 결측
LR_ALL = lr_ds.precip_norm.values          # (2208, hL, wL)

days = sorted(set(times.strftime("%Y-%m-%d")))
if args.days:
    days = days[:args.days]
print(f"{len(days)}일 ({days[0]} ~ {days[-1]})")

t0 = time.time()
for n, day in enumerate(days):
    cf = f"{CACHE}/{day}.npz"
    if os.path.exists(cf):
        continue
    tidx = np.where(times.strftime("%Y-%m-%d") == day)[0]
    assert len(tidx) == 24, (day, len(tidx))
    gt = np.full((24, H, W), np.nan, np.float32)
    bs = np.zeros((24, H, W), np.float32)
    fp = np.zeros((24, H, W), np.float32)
    for k, t in enumerate(tidx):
        hr = HR_ALL[t]; lr = LR_ALL[t]
        gt[k] = hr.astype(np.float32) * PMAX      # 결측(NaN)은 그대로 유지
        x = torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev)
        bs[k] = predict(base, x, coord, cell, H, W, adapter=False)
        fp[k] = predict(ft, x, coord, cell, H, W, adapter=True)
        torch.cuda.empty_cache()
    np.savez_compressed(cf, gt=gt, base=bs, ft=fp)
    print(f"  {day}  ({n + 1}/{len(days)}, {time.time() - t0:.0f}s)", flush=True)

print("percentile 계산 중...")
GT = np.empty((len(days) * 24, H, W), np.float32)
BASE = np.empty_like(GT); FT = np.empty_like(GT)
for i, day in enumerate(days):
    z = np.load(f"{CACHE}/{day}.npz")
    GT[i * 24:(i + 1) * 24] = z["gt"]; BASE[i * 24:(i + 1) * 24] = z["base"]; FT[i * 24:(i + 1) * 24] = z["ft"]

ok = np.isfinite(GT)
BASE = np.where(ok, BASE, np.nan); FT = np.where(ok, FT, np.nan)
valid = ok.mean(0) > 0.5           # 대부분 시각에 GT 있는 픽셀만 (해양/결측 제외)

out = {}
for tag, arr in [("gt", GT), ("base", BASE), ("ft", FT)]:
    for q in (90, 99):
        m = np.nanpercentile(arr, q, axis=0)
        m[~valid] = np.nan
        out[f"q{q}_{tag}"] = m.astype(np.float32)
out["valid"] = valid; out["lat"] = lat; out["lon"] = lon
out["ep_base"] = ep_b; out["ep_ft"] = ep_f; out["n_hours"] = len(days) * 24
npz_out = f"{OUTDIR}/station_eval/preds_npz/quantile_hsr_8x_mlp_ft_2021jja.npz"
np.savez_compressed(npz_out, **out)
print("saved", npz_out)
