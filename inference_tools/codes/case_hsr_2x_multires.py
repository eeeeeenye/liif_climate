"""
HSR 2x LIIF(_climate_edsr_radar_hsr_2x_minmax_liif, 입력 = 1km LR) 의 배율별 출력 케이스: 1km -> 500m / 200m / 100m.
슬라이드의 "High-Resolution Precipitation Reconstruction" 오른쪽 그림(LR | GT | 배율별 SR)의 HSR 버전용 산출물 생성.
  - 학습 배율은 x2(1km->500m) 뿐. 200m(x5), 100m(x10)는 LIIF 연속 표현으로 외삽한 것.
  - 입력 = 학습 때와 같은 cdo remapbil 1km (HSR_val_norm_2x_obscrop.nc), GT = 500m (HSR_val_500m_norm_obscrop.nc). 2021 JJA = val (out-of-sample).
  - ckpt = epoch-last (epoch-best-psnr 는 val psnr=inf(건조 배치) 때문에 epoch 8 로 잡혀 있어 쓰지 않음).
  - 전 도메인(1326x1234)은 500m 만. 200m/100m 는 전 도메인이 4천만 질의/시간이라 너무 느려서, 일강수 최대 지점 주변 창(REG)만 질의.
  - 24시간 합(일누적) + 그날 GT 도메인평균이 가장 큰 1시간(피크 시각)을 저장. 비교용 bilinear 100m(같은 LR) 도 창에서 저장.

좌표 규약 (학습/기존 추론과 동일): LR 과 HR 이 같은 lat/lon 상자를 덮고, 각각 전체 배열을 [-1,1] 로 정규화.
  N배 출력 질의 = 같은 상자를 촘촘히 (H*k, W*k) 한 격자의 부분집합, cell = 2/(H*k), 2/(W*k).

사용:  python case_hsr_2x_multires.py 2021-07-06 [--center LAT LON] [--max-hours N]     (N: 시간 측정용으로 앞 N시간만, 저장 안 함)
출력:  inference_tools/case_studies/multires/case_multires_<날짜>.npz
env: liif, 물리 GPU 2 (CUDA_VISIBLE_DEVICES=2 -> cuda:0)
"""
import os, sys, copy, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from scipy.ndimage import uniform_filter
import models
from utils import make_coord

dev = torch.device("cuda:0")
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid"
LR_NC = f"{M}/HSR_val_norm_2x_obscrop.nc"
HR_NC = f"{M}/HSR_val_500m_norm_obscrop.nc"
CKPT = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth"
OUTDIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
PMAX = 74.8784
BSIZE = 500_000
REG_H, REG_W = 350, 342                                  # 창 크기 (500m 픽셀 수, 짝수) ~ 175km x 200km

argv = sys.argv[1:]
max_hours = int(argv[argv.index("--max-hours") + 1]) if "--max-hours" in argv else 24
center = tuple(float(v) for v in argv[argv.index("--center") + 1:argv.index("--center") + 3]) if "--center" in argv else None
skip = set()
for f in ("--max-hours", "--center"):
    if f in argv:
        i = argv.index(f); skip |= set(range(i, i + (3 if f == "--center" else 2)))
dates = [a for i, a in enumerate(argv) if i not in skip] or ["2021-07-06"]

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
spec = copy.deepcopy(ck["model"])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
print("2x LIIF epoch", ck.get("epoch"))


@torch.no_grad()
def predict(lr_norm, coord, cell, shape, feat=True):
    if feat:
        model.gen_feat(lr_norm)
    out = []
    for s in range(0, coord.shape[1], BSIZE):
        out.append(model.query_rgb(coord[:, s:s + BSIZE], cell[:, s:s + BSIZE]))
    return (torch.cat(out, 1).view(*shape).clamp(0, 1) * PMAX).cpu().numpy()


hr_ds = xr.open_dataset(HR_NC); lr_ds = xr.open_dataset(LR_NC)
lat, lon = hr_ds.lat.values, hr_ds.lon.values
times = pd.to_datetime(hr_ds.time.values)
assert np.array_equal(times, pd.to_datetime(lr_ds.time.values))
H, W = hr_ds.sizes["lat"], hr_ds.sizes["lon"]
hL, wL = lr_ds.sizes["lat"], lr_ds.sizes["lon"]
print(f"HR {H}x{W} (500m) | LR {hL}x{wL} (1km, x{H / hL:.2f} / x{W / wL:.2f})")
dlat, dlon = np.diff(lat).mean(), np.diff(lon).mean()


def norm_axis(n_full, idx):
    """n_full 개로 나눈 [-1,1] 격자의 idx 번째 셀 중심 (make_coord 와 동일 규약)."""
    return -1 + (2 * idx + 1) / n_full


def query_grid(r0, r1, c0, c1, k):
    """500m 픽셀 [r0,r1) x [c0,c1) 창을 k 배 촘촘히 (k = 1, 2.5, 5). 전 격자 (H*k, W*k) 의 부분집합."""
    Hk, Wk = int(round(H * k)), int(round(W * k))
    ri = np.arange(int(round(r0 * k)), int(round(r1 * k))); ci = np.arange(int(round(c0 * k)), int(round(c1 * k)))
    yy, xx = norm_axis(Hk, ri), norm_axis(Wk, ci)
    coord = torch.from_numpy(np.stack(np.meshgrid(yy, xx, indexing="ij"), -1).reshape(-1, 2).astype(np.float32)).unsqueeze(0).to(dev)
    cell = torch.empty_like(coord); cell[:, :, 0] = 2.0 / Hk; cell[:, :, 1] = 2.0 / Wk
    la = lat[0] - dlat / 2 + (ri + 0.5) * (H * dlat / Hk); lo = lon[0] - dlon / 2 + (ci + 0.5) * (W * dlon / Wk)
    return coord, cell, (len(ri), len(ci)), la, lo


def bilinear_at(x, coord, shape):
    """LR(x: 1,1,hL,wL) 을 coord((y,x) in [-1,1]) 에서 bilinear 샘플 -- 전 격자 F.interpolate(align_corners=False) 와 동일."""
    g = coord[..., [1, 0]].view(1, shape[0], shape[1], 2)
    return (F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=False)[0, 0] * PMAX).clamp(min=0).cpu().numpy()


full_c, full_cell, full_shape, _, _ = query_grid(0, H, 0, W, 1.0)

for date in dates:
    npz = f"{OUTDIR}/case_studies/multires/case_multires_{date}.npz"
    if os.path.exists(npz) and max_hours == 24:
        print("이미 있음, 건너뜀:", npz); continue
    tidx = np.where(times.strftime("%Y-%m-%d") == date)[0][:max_hours]
    assert len(tidx) > 0, date
    gt_h = np.stack([np.nan_to_num(hr_ds.precip_norm.isel(time=int(t)).values) * PMAX for t in tidx])
    peak_k = int(gt_h.mean((1, 2)).argmax())
    daily_gt = gt_h.sum(0)
    # 창 중심: 일강수 20km(40px) 평활 최대 지점 (--center 로 덮어쓰기)
    if center is None:
        sm = uniform_filter(daily_gt, 40, mode="constant")
        cr, cc = np.unravel_index(sm.argmax(), sm.shape)
    else:
        cr, cc = np.abs(lat - center[0]).argmin(), np.abs(lon - center[1]).argmin()
    r0 = int(np.clip(cr - REG_H // 2, 0, H - REG_H)) // 2 * 2; c0 = int(np.clip(cc - REG_W // 2, 0, W - REG_W)) // 2 * 2
    r1, c1 = r0 + REG_H, c0 + REG_W
    print(f"{date}: {len(tidx)}시간, 피크 시각 {peak_k:02d} KST (GT 도메인평균 {gt_h[peak_k].mean():.2f} mm/h, max {gt_h[peak_k].max():.1f})")
    print(f"  창(500m px) rows {r0}:{r1} cols {c0}:{c1} -> lat {lat[r0]:.2f}~{lat[r1 - 1]:.2f}, lon {lon[c0]:.2f}~{lon[c1 - 1]:.2f}")

    reg = {"m500": query_grid(r0, r1, c0, c1, 1.0), "m200": query_grid(r0, r1, c0, c1, 2.5), "m100": query_grid(r0, r1, c0, c1, 5.0)}
    for k, g in reg.items():
        print(f"  창 {k}: {g[2][0]}x{g[2][1]} ({g[2][0] * g[2][1] / 1e6:.2f}M 질의/시간)")

    accF = np.zeros((H, W), np.float32)
    acc = {k: np.zeros(g[2], np.float32) for k, g in reg.items()}
    acc["bil100"] = np.zeros(reg["m100"][2], np.float32)
    daily_lr = np.zeros((hL, wL), np.float32)
    valid = np.zeros((H, W), bool); lr_valid = np.zeros((hL, wL), bool)
    peak = {}
    t0 = time.time()
    for k, t in enumerate(tidx):
        hr = hr_ds.precip_norm.isel(time=int(t)).values; lr = lr_ds.precip_norm.isel(time=int(t)).values
        valid |= np.isfinite(hr); lr_valid |= np.isfinite(lr)
        lr0 = np.nan_to_num(lr).astype(np.float32); daily_lr += lr0 * PMAX
        x = torch.from_numpy(lr0)[None, None].to(dev)
        fF = np.clip(predict(x, full_c, full_cell, full_shape), 0, None); accF += fF
        fields = {}
        for name, (coord, cell, shape, _, _) in reg.items():
            fields[name] = np.clip(predict(x, coord, cell, shape, feat=False), 0, None); acc[name] += fields[name]
        fields["bil100"] = bilinear_at(x, reg["m100"][0], reg["m100"][2]); acc["bil100"] += fields["bil100"]
        if k == peak_k:
            peak = dict(fields, full=fF, gt=gt_h[k], lr=lr0 * PMAX)
        torch.cuda.empty_cache()
        print(f"  {k + 1}/{len(tidx)}h  {time.time() - t0:.0f}s", flush=True)

    if max_hours != 24:
        print("시간 측정 모드: 저장 안 함"); continue
    np.savez(npz, ep=ck.get("epoch"), peak_hour=peak_k, region=np.array([r0, r1, c0, c1]), valid=valid, lr_valid=lr_valid,
             lat=lat, lon=lon, lr_lat=lr_ds.lat.values, lr_lon=lr_ds.lon.values,
             lat200=reg["m200"][3], lon200=reg["m200"][4], lat100=reg["m100"][3], lon100=reg["m100"][4],
             d_lr=daily_lr, d_gt=daily_gt, d_full500=accF, d_m500=acc["m500"], d_m200=acc["m200"], d_m100=acc["m100"], d_bil100=acc["bil100"],
             h_lr=peak["lr"], h_gt=peak["gt"], h_full500=peak["full"], h_m500=peak["m500"], h_m200=peak["m200"], h_m100=peak["m100"], h_bil100=peak["bil100"])
    print("saved", npz, f"({time.time() - t0:.0f}s)")
