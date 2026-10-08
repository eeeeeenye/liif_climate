"""
[DEPRECATED] slope/relief 채널 기능이 TopoInMemory에서 제거됨(효과 없음).
옛 3채널 run을 참조하고 relief 채널 마스킹(sr_topo[:, 2:3]) 로직이 있어
현재는 그대로 돌아가지 않는다. 검증 결론(모델이 slope/relief 무시)만 기록으로 남김.

slope/relief 피처가 실제로 효과가 있는지 검증.

검증기간(2020-2022, 1095일) 전체에 대해 도메인 지표 집계:
  A) baseline           : _climate_edsr_topo_80x100_topo_precip_minmax        (elev only)
  B) slope/relief        : _climate_edsr_topo_80x100_topo_slope_relief_ver     (elev+slope+relief)
  C) slope/relief (ablate): 같은 모델인데 추론 시 slope,relief 채널을 0으로  -> 채널 사용 여부 확인

추가로 지형 층화(산악 vs 평지, relief 기준)로 지표를 나눠 계산.
정규화: minmax, mm/day = value * 506.01996

주의: A vs B 는 학습기간/topo 파일이 달라 완전한 ablation 아님.
      B vs C 는 완전히 같은 모델 -> "모델이 slope/relief 를 쓰는가" 만 본다.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import copy
from collections import OrderedDict

import numpy as np
import torch
import yaml
import xarray as xr

sys.path.append("/home/inhye_yoo/deeplearning/inr/liif")
from models.models import make as make_model
from datasets.image_folder import TopoInMemory
from utils import make_coord

SAVE_ROOT = "/home/inhye_yoo/deeplearning/inr/liif/save"
BASE_DIR  = os.path.join(SAVE_ROOT, "_climate_edsr_topo_80x100_topo_precip_minmax")           # 3년(2000-2002), topo=on_IMERG
MATCH_DIR = os.path.join(SAVE_ROOT, "_climate_edsr_topo_80x100_topo_precip_minmax_topo_raw")  # 12년(1998-2009), topo=Ice_g_gmt4, elev only <- B의 정확한 대조군
SR_DIR    = os.path.join(SAVE_ROOT, "_climate_edsr_topo_80x100_topo_slope_relief_ver")        # 12년, topo=Ice_g_gmt4, elev+slope+relief
CKPT      = "epoch-best-loss.pth"

LR_RAW = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_02deg_1998_202504_noleaf.nc"
HR_RAW = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_01deg_1998_202504_noleaf.nc"
DATE0, DATE1 = "2020-01-01", "2022-12-31"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOPO_LAT, TOPO_LON = (33, 43), (124, 132)
VMAX = 506.01995849609375


def load_model(run_dir):
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    da = cfg["val_dataset"]["dataset"]["args"]
    topo = TopoInMemory(path_1=da["topo_nc_path"], lat_range=TOPO_LAT, lon_range=TOPO_LON,
                        **da.get("topo_kwargs", {}))[0].unsqueeze(0).to(DEVICE)
    ck = torch.load(os.path.join(run_dir, CKPT), map_location="cpu")
    m = make_model({"name": "liif", "args": copy.deepcopy(ck["model"]["args"])}).to(DEVICE)
    own = m.state_dict()
    compat = OrderedDict((k, v) for k, v in ck["model"]["sd"].items()
                         if k in own and v.shape == own[k].shape)
    m.load_state_dict(compat, strict=False)
    m.eval()
    return m, topo, int(ck.get("epoch", -1))


# ------------------------------------------------------------------
lr_da = xr.open_dataset(LR_RAW)["precipitation"].sel(time=slice(DATE0, DATE1))
hr_da = xr.open_dataset(HR_RAW)["precipitation"].sel(time=slice(DATE0, DATE1))
T, H, W = hr_da.shape
print(f"val days = {T},  HR grid = {H}x{W}")

lr = np.nan_to_num(lr_da.values).astype(np.float32) / VMAX          # (T,50,40) 정규화 입력
gt = np.nan_to_num(hr_da.values).astype(np.float32) * 1.0           # (T,100,80) mm/day

coord = make_coord((H, W)).unsqueeze(0).to(DEVICE)
cell = torch.ones_like(coord)
cell[..., 0] *= 2.0 / H
cell[..., 1] *= 2.0 / W

base_m, base_topo, base_ep = load_model(BASE_DIR)
match_m, match_topo, match_ep = load_model(MATCH_DIR)
sr_m, sr_topo, sr_ep = load_model(SR_DIR)
print(f"baseline(3yr) epoch={base_ep} topoCh={base_topo.shape[1]} | "
      f"matched(12yr,elev) epoch={match_ep} topoCh={match_topo.shape[1]} | "
      f"slope/relief epoch={sr_ep} topoCh={sr_topo.shape[1]}")

sr_topo_ablate = sr_topo.clone()
sr_topo_ablate[:, 1:] = 0.0                                          # slope, relief 채널 0

# 산악 마스크: slope/relief 모델의 relief 채널을 HR 격자로 리샘플, 상위 30%
relief = torch.nn.functional.interpolate(sr_topo[:, 2:3], size=(H, W), mode="bilinear",
                                         align_corners=False)[0, 0].cpu().numpy()
mtn_mask = relief >= np.quantile(relief, 0.70)
flt_mask = relief <= np.quantile(relief, 0.30)
print(f"산악 픽셀 {mtn_mask.sum()}, 평지 픽셀 {flt_mask.sum()}")


@torch.no_grad()
def predict_all(model, topo):
    model.topo = topo
    out = np.empty((T, H, W), np.float32)
    for t in range(T):
        inp = torch.from_numpy(lr[t])[None, None].to(DEVICE)
        model.gen_feat(inp)
        p = model.query_rgb(coord, cell).view(H, W).clamp_(0.0, None)
        out[t] = p.cpu().numpy()
    return out * VMAX


runs = {
    "A. baseline 3yr (elev, on_IMERG)":   predict_all(base_m, base_topo),
    "A'. matched 12yr (elev, Ice_g)":     predict_all(match_m, match_topo),
    "B. slope/relief 12yr (full topo)":   predict_all(sr_m, sr_topo),
    "C. B와 동일모델, s+r 채널=0":         predict_all(sr_m, sr_topo_ablate),
}


def metrics(pred, mask=None):
    p, g = pred, gt
    if mask is not None:
        p = p[:, mask]; g = g[:, mask]
    err = p - g
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    r = np.corrcoef(p.ravel(), g.ravel())[0, 1]
    return rmse, mae, bias, r


print("\n" + "=" * 78)
print(f"{'run':30s} {'RMSE':>7} {'MAE':>7} {'bias':>7} {'corr':>8}   (mm/day, 전체 도메인)")
print("-" * 78)
for name, pr in runs.items():
    rmse, mae, bias, r = metrics(pr)
    print(f"{name:30s} {rmse:7.3f} {mae:7.3f} {bias:+7.3f} {r:8.5f}")

print("\n" + "=" * 78)
print("지형 층화 RMSE (mm/day)")
print(f"{'run':30s} {'산악(상위30%)':>14} {'평지(하위30%)':>14}")
print("-" * 78)
for name, pr in runs.items():
    rm_m = metrics(pr, mtn_mask)[0]
    rm_f = metrics(pr, flt_mask)[0]
    print(f"{name:30s} {rm_m:14.3f} {rm_f:14.3f}")

print("\n" + "=" * 78)
print("해석 가이드")
print("  B vs C  : 완전히 같은 모델. 차이 없으면 -> 모델이 slope/relief 를 무시.")
print("  B vs A' : 학습기간/topo파일 동일, elev vs elev+s/r 만 다름 -> 이게 진짜 ablation.")
print("  A vs A' : elev only, 3년 vs 12년 -> 데이터 개수 효과.")

daily = {name: np.sqrt(np.mean((pr - gt) ** 2, axis=(1, 2))) for name, pr in runs.items()}
np.savez(os.path.join(SR_DIR, "preds", "daily_rmse_2020_2022.npz"),
         time=hr_da["time"].values,
         A=daily["A. baseline 3yr (elev, on_IMERG)"],
         Ap=daily["A'. matched 12yr (elev, Ice_g)"],
         B=daily["B. slope/relief 12yr (full topo)"],
         C=daily["C. B와 동일모델, s+r 채널=0"])
try:
    from scipy.stats import wilcoxon
    dA = daily["A. baseline 3yr (elev, on_IMERG)"]
    dAp = daily["A'. matched 12yr (elev, Ice_g)"]
    dB = daily["B. slope/relief 12yr (full topo)"]
    dC = daily["C. B와 동일모델, s+r 채널=0"]
    print("\n일별 RMSE paired Wilcoxon (mean ΔRMSE, mm/day):")
    print(f"  B vs A' (slope/relief 효과) : p = {wilcoxon(dB, dAp).pvalue:.2e}   ΔRMSE = {np.mean(dB - dAp):+.4f}")
    print(f"  B vs C  (채널 사용 여부)     : p = {wilcoxon(dB, dC).pvalue:.2e}   ΔRMSE = {np.mean(dB - dC):+.4f}")
    print(f"  A vs A' (데이터 개수 효과)   : p = {wilcoxon(dA, dAp).pvalue:.2e}   ΔRMSE = {np.mean(dAp - dA):+.4f}")
except Exception as e:
    print("scipy 없음, Wilcoxon 생략:", e)
