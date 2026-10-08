"""
_climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h (8x LIIF + MLP adapter ASOS finetune) 의
[2019 JJA = 이 모델의 학습(train) 데이터. station_preds_dump_val2021_8x_ft.py 를 train_dataset 으로 바꾼 복사본]
2019 JJA train (지점, 시각) 별 예측값을 덤프한다. station_preds_dump_val2021_new.py 의 8x/ft 단독 버전.
  - 입력 LR = HSR_val_norm_8x_obscrop.nc (config val_dataset 그대로), 관측소 중심 48x48 LR crop -> 지점 1점 질의.
  - 어댑터는 query_rgb 안에서 이미 적용됨 (correction_layer 를 다시 부르지 않음).
  - lag1h 정렬: ASOS[T] <-> nc[T-1h] (obs csv 가 이미 그렇게 만들어져 있음).
사용:  python station_preds_dump_train2019_8x_ft.py            # CKPT=epoch-best-loss (기본)
       CKPT=epoch-last python station_preds_dump_train2019_8x_ft.py
출력:  inference_tools/station_eval/preds_npz/station_preds_train2019_8x_ft_<CKPT>.npz   (tid, lat, lon, obs, pf, hsr_hr : mm/h)
env: liif, 물리 GPU 2 (CUDA_VISIBLE_DEVICES=2 -> cuda:0)
"""
import sys, os, copy, json
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import torch
import yaml
import xarray as xr
import datasets, models

device = torch.device("cuda:0")
RUN = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h"
CKPT_TAG = os.environ.get("CKPT", "epoch-best-loss")
CFG = "/home/inhye_yoo/deeplearning/inr/liif/configs/train_climate/climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h.yaml"
FT_CKPT = f"{RUN}/{CKPT_TAG}.pth"
META = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/meta_HSR_500m_2019JJA_obscrop_lag1h_train.json"
HR_NC = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/HSR_2019JJA_500m_norm_obscrop.nc"
OUT_NPZ = f"/home/inhye_yoo/deeplearning/inr/liif/inference_tools/station_eval/preds_npz/station_preds_train2019_8x_ft_{CKPT_TAG}.npz"
PMAX = 74.8784
CROP = 48

with open(CFG) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
meta = json.load(open(META))
lat_max, lat_min = meta['lat_range']
lon_min, lon_max = meta['lon_range']

val_ds = datasets.make(config['train_dataset']['dataset'])

sv = torch.load(FT_CKPT, map_location='cuda:0', weights_only=False)
spec = copy.deepcopy(sv['model'])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(device)
model.load_state_dict(spec["sd"], strict=True)
model.eval(); model.topo = None
print(f"{CKPT_TAG} (epoch {sv.get('epoch')})", flush=True)

_g = xr.open_dataset(HR_NC); NC_LAT = _g.lat.values; NC_LON = _g.lon.values
crops, coords, gts, lats, lons, tids, hsr_hr = [], [], [], [], [], [], []
for idx in range(len(val_ds)):
    img_lr, img_hr, obs_, _ = val_ds[idx]
    H_lr, W_lr = img_lr.shape[-2:]
    v = obs_['gt']
    if v.ndim == 1: v = v.unsqueeze(-1)
    mask = v.squeeze(-1) != -2.0
    if mask.sum() == 0: continue
    c = obs_['coord'][mask].clone(); v = v[mask]
    c[:, 0] = -c[:, 0]
    for k in range(c.shape[0]):
        y_, x_ = float(c[k, 0]), float(c[k, 1])
        lats.append(lat_min + (y_ + 1) / 2 * (lat_max - lat_min))
        lons.append(lon_min + (x_ + 1) / 2 * (lon_max - lon_min))
        ia = (y_ + 1) / 2 * (H_lr - 1); ja = (x_ + 1) / 2 * (W_lr - 1)
        x0 = min(max(int(round(ia - CROP / 2)), 0), H_lr - CROP)
        y0 = min(max(int(round(ja - CROP / 2)), 0), W_lr - CROP)
        crops.append(img_lr[:, x0:x0 + CROP, y0:y0 + CROP])
        coords.append(torch.tensor([((ia - x0) / (CROP - 1)) * 2 - 1, ((ja - y0) / (CROP - 1)) * 2 - 1], dtype=torch.float32))
        gts.append(float(v[k, 0]))
        tids.append(idx)
        r_ = int(np.abs(NC_LAT - lats[-1]).argmin()); c_ = int(np.abs(NC_LON - lons[-1]).argmin())
        hsr_hr.append(float(img_hr[0, r_, c_]))         # 원시 HSR 500m 최근접 격자 (정규화값)

N = len(crops)
print("(시각,관측소) 쌍:", N, flush=True)
crops_t = torch.stack(crops); coords_t = torch.stack(coords).unsqueeze(1)

out = []
with torch.no_grad():
    for s in range(0, N, 512):
        inp = crops_t[s:s + 512].to(device); co = coords_t[s:s + 512].to(device)
        cell = torch.ones_like(co); cell[:, :, 0] *= 2 / 1000; cell[:, :, 1] *= 2 / 1000   # wrapper 의 obs cell 과 동일
        out.append(model(inp, None, co, cell).clamp(0, 1).squeeze(-1).squeeze(-1).cpu())
pf = torch.cat(out).numpy() * PMAX
ob = np.array(gts) * PMAX

np.savez(OUT_NPZ, tid=np.array(tids), lat=np.array(lats), lon=np.array(lons), obs=ob, pf=pf, hsr_hr=np.array(hsr_hr) * PMAX)
print('npz 저장:', OUT_NPZ)
print("corr(pred, ASOS) = %.3f, corr(HSR500m, ASOS) = %.3f" % (
    np.corrcoef(pf, ob)[0, 1], np.corrcoef(np.array(hsr_hr) * PMAX, ob)[0, 1]))
