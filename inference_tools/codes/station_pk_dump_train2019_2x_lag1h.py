"""
_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h (2x LIIF + MLP adapter, ASOS finetune) 의
2019 JJA train (지점, 시각) 별로 [어댑터 입력 p_k (방향별, mm/h)] 와 [lag1h 정렬된 ASOS] 를 덤프한다.
  - 목적: adapter transfer function 그림(파란 곡선)의 진짜 기준 E[ASOS | p_k] 를 정렬된 짝으로 계산.
    (기존 adapter_pk_cache_hsr.npz 는 lag1h csv 생성(09-19 17:04) 이전(13:28)에 만들어져 옛 lag0 짝일 가능성이 높아 폐기)
  - p_k 는 model.add_layer 의 forward hook 으로 잡는다 (le_preds 는 어댑터 통과 후 값이라 쓰면 안 됨).
    현재 liif_baseline.py 는 query_rgb 루프 안에서 방향별로 어댑터를 적용하므로 4번 호출 -> P: [N,4].
    (코드가 blend 후 1회 적용으로 바뀌어 있으면 1번 호출 -> P: [N,1]; 아래에서 자동 처리)
  - 입력 LR = HSR_2019JJA_norm_2x_obscrop.nc (config train_dataset 그대로), 관측소 중심 48x48 LR crop -> 지점 1점 질의.
  - ASOS 짝: obs csv(lag1h_train) 그대로. 끝에서 corr(HSR 500m 격자, ASOS) 로 정렬 검증 (기대값 ~0.70).
사용:  python station_pk_dump_train2019_2x_lag1h.py            # CKPT=epoch-best-loss (기본)
       CKPT=epoch-last python station_pk_dump_train2019_2x_lag1h.py
출력:  inference_tools/station_eval/preds_npz/station_pk_train2019_2x_lag1h_<CKPT>.npz
       (tid, lat, lon, obs, pf, hsr_hr : mm/h,  P: [N,K] 어댑터 입력 mm/h,  W: [N,K] 방향별 blend 가중치)
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
RUN = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h"
CKPT_TAG = os.environ.get("CKPT", "epoch-best-loss")
CFG = "/home/inhye_yoo/deeplearning/inr/liif/configs/train_climate/climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h.yaml"
FT_CKPT = f"{RUN}/{CKPT_TAG}.pth"
META = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/meta_HSR_500m_2019JJA_obscrop_lag1h_train.json"
HR_NC = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/HSR_2019JJA_500m_norm_obscrop.nc"
OUT_NPZ = f"/home/inhye_yoo/deeplearning/inr/liif/inference_tools/station_eval/preds_npz/station_pk_train2019_2x_lag1h_{CKPT_TAG}.npz"
PMAX = 74.8784
CROP = 48

with open(CFG) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
meta = json.load(open(META))
lat_max, lat_min = meta['lat_range']
lon_min, lon_max = meta['lon_range']

train_ds = datasets.make(config['train_dataset']['dataset'])

sv = torch.load(FT_CKPT, map_location='cuda:0', weights_only=False)
spec = copy.deepcopy(sv['model'])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(device)
model.load_state_dict(spec["sd"], strict=True)
model.eval(); model.topo = None
print(f"{CKPT_TAG} (epoch {sv.get('epoch')})", flush=True)

# 어댑터 입력(= 방향별 imnet 출력 p_k) 캡처: 한 forward 안에서 add_layer 가 호출된 횟수만큼 쌓임
_cap = []
model.add_layer.register_forward_hook(lambda mod, inp, out: _cap.append(inp[0].detach().cpu()))

_g = xr.open_dataset(HR_NC); NC_LAT = _g.lat.values; NC_LON = _g.lon.values
crops, coords, gts, lats, lons, tids, hsr_hr = [], [], [], [], [], [], []
for idx in range(len(train_ds)):
    img_lr, img_hr, obs_, _ = train_ds[idx]
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

out, Pk, Wk = [], [], []
with torch.no_grad():
    for s in range(0, N, 512):
        inp = crops_t[s:s + 512].to(device); co = coords_t[s:s + 512].to(device)
        cell = torch.ones_like(co); cell[:, :, 0] *= 2 / 1000; cell[:, :, 1] *= 2 / 1000   # wrapper 의 obs cell 과 동일
        _cap.clear()
        out.append(model(inp, None, co, cell).clamp(0, 1).squeeze(-1).squeeze(-1).cpu())
        Pk.append(torch.stack([p.view(p.shape[0], -1)[:, 0] for p in _cap], 1))        # [B,K]
        Wk.append(torch.stack([w.view(w.shape[0], -1)[:, 0].cpu() for w in getattr(model, 'le_weights', [])], 1)
                  if len(getattr(model, 'le_weights', [])) == len(_cap) else torch.full((len(_cap[0]), len(_cap)), np.nan))
pf = torch.cat(out).numpy() * PMAX
P = torch.cat(Pk).numpy() * PMAX
W = torch.cat(Wk).numpy()
ob = np.array(gts) * PMAX
hs = np.array(hsr_hr) * PMAX

np.savez(OUT_NPZ, tid=np.array(tids), lat=np.array(lats), lon=np.array(lons), obs=ob, pf=pf, hsr_hr=hs, P=P, W=W)
print('npz 저장:', OUT_NPZ, " P shape:", P.shape)

# ---- 검증/요약 ----
print("corr(HSR500m, ASOS) = %.3f   (lag1h 정렬 기대값 ~0.70, lag0 이면 ~0.46)" % np.corrcoef(hs, ob)[0, 1])
print("corr(pred,    ASOS) = %.3f" % np.corrcoef(pf, ob)[0, 1])
pb = P.mean(1)
print("corr(mean p_k, ASOS) = %.3f  (어댑터 통과 전)" % np.corrcoef(pb, ob)[0, 1])
edges = [0, 0.1, 1, 2, 4, 6, 8, 10, 14, 18, 22, 26, 34, 100]
print(" bin(mean p_k)      n   E[ASOS|p]  E[p]   E[pred]")
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (pb >= lo) & (pb < hi)
    if s.sum() >= 3:
        print("%6.1f-%5.1f %7d   %7.2f  %6.2f  %6.2f" % (lo, hi, s.sum(), ob[s].mean(), pb[s].mean(), pf[s].mean()))
