"""
지점별 예측 덤프 (Val 2021 JJA, lag1h): base / MLP ft(MSE) / BG(mean mu) 를 같은 (시각,지점) 쌍에서 계산.
station_preds_dump_val2021.py 와 같은 방식 (관측소 중심 48x48 LR crop + 관측소 좌표, cell 2/1000, model(...) forward).

  - 쌍 = fillzero val csv 의 전 (시각,지점) (건조 시각 포함, 212k).  rec = 기존 lag1h csv(NaN 버전)에서 기록이 있던 쌍.
  - BG: adapter 는 LIIF 안에서 local ensemble 전 방향별 1회 -> forward 의 2번째 correction_layer 호출(3채널)은 통과시키는 래퍼.
        mu = p*alpha*beta (mm/h), p 도 저장.
사용:  BG_CKPT=... FT_CKPT=... python station_preds_dump_val2021_bg.py
출력:  inference_tools/station_eval/preds_npz/station_preds_val2021_bg.npz   env: liif, 물리 GPU 2
"""
import sys, os, copy, json
ROOT = "/home/inhye_yoo/deeplearning/inr/liif"
sys.path.insert(0, ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import pandas as pd
import torch
import yaml
import xarray as xr
import datasets, models
import models.bg_adapter          # noqa: F401
from losses_bernoulli_gamma import bg_params

device = torch.device("cuda:0")
CFG = f"{ROOT}/configs/train_climate/climate_edsr_radar_hsr_2x_minmax_liif_ft_bg_mlp_lag1h.yaml"
S_ = f"{ROOT}/save"
BASE_CKPT = f"{S_}/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth"
FT_CKPT = os.environ.get("FT_CKPT", f"{S_}/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h/epoch-best-loss.pth")
BG_CKPT = os.environ.get("BG_CKPT", f"{S_}/_climate_edsr_radar_hsr_2x_minmax_liif_ft_bg_mlp_lag1h/epoch-best-nll.pth")
OLD_CSV = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv"
HR_NC = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid/HSR_val_500m_norm_obscrop.nc"
OUT_NPZ = f"{ROOT}/inference_tools/station_eval/preds_npz/station_preds_val2021_bg.npz"
PMAX, CROP = 74.8784, 48

config = yaml.load(open(CFG), Loader=yaml.FullLoader)
vargs = config['val_dataset']['dataset']['args']
vargs['load_into_memory'] = False
meta = json.load(open(vargs['meta_path']))
lat_max, lat_min = meta['lat_range']; lon_min, lon_max = meta['lon_range']
fz = pd.read_csv(vargs['obs_path_3'])
st = fz[['x', 'y']].drop_duplicates().reset_index(drop=True)
S = st[['y', 'x']].values
print("관측소 수:", len(S), " val csv:", vargs['obs_path_3'].split('/')[-1])

# 기록 여부 (기존 NaN 버전 csv 에서 v 가 있던 쌍)
old = pd.read_csv(OLD_CSV)
assert len(old) == len(fz) and (old.x.values == fz.x.values).all() and (old.time.values == fz.time.values).all()
nc_times = pd.to_datetime(xr.open_dataset(HR_NC).time.values)
tmap = {t: i for i, t in enumerate(nc_times)}
skey = {(x, y): k for k, (x, y) in enumerate(zip(st.x, st.y))}
rec_wide = np.zeros((len(nc_times), len(S)), bool)
rec_wide[pd.to_datetime(old.time).map(tmap).values, [skey[(x, y)] for x, y in zip(old.x, old.y)]] = old.v.notna().values

val_ds = datasets.make(config['val_dataset']['dataset'])


def load_model(path, bg=False):
    sv = torch.load(path, map_location='cuda:0', weights_only=False)
    spec = copy.deepcopy(sv['model'])
    m = models.make({"name": spec["name"], "args": spec["args"]}).to(device)
    m.load_state_dict(spec["sd"], strict=True); m.eval(); m.topo = None
    if bg:   # query_rgb 안(1채널 입력)에서만 adapter 적용, forward 끝의 두 번째 호출(3채널)은 통과
        m.correction_layer = lambda p: m.add_layer(p) if p.shape[-1] == 1 else p
    return m, sv.get("epoch")


base, _ = load_model(BASE_CKPT); ft, ep_f = load_model(FT_CKPT); bgm, ep_b = load_model(BG_CKPT, bg=True)
print(f"ft epoch {ep_f}, bg epoch {ep_b}")

crops, coords, gts, lats, lons, tids = [], [], [], [], [], []
for idx in range(len(val_ds)):
    img_lr, _, obs_, _ = val_ds[idx]
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
        gts.append(float(v[k, 0])); tids.append(idx)
    if idx % 500 == 0: print("crop", idx, "/", len(val_ds), flush=True)

N = len(crops); print("(시각,관측소) 쌍:", N)
crops_t = torch.stack(crops); coords_t = torch.stack(coords).unsqueeze(1)


def run(model, bg=False):
    out, pout = [], []
    with torch.no_grad():
        for s in range(0, N, 512):
            inp = crops_t[s:s + 512].to(device); co = coords_t[s:s + 512].to(device)
            cell = torch.ones_like(co); cell[:, :, 0] *= 2 / 1000; cell[:, :, 1] *= 2 / 1000
            o = model(inp, None, co, cell)
            if bg:
                p, a, b = bg_params(o)
                out.append((p * a * b).reshape(-1).cpu()); pout.append(p.reshape(-1).cpu())
            else:
                out.append(o.clamp(0, 1).reshape(-1).cpu() * PMAX)
    return (torch.cat(out).numpy(), torch.cat(pout).numpy()) if bg else torch.cat(out).numpy()


pb = run(base); pf = run(ft); mu, pp = run(bgm, bg=True)
lats, lons = np.array(lats), np.array(lons)
D = np.sqrt((lats[:, None] - S[None, :, 0]) ** 2 + (lons[:, None] - S[None, :, 1]) ** 2)
sid = D.argmin(1); print("snap 최대 거리(deg):", D.min(1).max())
tids = np.array(tids)
np.savez(OUT_NPZ, tid=tids, sid=sid, lat=S[sid, 0], lon=S[sid, 1], obs=np.array(gts) * PMAX, pb=pb, pf=pf, mu=mu, p=pp,
         rec=rec_wide[tids, sid], ep_ft=ep_f, ep_bg=ep_b)
print("saved", OUT_NPZ, " rec 쌍:", int(rec_wide[tids, sid].sum()))
