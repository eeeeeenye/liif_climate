import sys, os, copy, json
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import pandas as pd
import torch
import yaml
import datasets, models

device = torch.device("cuda:0")   # CUDA_VISIBLE_DEVICES=2 -> 프로세스 안에서는 cuda:0 (물리 GPU 2번)
SAVE = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_asos_lag1h"
CFG = f"{SAVE}/config.yaml"   # val 데이터는 기존 lag1h 와 동일
BASE_CKPT = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth"
# 사용: python station_metrics_val2021_asos_direct.py <ft_ckpt1>:<out1.csv> [<ft_ckpt2>:<out2.csv> ...]   (ASOS 직접 파인튜닝 LIIF, adapter 없음)
JOBS = [a.split(":") for a in sys.argv[1:]]
META = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/meta_HSR_500m_2019JJA_obscrop_lag1h_val.json"
OBS_CSV = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv"
PMAX = 74.8784
CROP = 48
MIN_N = 30

with open(CFG) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
meta = json.load(open(META))
lat_max, lat_min = meta['lat_range']
lon_min, lon_max = meta['lon_range']

st = pd.read_csv(OBS_CSV)[['x', 'y']].drop_duplicates().reset_index(drop=True)
S = st[['y', 'x']].values  # lat, lon
print("CSV 관측소 수:", len(S))

config['val_dataset']['dataset']['args']['load_into_memory'] = False   # 전 시각 한 번씩만 읽으므로 RAM 적재 불필요
val_ds = datasets.make(config['val_dataset']['dataset'])

def load_model(path):
    sv = torch.load(path, map_location='cuda:0', weights_only=False)
    spec = copy.deepcopy(sv['model'])
    m = models.make({"name": spec["name"], "args": spec["args"]}).to(device)
    m.load_state_dict(spec["sd"], strict=True)
    m.eval(); m.topo = None
    return m

base = load_model(BASE_CKPT)

crops, coords, gts, lats, lons = [], [], [], [], []
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
        gts.append(float(v[k, 0]))

N = len(crops)
print("(시각,관측소) 쌍:", N)
crops_t = torch.stack(crops); coords_t = torch.stack(coords).unsqueeze(1)

def run(model):
    out = []
    with torch.no_grad():
        for s in range(0, N, 512):
            inp = crops_t[s:s + 512].to(device); co = coords_t[s:s + 512].to(device)
            cell = torch.ones_like(co); cell[:, :, 0] *= 2 / 1000; cell[:, :, 1] *= 2 / 1000
            out.append(model(inp, None, co, cell).clamp(0, 1).squeeze(-1).squeeze(-1).cpu())
    return torch.cat(out).numpy() * PMAX

pb = run(base); ob = np.array(gts) * PMAX
D = np.sqrt((np.array(lats)[:, None] - S[None, :, 0]) ** 2 + (np.array(lons)[:, None] - S[None, :, 1]) ** 2)
sid = D.argmin(1)
print("snap 최대 거리(deg):", D.min(1).max())

for FT_CKPT, OUT_CSV in JOBS:
    ft = load_model(FT_CKPT)
    print("ft ckpt:", FT_CKPT, "epoch:", torch.load(FT_CKPT, map_location="cpu", weights_only=False)["epoch"])
    pf = run(ft)
    rows = []
    for s in range(len(S)):
        m = sid == s
        n = int(m.sum())
        if n < MIN_N: continue
        o, a, b = ob[m], pb[m], pf[m]
        def corr(p): return np.corrcoef(p, o)[0, 1] if p.std() > 0 and o.std() > 0 else np.nan
        rows.append(dict(lat=S[s, 0], lon=S[s, 1], n=n,
                         bias_b=(a - o).mean(), bias_f=(b - o).mean(),
                         mae_b=np.abs(a - o).mean(), mae_f=np.abs(b - o).mean(),
                         corr_b=corr(a), corr_f=corr(b)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("저장:", OUT_CSV, "| 지점 %d/%d" % (len(df), len(S)))
    print("전체 pooled: base bias %+.3f MAE %.3f | ft bias %+.3f MAE %.3f" % ((pb - ob).mean(), np.abs(pb - ob).mean(), (pf - ob).mean(), np.abs(pf - ob).mean()))
    print("지점평균 |bias| base=%.3f ft=%.3f | MAE base=%.3f ft=%.3f | corr base=%.3f ft=%.3f" % (
        df.bias_b.abs().mean(), df.bias_f.abs().mean(), df.mae_b.mean(), df.mae_f.mean(), df.corr_b.mean(), df.corr_f.mean()))
