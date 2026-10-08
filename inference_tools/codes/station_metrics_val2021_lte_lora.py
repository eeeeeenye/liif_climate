import sys, os, copy, json
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import pandas as pd
import torch
import yaml
import datasets, models
from models.lora import build_lora_lte

device = torch.device("cuda:0")   # CUDA_VISIBLE_DEVICES=2 -> 프로세스 안에서는 cuda:0 (물리 GPU 2번)
# LTE + LoRA(ASOS finetune) 버전.  사용:  python station_metrics_val2021_lte_lora.py <lora_ckpt.pth> <out.csv>
#   base = 사전학습 LTE, ft = LoRA 체크포인트 (학습 중 덮어쓰기 방지를 위해 복사본을 넘길 것)
SAVE = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_lte_lora_r16_asos_lag1h"
CFG = f"{SAVE}/config.yaml"
FT_CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{SAVE}/epoch-last.pth"
META = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/meta_HSR_500m_2019JJA_obscrop_lag1h_val.json"
OBS_CSV = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv"
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/station_eval/tables/station_metrics_val2021_lte_lora_r16.csv"
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

ck = torch.load(FT_CKPT, map_location="cpu", weights_only=False)
print("ft ckpt epoch:", ck["epoch"], "rank:", ck["rank"])

def load_base():
    sv = torch.load(ck["base_ckpt"], map_location="cpu", weights_only=False)
    m = models.make(sv["model"], load_sd=True).to(device)
    m.eval(); m.topo = None
    return m

def load_ft():
    m, _ = build_lora_lte(ck["base_ckpt"], ck["rank"], ck["layers"], ck["lora"], device=device)
    m.eval(); m.topo = None
    return m

base = load_base(); ft = load_ft()

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

pb = run(base); pf = run(ft); ob = np.array(gts) * PMAX

# 역변환한 lat/lon을 CSV 관측소 좌표로 snap (역변환이 맞는지 검증도 겸함)
D = np.sqrt((np.array(lats)[:, None] - S[None, :, 0]) ** 2 + (np.array(lons)[:, None] - S[None, :, 1]) ** 2)
sid = D.argmin(1)
print("snap 최대 거리(deg):", D.min(1).max())

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
print("저장 완료:", OUT_CSV)
print("표시 관측소 수(n>=%d): %d / %d" % (MIN_N, len(df), len(S)))
print("관측소 평균  |bias| base=%.3f ft=%.3f | MAE base=%.3f ft=%.3f | corr base=%.3f ft=%.3f" % (
    df.bias_b.abs().mean(), df.bias_f.abs().mean(), df.mae_b.mean(), df.mae_f.mean(), df.corr_b.mean(), df.corr_f.mean()))
print("finetuned가 더 나은 관측소 비율: |bias| %.0f%%, MAE %.0f%%, corr %.0f%%" % (
    100 * (df.bias_f.abs() < df.bias_b.abs()).mean(), 100 * (df.mae_f < df.mae_b).mean(), 100 * (df.corr_f > df.corr_b).mean()))
