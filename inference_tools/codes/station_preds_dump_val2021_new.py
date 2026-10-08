import sys, os, copy, json
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import pandas as pd
import torch
import yaml
import datasets, models

device = torch.device("cuda:0")   # CUDA_VISIBLE_DEVICES=2 -> 프로세스 안에서는 cuda:0 (물리 GPU 2번)
CFG = "/home/inhye_yoo/deeplearning/inr/liif/configs/train_climate/climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h.yaml"
BASE_CKPT = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth"
FT_CKPT = "/tmp/claude-1018/-home-inhye-yoo/8baf0072-2a66-45c7-a002-6332494aa55e/scratchpad/ft_mse_new_best_loss.pth"   # ft_mlp_1e-3_lag1h epoch-best-loss (adapter 1회 적용 수정 후)
META = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/meta_HSR_500m_2019JJA_obscrop_lag1h_val.json"
OBS_CSV = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv"
OUT_CSV = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/station_eval/tables/station_metrics_val2021_lag1h_mse_dump_new.csv"
OUT_NPZ = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/station_eval/preds_npz/station_preds_val2021_lag1h_mse_new.npz"
HR_NC = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid/HSR_val_500m_norm_obscrop.nc"
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

val_ds = datasets.make(config['val_dataset']['dataset'])

def load_model(path):
    sv = torch.load(path, map_location='cuda:0', weights_only=False)
    spec = copy.deepcopy(sv['model'])
    m = models.make({"name": spec["name"], "args": spec["args"]}).to(device)
    m.load_state_dict(spec["sd"], strict=True)
    m.eval(); m.topo = None
    return m

base = load_model(BASE_CKPT); ft = load_model(FT_CKPT)
print("ft ckpt epoch:", torch.load(FT_CKPT, map_location="cpu", weights_only=False)["epoch"])

import xarray as xr
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

np.savez(OUT_NPZ, tid=np.array(tids), lat=np.array(lats), lon=np.array(lons), obs=ob, pb=pb, pf=pf, hsr_hr=np.array(hsr_hr) * PMAX)
print('npz 저장:', OUT_NPZ)
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
