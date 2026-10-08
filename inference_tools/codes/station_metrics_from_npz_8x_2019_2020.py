"""
station_preds_dump_val2021_8x_ft_2019_2020.py 가 만든 npz (obs, pf, hsr_hr; mm/h) 로 지점별 지표 CSV 를 만든다 (CPU 만 사용, 모델 재추론 없음).
  base = 원시 HSR 500m 최근접 (hsr_hr), ft = 8x MLP ft (pf).  관측소 좌표는 obs csv 의 (x,y) 로 snap.
사용: python station_metrics_from_npz_8x_2019_2020.py [npz] [out_csv]      env: liif
"""
import sys
import numpy as np, pandas as pd

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
NPZ = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/station_eval/preds_npz/station_preds_val2021_8x_ft_2019_2020_epoch-best-loss.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/station_eval/tables/station_metrics_val2021_raw_vs_8x_ft_2019_2020.csv"
OBS_CSV = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv"
MIN_N = 30

d = np.load(NPZ); ob, pf, pb = d["obs"], d["pf"], d["hsr_hr"]
S = pd.read_csv(OBS_CSV)[["x", "y"]].drop_duplicates().reset_index(drop=True)[["y", "x"]].values   # lat, lon
D = np.sqrt((d["lat"][:, None] - S[None, :, 0]) ** 2 + (d["lon"][:, None] - S[None, :, 1]) ** 2)
sid = D.argmin(1); print("snap 최대 거리(deg):", D.min(1).max())
rows = []
for s in range(len(S)):
    m = sid == s; n = int(m.sum())
    if n < MIN_N: continue
    o, a, b = ob[m], pb[m], pf[m]
    corr = lambda p: np.corrcoef(p, o)[0, 1] if p.std() > 0 and o.std() > 0 else np.nan
    rows.append(dict(lat=S[s, 0], lon=S[s, 1], n=n, bias_b=(a - o).mean(), bias_f=(b - o).mean(),
                     mae_b=np.abs(a - o).mean(), mae_f=np.abs(b - o).mean(), corr_b=corr(a), corr_f=corr(b)))
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
print(f"stations {len(df)}/{len(S)}, n total {df.n.sum()}  saved {OUT}")
