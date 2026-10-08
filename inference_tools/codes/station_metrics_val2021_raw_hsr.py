""" raw HSR(500m, 2021 JJA val) vs ASOS 지점별 지표 -> 기존 finetune CSV 의 'base' 자리에 raw 를 넣은 CSV 생성.

    - 쌍: val obs CSV(lag1h 정렬본)의 값이 있는 (시각, 관측소) 행 = 기존 지표와 동일 31,533 쌍.
      CSV 시각 라벨은 이미 ASOS[T] -> T-1h 로 옮겨져 있어 nc time 과 그대로 매칭.
    - raw = HSR_val_500m_norm_obscrop.nc 의 관측소 최근접 격자값 x 74.8784 (mm/h).  모델/보간 없음.
    - 입력 ft 는 기존 station_metrics_val2021_lag1h_new.csv 의 *_f 열을 그대로 재사용.
    출력: 컬럼이 기존과 같고 *_b = raw, *_f = finetuned.  (plot_station_metric_maps.py 에 그대로 넣을 수 있음)

    python station_metrics_val2021_raw_hsr.py [ft_csv] [out_csv]      env: liif (CPU 만 사용)
"""
import sys
import numpy as np
import pandas as pd
import xarray as xr

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
FT_CSV = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/station_eval/tables/station_metrics_val2021_lag1h_new.csv"
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/station_eval/tables/station_metrics_val2021_raw_vs_ft_lag1h_new.csv"
NC = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid/HSR_val_500m_norm_obscrop.nc"
OBS = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv"
PMAX = 74.8784
MIN_N = 30

obs = pd.read_csv(OBS).dropna(subset=["v"]).reset_index(drop=True)
obs["time"] = pd.to_datetime(obs["time"])
print("유효 (시각,관측소) 쌍:", len(obs))

ds = xr.open_dataset(NC)
tvals = pd.DatetimeIndex(ds.time.values)
lat, lon = ds.lat.values, ds.lon.values
assert (np.diff(lat) > 0).all() and (np.diff(lon) > 0).all()

def nearest(arr, v):
    i = np.clip(np.searchsorted(arr, v), 1, len(arr) - 1)
    return np.where(np.abs(arr[i] - v) < np.abs(arr[i - 1] - v), i, i - 1)

t_idx = tvals.get_indexer(obs["time"])
assert (t_idx >= 0).all(), "CSV 시각이 nc time 에 없음"
iy, ix = nearest(lat, obs["y"].values), nearest(lon, obs["x"].values)
print("최근접 격자 최대 거리(deg): lat %.5f lon %.5f" % (np.abs(lat[iy] - obs["y"]).max(), np.abs(lon[ix] - obs["x"]).max()))

raw = np.empty(len(obs), dtype=np.float32)
var = ds["precip_norm"]
for t in np.unique(t_idx):                      # 시각별로 한 번씩만 슬라이스 읽기
    sel = np.where(t_idx == t)[0]
    sl = var.isel(time=int(t)).values           # [lat, lon]
    raw[sel] = sl[iy[sel], ix[sel]] * PMAX
o = obs["v"].values.astype(np.float64)
print("전체: raw mean %.3f obs mean %.3f  bias %+.3f  MAE %.3f  r %.3f" % (
    raw.mean(), o.mean(), (raw - o).mean(), np.abs(raw - o).mean(), np.corrcoef(raw, o)[0, 1]))

ft = pd.read_csv(FT_CSV)
key = lambda d: list(zip(d["lat"].round(4), d["lon"].round(4)))
rows = []
obs["sk"] = key(obs.rename(columns={"y": "lat", "x": "lon"}))
for (la, lo), g in obs.groupby("sk"):
    if len(g) < MIN_N:
        continue
    p, q = raw[g.index.values], o[g.index.values]
    r = np.corrcoef(p, q)[0, 1] if p.std() > 0 and q.std() > 0 else np.nan
    rows.append(dict(lat=la, lon=lo, n=len(g), bias_b=(p - q).mean(), mae_b=np.abs(p - q).mean(), corr_b=r))
rw = pd.DataFrame(rows)

ft["lat_k"], ft["lon_k"] = ft["lat"].round(4), ft["lon"].round(4)
rw["lat_k"], rw["lon_k"] = rw["lat"], rw["lon"]
m = ft[["lat_k", "lon_k", "n", "bias_f", "mae_f", "corr_f"]].merge(
    rw[["lat_k", "lon_k", "n", "bias_b", "mae_b", "corr_b"]], on=["lat_k", "lon_k"], suffixes=("", "_raw"))
assert len(m) == len(ft) == len(rw), (len(m), len(ft), len(rw))
assert (m["n"] == m["n_raw"]).all(), "ft 지표와 raw 지표의 표본 수가 지점별로 다름"
m["lat"], m["lon"] = ft["lat"].values, ft["lon"].values   # 병합 순서가 ft 와 동일(left 기준)
out = m[["lat", "lon", "n", "bias_b", "bias_f", "mae_b", "mae_f", "corr_b", "corr_f"]]
out.to_csv(OUT_CSV, index=False)
print("저장:", OUT_CSV)
print("지점평균 |bias| raw %.3f -> ft %.3f | MAE %.3f -> %.3f | corr %.3f -> %.3f" % (
    out.bias_b.abs().mean(), out.bias_f.abs().mean(), out.mae_b.mean(), out.mae_f.mean(), out.corr_b.mean(), out.corr_f.mean()))
