"""
_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h (MLP ft, lag1h) 의 2020 전 계절 관측소 (지점,시각)별 예측값 덤프.
  station_preds_dump_val2021_new.py 의 2020년판. val_dataset(2021 JJA 전용) 대신 hsr2020/<날짜>_{lr,hr}.nc (prep_hsr_2021_nonjja.py 로 생성)를 직접 읽어
  table_hsr_2x_scales_2021.py 와 같은 방식(모델 gen_feat 1회 + 지점 query_rgb)으로 x2(500m) 만 질의.  x4/x8/x16 은 이 그림에 필요 없어 생략.
  lag1h 정렬: ASOS[T] <-> HSR[T-1h] (make_hsr_hourly_obs.py 와 동일).
출력: inference_tools/station_eval/preds_npz/station_preds_2020.npz  (obs, pf, hsr_hr : mm/h; plot_bias_by_category_new.py 와 키 호환, pb 는 이 그림에서 안 씀 -> hsr_hr 로 채움)
사용: python station_preds_dump_2020.py
env: liif, 물리 GPU 2 (CUDA_VISIBLE_DEVICES=2 -> cuda:0)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import copy
import time
import numpy as np
import pandas as pd
import torch
import xarray as xr

import sys
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
import models

dev = torch.device("cuda:0")
NJ = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid/hsr2020"
OBS = "/home/inhye_yoo/data/observation/raw/ASOS/raw/hourly/KMA_ASOS_2020_2024/OBS_ASOS_precip_2020.csv"
META = "/home/inhye_yoo/data/observation/raw/META_관측지점정보_20251115184700.csv"
RUN = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h"
CKPT = os.environ.get("CKPT", f"{RUN}/epoch-best-loss.pth")
OUT_NPZ = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/station_eval/preds_npz/station_preds_2020.npz"
PMAX = 74.8784

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
spec = copy.deepcopy(ck["model"])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
print("2x LIIF epoch", ck.get("epoch"), flush=True)

# --- 관측소 좌표 (META, 2020-07-01 시점) + ASOS 시간별 (lag1h) ---
m = pd.read_csv(META, encoding="cp949").dropna(subset=["지점", "위도", "경도"]).copy()
m["시작일"] = pd.to_datetime(m["시작일"], errors="coerce"); m["종료일"] = pd.to_datetime(m["종료일"], errors="coerce").fillna(pd.Timestamp("2100-01-01"))
ref = pd.Timestamp("2020-07-01"); coords = {}
for sid, g in m.groupby("지점"):
    g = g.sort_values("시작일"); hit = g[(g["시작일"] <= ref) & (g["종료일"] >= ref)]
    row = hit.iloc[-1] if len(hit) else g.iloc[(g["시작일"] - ref).abs().argsort().iloc[0]]
    coords[int(sid)] = (float(row["위도"]), float(row["경도"]))
obs = pd.read_csv(OBS, encoding="cp949", usecols=["지점", "일시", "강수량(mm)"])
obs["t"] = pd.to_datetime(obs["일시"], errors="coerce") - pd.Timedelta(hours=1)          # ASOS[T] -> HSR[T-1h]
obs["v"] = pd.to_numeric(obs["강수량(mm)"], errors="coerce")
obs = obs.dropna(subset=["t", "v"]).drop_duplicates(subset=["지점", "t"], keep="last")
obs = obs[obs["지점"].isin(coords)]
obs["lat"] = obs["지점"].map(lambda s: coords[s][0]); obs["lon"] = obs["지점"].map(lambda s: coords[s][1])
obs["day"] = obs.t.dt.strftime("%Y-%m-%d")
print(f"ASOS 2020: {len(obs)} (지점,시각) 유효, 지점 {obs['지점'].nunique()}", flush=True)

_ref = xr.open_dataset(f"{NJ}/2020-01-01_hr.nc"); LAT, LON = _ref.lat.values, _ref.lon.values
H, W = len(LAT), len(LON); dlat, dlon = np.diff(LAT).mean(), np.diff(LON).mean()
assert dlat > 0 and dlon > 0, "lat/lon 오름차순 가정"


def cell_index(lat, lon):
    """지점 -> 500m 격자의 최근접 셀 (ri, ci). 격자가 완전 균일하지 않아 실제 격자로 보간 (table_hsr_2x_scales_2021.py 와 동일)."""
    fr = np.interp(lat, LAT, np.arange(H)); fc = np.interp(lon, LON, np.arange(W))
    return np.clip(np.rint(fr).astype(int), 0, H - 1), np.clip(np.rint(fc).astype(int), 0, W - 1)


def nearest(vec, x):
    return np.abs(vec[None, :] - x[:, None]).argmin(1)


@torch.no_grad()
def run_day(date):
    f = f"{NJ}/{date}_lr.nc"
    if not os.path.exists(f):
        return None
    lr_ds, hr_ds = xr.open_dataset(f), xr.open_dataset(f"{NJ}/{date}_hr.nc")
    o = obs[obs.day == date]
    o = o[(o.lat >= LAT[0]) & (o.lat <= LAT[-1]) & (o.lon >= LON[0]) & (o.lon <= LON[-1])]
    if not len(o):
        return None
    sids = np.array(sorted(o["지점"].unique()))
    slat = np.array([coords[s][0] for s in sids]); slon = np.array([coords[s][1] for s in sids]); n = len(sids)
    rr, cc = cell_index(slat, slon)
    lr_r, lr_c = nearest(lr_ds.lat.values, slat), nearest(lr_ds.lon.values, slon)
    Hk, Wk = H, W
    coord = torch.from_numpy(np.stack([-1 + (2 * rr + 1) / Hk, -1 + (2 * cc + 1) / Wk], -1).astype(np.float32))[None].to(dev)
    cell = torch.zeros_like(coord); cell[..., 0] = 2.0 / Hk; cell[..., 1] = 2.0 / Wk
    pos = {s: i for i, s in enumerate(sids)}
    rec = {k: [] for k in ["sid", "obs", "pf", "hsr_hr"]}
    for h in range(24):
        oh = o[o.t.dt.hour == h]
        if not len(oh):
            continue
        lr = lr_ds.precip_norm.isel(time=h).values; hr = hr_ds.precip_norm.isel(time=h).values
        model.gen_feat(torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev))
        p = (model.query_rgb(coord, cell).clamp(0, 1) * PMAX).clamp(min=0).cpu().numpy().ravel()
        raw = hr[rr, cc] * PMAX
        for s, v in zip(oh["지점"].values, oh["v"].values):
            i = pos[s]
            if not np.isfinite(raw[i]):                    # 레이더 영역 밖은 제외
                continue
            rec["sid"].append(s); rec["obs"].append(v); rec["pf"].append(p[i]); rec["hsr_hr"].append(raw[i])
    return rec


if __name__ == "__main__":
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range("2020-01-01", "2020-12-31")]
    all_rec = {k: [] for k in ["sid", "obs", "pf", "hsr_hr"]}
    t0 = time.time()
    for i, d in enumerate(days):
        r = run_day(d)
        if r:
            for k in all_rec:
                all_rec[k] += r[k]
        if (i + 1) % 30 == 0:
            print(f"  {d}  ({i + 1}/{len(days)})  {time.time() - t0:.0f}s  누적 {len(all_rec['obs'])}", flush=True)
    obs_a, pf_a, hsr_a = np.array(all_rec["obs"]), np.array(all_rec["pf"]), np.array(all_rec["hsr_hr"])
    np.savez(OUT_NPZ, sid=np.array(all_rec["sid"]), obs=obs_a, pb=hsr_a, pf=pf_a, hsr_hr=hsr_a)   # pb 는 이 그림에서 안 씀(더미=hsr_hr)
    print(f"저장 {OUT_NPZ}  n={len(obs_a)}")
    print(f"평균: obs={obs_a.mean():.3f} hsr={hsr_a.mean():.3f} pf={pf_a.mean():.3f}")
