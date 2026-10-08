"""
HSR 2x LIIF (MLP ft lag1h) 의 2021 전 계절 관측소 지표표: Input(1km) / GT(Raw HSR 500m) / x2 / x4 / x8 / x16  vs  ASOS 시간별 (mm/h).
  - 입력은 1km LR 고정, x2=500m(학습 배율) / x4=250m / x8=125m / x16=62.5m (x4 이상은 외삽).  관측소가 속한 각 배율 격자의 최근접 셀 중심을 질의.
  - lag1h 정렬: ASOS[T] <-> HSR[T-1h] (make_hsr_hourly_obs.py 와 동일).  ASOS 결측(NaN/미기록) 시각은 제외 (기록 없음 != 무강수).
  - Raw HSR 가 NaN 인 (지점, 시각) (레이더 영역 밖) 은 모든 행에서 제외 -> 행끼리 같은 표본.
  - 모델은 시간별 인코더를 전체 LR 에 한 번 돌리고 지점만 질의 (case_3hourly_hsr_2x_finer.py 와 같은 좌표 규약). 어댑터는 query_rgb 안에서 자동 적용.
  - 날짜별 npz 캐시(OUT/days/<날짜>.npz)라 끊겨도 이어서. 마지막에 records + 표(csv) 생성.
  - 데이터: 2021 JJA = HSR_val_*_obscrop.nc, 그 밖 = hsr2021_nonjja/<날짜>_{lr,hr}.nc (12월은 미준비 -> 있는 날까지만).
사용:  python table_hsr_2x_scales_2021.py            # 전체 (있는 날 전부)
       python table_hsr_2x_scales_2021.py 2021-07-06 # 특정 날짜만 (시험용, 표는 만들지 않음)
       CKPT=<pth> OUT=<dir> ...
출력:  inference_tools/station_eval/tables/table_2021_scales/{days/*.npz, table_2021_scales.csv}
env: liif, 물리 GPU 2 (CUDA_VISIBLE_DEVICES=2 -> cuda:0)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import copy
import glob
import time
import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
import models

dev = torch.device("cuda:0")
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid"
NJ = f"{M}/hsr2021_nonjja"
LR_JJA, HR_JJA = f"{M}/HSR_val_norm_2x_obscrop.nc", f"{M}/HSR_val_500m_norm_obscrop.nc"
OBS = "/home/inhye_yoo/data/observation/raw/ASOS/raw/hourly/KMA_ASOS_2020_2024/OBS_ASOS_precip_2021.csv"
META = "/home/inhye_yoo/data/observation/raw/META_관측지점정보_20251115184700.csv"
RUN = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h"
CKPT = os.environ.get("CKPT", f"{RUN}/epoch-best-loss.pth")
OUT = os.environ.get("OUT", "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/station_eval/tables/table_2021_scales")
PMAX = 74.8784
KS = [1, 2, 4, 8]                       # x2/x4/x8/x16 = 500/250/125/62.5 m (500m 격자 대비 촘촘한 정도)
ROWS = ["inp", "raw", "m500", "m250", "m125", "m62"]
os.makedirs(f"{OUT}/days", exist_ok=True)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
spec = copy.deepcopy(ck["model"])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
print("2x LIIF epoch", ck.get("epoch"), flush=True)

# --- 관측소 좌표 (META, 2021-07-01 시점) + ASOS 시간별 (lag1h) ---
m = pd.read_csv(META, encoding="cp949").dropna(subset=["지점", "위도", "경도"]).copy()
m["시작일"] = pd.to_datetime(m["시작일"], errors="coerce"); m["종료일"] = pd.to_datetime(m["종료일"], errors="coerce").fillna(pd.Timestamp("2100-01-01"))
ref = pd.Timestamp("2021-07-01"); coords = {}
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
print(f"ASOS 2021: {len(obs)} (지점,시각) 유효, 지점 {obs['지점'].nunique()}", flush=True)


def open_day(date):
    """(lr_ds, hr_ds, time index list) 또는 None."""
    f = f"{NJ}/{date}_lr.nc"
    if os.path.exists(f):
        return xr.open_dataset(f), xr.open_dataset(f"{NJ}/{date}_hr.nc"), list(range(24))
    if "06-01" <= date[5:] <= "08-31":
        t = pd.to_datetime(JJA_HR.time.values).strftime("%Y-%m-%d")
        idx = list(np.where(t == date)[0])
        if len(idx) == 24:
            return JJA_LR, JJA_HR, idx
    return None


JJA_LR, JJA_HR = xr.open_dataset(LR_JJA), xr.open_dataset(HR_JJA)
_ref = xr.open_dataset(f"{NJ}/2021-01-01_hr.nc")
LAT, LON = _ref.lat.values, _ref.lon.values
_lr = xr.open_dataset(f"{NJ}/2021-01-01_lr.nc"); LLAT, LLON = _lr.lat.values, _lr.lon.values
assert np.allclose(LAT, JJA_HR.lat.values) and np.allclose(LON, JJA_HR.lon.values), "JJA/nonjja 격자 불일치"
assert np.allclose(LLAT, JJA_LR.lat.values) and np.allclose(LLON, JJA_LR.lon.values), "JJA/nonjja LR 격자 불일치"
H, W = len(LAT), len(LON); dlat, dlon = np.diff(LAT).mean(), np.diff(LON).mean()
assert dlat > 0 and dlon > 0, "lat/lon 오름차순 가정"


def cell_index(lat, lon, k):
    """지점 -> k 배 촘촘한 격자의 최근접 셀 (ri, ci).  lat/lon 격자가 완전 균일하지 않아(위도 간격 표준편차 2e-5, 누적 ~3셀) 실제 격자에서
    지점의 (소수) 500m 인덱스를 보간으로 구한 뒤 k 배 격자로 옮김 -> k=1 은 기존 예측 nc 의 최근접 셀과 정확히 일치."""
    fr = np.interp(lat, LAT, np.arange(H)); fc = np.interp(lon, LON, np.arange(W))
    ri = np.rint((fr + 0.5) * k - 0.5).astype(int); ci = np.rint((fc + 0.5) * k - 0.5).astype(int)
    return np.clip(ri, 0, H * k - 1), np.clip(ci, 0, W * k - 1)


def nearest(vec, x):
    return np.abs(vec[None, :] - x[:, None]).argmin(1)


@torch.no_grad()
def run_day(date):
    out = f"{OUT}/days/{date}.npz"
    if os.path.exists(out):
        return
    od = open_day(date)
    if od is None:
        print(f"{date}: 입력 없음, 건너뜀"); return
    lr_ds, hr_ds, tidx = od
    o = obs[obs.day == date]
    o = o[(o.lat >= LAT[0]) & (o.lat <= LAT[-1]) & (o.lon >= LON[0]) & (o.lon <= LON[-1])]        # 격자 안 지점만
    if not len(o):
        np.savez(out, sid=np.zeros(0, int), hour=np.zeros(0, int), obs=np.zeros(0), **{k: np.zeros(0, np.float32) for k in ROWS}); return
    sids = np.array(sorted(o["지점"].unique()))
    slat = np.array([coords[s][0] for s in sids]); slon = np.array([coords[s][1] for s in sids]); n = len(sids)
    rr, cc = nearest(LAT, slat), nearest(LON, slon)
    lr_r, lr_c = nearest(LLAT, slat), nearest(LLON, slon)
    coord, cell = [], []
    for k in KS:
        ri, ci = cell_index(slat, slon, k)
        Hk, Wk = H * k, W * k
        coord.append(np.stack([-1 + (2 * ri + 1) / Hk, -1 + (2 * ci + 1) / Wk], -1))
        c_ = np.zeros((n, 2)); c_[:, 0] = 2.0 / Hk; c_[:, 1] = 2.0 / Wk; cell.append(c_)
    coord = torch.from_numpy(np.concatenate(coord).astype(np.float32))[None].to(dev)
    cell = torch.from_numpy(np.concatenate(cell).astype(np.float32))[None].to(dev)
    rec = {k: [] for k in ["sid", "hour", "obs"] + ROWS}
    for h, ti in enumerate(tidx):
        oh = o[o.t.dt.hour == h]
        if not len(oh):
            continue
        lr = lr_ds.precip_norm.isel(time=ti).values; hr = hr_ds.precip_norm.isel(time=ti).values
        model.gen_feat(torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev))
        p = (model.query_rgb(coord, cell).clamp(0, 1) * PMAX).clamp(min=0).cpu().numpy().reshape(len(KS), n)
        raw = hr[rr, cc] * PMAX; inp = lr[lr_r, lr_c] * PMAX
        pos = {s: i for i, s in enumerate(sids)}
        for s, v in zip(oh["지점"].values, oh["v"].values):
            i = pos[s]
            rec["sid"].append(s); rec["hour"].append(h); rec["obs"].append(v)
            rec["inp"].append(inp[i]); rec["raw"].append(raw[i])
            for name, kk in zip(["m500", "m250", "m125", "m62"], range(len(KS))):
                rec[name].append(p[kk, i])
    np.savez(out, sid=np.array(rec["sid"], int), hour=np.array(rec["hour"], int), obs=np.array(rec["obs"], float),
             **{k: np.array(rec[k], np.float32) for k in ROWS})


def build_table():
    parts = []
    for f in sorted(glob.glob(f"{OUT}/days/*.npz")):
        z = np.load(f); d = os.path.basename(f)[:-4]
        if len(z["obs"]):
            parts.append(pd.DataFrame({"day": d, "sid": z["sid"], "hour": z["hour"], "obs": z["obs"], **{k: z[k] for k in ROWS}}))
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df); df = df[np.isfinite(df["raw"]) & np.isfinite(df["inp"])].reset_index(drop=True)      # 레이더 밖 제외 (행끼리 같은 표본)
    print(f"표본 {n0} -> 레이더 유효 {len(df)}")
    mon = pd.to_datetime(df["day"]).dt.month
    seas = {"Whole": mon >= 1, "JFM": mon.isin([1, 2, 3]), "AMJ": mon.isin([4, 5, 6]), "JAS": mon.isin([7, 8, 9]), "OND": mon.isin([10, 11, 12])}
    recs = []
    for name in ROWS:
        for met in ["RMSE", "MAE", "Bias", "r"]:
            row = {"Model": name, "Metric": met}
            for sn, msk in seas.items():
                v, o = df.loc[msk, name].values, df.loc[msk, "obs"].values
                if met == "r":
                    row[sn] = np.corrcoef(v, o)[0, 1]
                else:
                    e = v - o
                    row[sn] = np.sqrt(np.mean(e ** 2)) if met == "RMSE" else (np.mean(np.abs(e)) if met == "MAE" else np.mean(e))
            recs.append(row)
    tab = pd.DataFrame(recs)
    for sn in seas:
        tab[sn] = tab[sn].map(lambda v: f"{v:.4f}")                                    # 소수점 4자리
    n_row = {"Model": "n", "Metric": "samples", **{sn: str(int(msk.sum())) for sn, msk in seas.items()}}
    tab = pd.concat([tab, pd.DataFrame([n_row])], ignore_index=True)
    tab.to_csv(f"{OUT}/table_2021_scales.csv", index=False)
    df.to_csv(f"{OUT}/records_2021.csv.gz", index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    args = sys.argv[1:]
    days = args or [d.strftime("%Y-%m-%d") for d in pd.date_range("2021-01-01", "2021-12-31")]
    t0 = time.time()
    for i, d in enumerate(days):
        run_day(d)
        if (i + 1) % 10 == 0 or args:
            print(f"  {d}  ({i + 1}/{len(days)})  {time.time() - t0:.0f}s", flush=True)
    if not args:
        build_table()
