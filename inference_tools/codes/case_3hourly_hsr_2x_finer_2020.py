"""
case_3hourly_hsr_2x_finer.py 의 2020년판.  2021 JJA val nc 대신 hsr2020/<날짜>_{lr,hr}.nc (prep_hsr_2021_nonjja.py 로 생성, 366일 전부 있음) 를 직접 읽고,
ASOS 는 2020 원본(OBS_ASOS_precip_2020.csv, lag1h) 사용.  나머지 로직(배율/저장 포맷)은 원본과 동일 -> plot_case_3hourly_hsr_2x_finer.py 그대로 재사용 가능.
사용: python case_3hourly_hsr_2x_finer_2020.py 2020-08-07
      전 도메인: REG_H=1326 REG_W=1234 TAG=_full NOCOMPRESS=1 python ...
출력: inference_tools/case_studies/3hourly/case_3h_<날짜>_2x_finer<TAG>.npz   (2021 판과 파일명 규칙 동일)
env: liif, 물리 GPU 2
"""
import os, sys, copy, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.ndimage import uniform_filter
import models

dev = torch.device("cuda:0")
NJ = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid/hsr2020"
OBS = "/home/inhye_yoo/data/observation/raw/ASOS/raw/hourly/KMA_ASOS_2020_2024/OBS_ASOS_precip_2020.csv"
CKPT = os.environ.get("CKPT", "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h/epoch-best-loss.pth")
TAG = os.environ.get("TAG", "")
OUTDIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
PMAX = 74.8784
BSIZE = int(os.environ.get("BSIZE", "500000"))
REG_H, REG_W = int(os.environ.get("REG_H", 200)), int(os.environ.get("REG_W", 200))
dates = sys.argv[1:] or ["2020-08-07"]

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
spec = copy.deepcopy(ck["model"])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
print("2x LIIF epoch", ck.get("epoch"))

_ref = xr.open_dataset(f"{NJ}/2020-01-01_hr.nc"); lat, lon = _ref.lat.values, _ref.lon.values
H, W = _ref.sizes["lat"], _ref.sizes["lon"]
dlat, dlon = np.diff(lat).mean(), np.diff(lon).mean()
obs = pd.read_csv(OBS, encoding="cp949", usecols=["지점", "일시", "강수량(mm)"])
obs["t"] = pd.to_datetime(obs["일시"], errors="coerce") - pd.Timedelta(hours=1)      # ASOS[T] -> HSR[T-1h] (make_hsr_hourly_obs.py 와 동일)
obs["v"] = pd.to_numeric(obs["강수량(mm)"], errors="coerce")
obs = obs.dropna(subset=["t", "v"]).drop_duplicates(subset=["지점", "t"], keep="last")
META = "/home/inhye_yoo/data/observation/raw/META_관측지점정보_20251115184700.csv"
m = pd.read_csv(META, encoding="cp949").dropna(subset=["지점", "위도", "경도"]).copy()
m["시작일"] = pd.to_datetime(m["시작일"], errors="coerce"); m["종료일"] = pd.to_datetime(m["종료일"], errors="coerce").fillna(pd.Timestamp("2100-01-01"))
ref = pd.Timestamp("2020-07-01"); coords = {}
for sid, g in m.groupby("지점"):
    g = g.sort_values("시작일"); hit = g[(g["시작일"] <= ref) & (g["종료일"] >= ref)]
    row = hit.iloc[-1] if len(hit) else g.iloc[(g["시작일"] - ref).abs().argsort().iloc[0]]
    coords[int(sid)] = (float(row["위도"]), float(row["경도"]))
obs["x"] = obs["지점"].map(lambda s: coords.get(s, (np.nan, np.nan))[1])          # lon
obs["y"] = obs["지점"].map(lambda s: coords.get(s, (np.nan, np.nan))[0])          # lat
obs = obs.dropna(subset=["x", "y"])


@torch.no_grad()
def predict(coord, cell, shape):
    out = [model.query_rgb(coord[:, s:s + BSIZE], cell[:, s:s + BSIZE]) for s in range(0, coord.shape[1], BSIZE)]
    return (torch.cat(out, 1).view(*shape).clamp(0, 1) * PMAX).clamp(min=0).cpu().numpy()


def query_grid(r0, r1, c0, c1, k):
    Hk, Wk = int(round(H * k)), int(round(W * k))
    ri = np.arange(int(round(r0 * k)), int(round(r1 * k))); ci = np.arange(int(round(c0 * k)), int(round(c1 * k)))
    yy, xx = -1 + (2 * ri + 1) / Hk, -1 + (2 * ci + 1) / Wk
    coord = torch.from_numpy(np.stack(np.meshgrid(yy, xx, indexing="ij"), -1).reshape(-1, 2).astype(np.float32)).unsqueeze(0).to(dev)
    cell = torch.empty_like(coord); cell[:, :, 0] = 2.0 / Hk; cell[:, :, 1] = 2.0 / Wk
    la = lat[0] - dlat / 2 + (ri + 0.5) * (H * dlat / Hk); lo = lon[0] - dlon / 2 + (ci + 0.5) * (W * dlon / Wk)
    return coord, cell, (len(ri), len(ci)), la, lo


for date in dates:
    npz = f"{OUTDIR}/case_studies/3hourly/case_3h_{date}_2x_finer{TAG}.npz"
    if os.path.exists(npz):
        print("이미 있음, 건너뜀:", npz); continue
    hr_ds = xr.open_dataset(f"{NJ}/{date}_hr.nc"); lr_ds = xr.open_dataset(f"{NJ}/{date}_lr.nc")
    assert np.allclose(hr_ds.lat.values, lat) and np.allclose(hr_ds.lon.values, lon), date
    gt_d = np.stack([np.nan_to_num(hr_ds.precip_norm.isel(time=t).values) * PMAX for t in range(24)]).sum(0)
    o = obs[(obs.t.dt.strftime("%Y-%m-%d") == date) & obs.v.notna()]
    sxy = o.groupby(["x", "y"]).size().reset_index()[["x", "y"]].values
    sr = np.abs(lat[None, :] - sxy[:, 1:2]).argmin(1); sc = np.abs(lon[None, :] - sxy[:, 0:1]).argmin(1)
    if os.environ.get("CENTER"):
        clat, clon = (float(v) for v in os.environ["CENTER"].split(","))
        cr, cc = np.abs(lat - clat).argmin(), np.abs(lon - clon).argmin()
    else:
        sm = uniform_filter(gt_d, 90, mode="constant")
        best, cr, cc = -1, None, None
        for r_, c_ in zip(sr, sc):
            n_in = int(((np.abs(sr - r_) < REG_H // 2) & (np.abs(sc - c_) < REG_W // 2)).sum())
            if n_in >= 5 and sm[r_, c_] > best:
                best, cr, cc = sm[r_, c_], r_, c_
    r0 = int(np.clip(cr - REG_H // 2, 0, H - REG_H)) // 2 * 2; c0 = int(np.clip(cc - REG_W // 2, 0, W - REG_W)) // 2 * 2
    r1, c1 = r0 + REG_H, c0 + REG_W
    print(f"{date}: 창 rows {r0}:{r1} cols {c0}:{c1} -> lat {lat[r0]:.2f}~{lat[r1 - 1]:.2f}, lon {lon[c0]:.2f}~{lon[c1 - 1]:.2f}")
    g500 = query_grid(r0, r1, c0, c1, 1.0); g250 = query_grid(r0, r1, c0, c1, 2.0); g125 = query_grid(r0, r1, c0, c1, 4.0); g62 = query_grid(r0, r1, c0, c1, 8.0)

    raw = np.zeros((8, REG_H, REG_W), np.float32); inp = np.zeros((8, REG_H // 2, REG_W // 2), np.float32); m500 = np.zeros((8,) + g500[2], np.float32)
    m250 = np.zeros((8,) + g250[2], np.float32); m125 = np.zeros((8,) + g125[2], np.float32); m62 = np.zeros((8,) + g62[2], np.float32)
    valid = np.zeros((REG_H, REG_W), bool); t0 = time.time()
    for k in range(24):
        b = k // 3
        hr = hr_ds.precip_norm.isel(time=k).values; lr = lr_ds.precip_norm.isel(time=k).values
        hr = hr[r0:r1, c0:c1]; lr_w = lr[r0 // 2:r1 // 2, c0 // 2:c1 // 2]
        valid |= np.isfinite(hr)
        raw[b] += np.nan_to_num(hr) * PMAX; inp[b] += np.nan_to_num(lr_w) * PMAX
        x = torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev)
        with torch.no_grad():
            model.gen_feat(x)
        m500[b] += predict(g500[0], g500[1], g500[2])
        m250[b] += predict(g250[0], g250[1], g250[2]); m125[b] += predict(g125[0], g125[1], g125[2]); m62[b] += predict(g62[0], g62[1], g62[2])
        torch.cuda.empty_cache()
        print(f"  {k + 1}/24h  {time.time() - t0:.0f}s", flush=True)

    o = obs[(obs.t.dt.strftime("%Y-%m-%d") == date) & obs.v.notna()].copy()
    o["b3"] = o.t.dt.hour // 3
    st = o.groupby(["x", "y", "b3"]).v.sum().unstack("b3").reindex(columns=range(8))
    (np.savez if os.environ.get("NOCOMPRESS") else np.savez_compressed)(npz, raw=raw, inp=inp, m500=m500, m250=m250, m125=m125, m62=m62.astype(np.float16), valid=valid, region=np.array([r0, r1, c0, c1]),
                        lat=lat[r0:r1], lon=lon[c0:c1], lr_lat=lr_ds.lat.values[r0 // 2:r1 // 2], lr_lon=lr_ds.lon.values[c0 // 2:c1 // 2],
                        lat250=g250[3], lon250=g250[4], lat125=g125[3], lon125=g125[4], lat62=g62[3], lon62=g62[4],
                        st_lat=st.index.get_level_values("y").values, st_lon=st.index.get_level_values("x").values, st_asos=st.values.T,
                        ep=ck.get("epoch"))
    print("saved", npz, f"({time.time() - t0:.0f}s)")
