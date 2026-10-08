"""
case_3hourly_hsr_lag1h_new.py 의 2020년판.  2021 val JJA obscrop nc 대신 hsr2020/<날짜>_{lr,hr}.nc
(prep_hsr_2021_nonjja.py 로 이미 생성, 366일 전부 있음) 를 날짜별로 직접 읽고, ASOS 는 2020 원본
(OBS_ASOS_precip_2020.csv, lag1h) 사용.  나머지(모델 두 개: Radar pretrain / MLP ft, 저장 포맷)는 동일
-> plot_case_3hourly_new.py 를 PERIOD 만 바꿔서 그대로 재사용.
사용:  python case_3hourly_hsr_lag1h_2020.py 2020-06-01 2020-06-02 ...
출력:  <OUTDIR>/case_3h_<날짜>_lag1h_new.npz
env: liif, 물리 GPU 2 (CUDA_VISIBLE_DEVICES=2 -> cuda:0)
"""
import os, sys, copy
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm
import models
from utils import make_coord

dev = torch.device("cuda:0")
NJ = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid/hsr2020"
OBS = "/home/inhye_yoo/data/observation/raw/ASOS/raw/hourly/KMA_ASOS_2020_2024/OBS_ASOS_precip_2020.csv"
META = "/home/inhye_yoo/data/observation/raw/META_관측지점정보_20251115184700.csv"
S = "/home/inhye_yoo/deeplearning/inr/liif/save"
BASE_CKPT = f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth"
FT_CKPT = os.environ.get("FT_CKPT", f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h/epoch-best-loss.pth")
OUTDIR = os.environ.get("OUTDIR", "/home/inhye_yoo/deeplearning/inr/liif/inference_tools")
PMAX = 74.8784
BSIZE = 500_000
dates = sys.argv[1:] or ["2020-07-06"]


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    spec = copy.deepcopy(ck["model"])
    m = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
    m.load_state_dict(spec["sd"], strict=True); m.eval(); m.topo = None
    return m, ck.get("epoch")


@torch.no_grad()
def predict(model, lr_norm, coord, cell, H, W, adapter):
    model.gen_feat(lr_norm)
    out = []
    for s in range(0, coord.shape[1], BSIZE):
        p = model.query_rgb(coord[:, s:s + BSIZE], cell[:, s:s + BSIZE])
        if adapter:
            p = model.correction_layer(p)
        out.append(p)
    p = torch.cat(out, 1).view(H, W).clamp(0, 1) * PMAX
    return p.clamp(min=0).cpu().numpy()


base, ep_b = load_model(BASE_CKPT)
ft, ep_f = load_model(FT_CKPT)
print(f"pretrain epoch {ep_b}, ft epoch {ep_f}")

_ref = xr.open_dataset(f"{NJ}/2020-01-01_hr.nc")
lat, lon = _ref.lat.values, _ref.lon.values
H, W = _ref.sizes["lat"], _ref.sizes["lon"]
coord = make_coord((H, W)).unsqueeze(0).to(dev)
cell = torch.ones_like(coord); cell[:, :, 0] *= 2.0 / H; cell[:, :, 1] *= 2.0 / W

# --- 관측소 좌표 (META, 2020-07-01 시점) + ASOS 2020 원본 (lag1h) ---
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
obs["x"] = obs["지점"].map(lambda s: coords[s][1]); obs["y"] = obs["지점"].map(lambda s: coords[s][0])

for date in dates:
    npz = f"{OUTDIR}/case_studies/3hourly/case_3h_{date}_lag1h_new.npz"
    if os.path.exists(npz):
        print("이미 있음, 건너뜀:", npz); continue
    f = f"{NJ}/{date}_lr.nc"
    if not os.path.exists(f):
        print(f"{date}: 입력 없음, 건너뜀"); continue
    lr_ds, hr_ds = xr.open_dataset(f), xr.open_dataset(f"{NJ}/{date}_hr.nc")
    hL, wL = lr_ds.sizes["lat"], lr_ds.sizes["lon"]
    raw = np.zeros((8, H, W), np.float32); inp = np.zeros((8, hL, wL), np.float32)
    pre = np.zeros((8, H, W), np.float32); prd = np.zeros((8, H, W), np.float32); valid = np.zeros((H, W), bool)
    for k in tqdm(range(24), desc=date):
        b3 = k // 3
        hr = hr_ds.precip_norm.isel(time=k).values; lr = lr_ds.precip_norm.isel(time=k).values
        valid |= np.isfinite(hr)
        raw[b3] += np.nan_to_num(hr) * PMAX; inp[b3] += np.nan_to_num(lr) * PMAX
        x = torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev)
        pre[b3] += predict(base, x, coord, cell, H, W, adapter=False)
        prd[b3] += predict(ft, x, coord, cell, H, W, adapter=False)   # query_rgb 안에서 adapter 이미 적용됨
        torch.cuda.empty_cache()
    o = obs[(obs.t.dt.strftime("%Y-%m-%d") == date) & obs.v.notna()].copy()
    o["b3"] = o.t.dt.hour // 3
    st = o.groupby(["x", "y", "b3"]).v.sum().unstack("b3").reindex(columns=range(8))
    st_lon = st.index.get_level_values("x").values; st_lat = st.index.get_level_values("y").values; st_asos = st.values.T
    np.savez_compressed(npz, raw=raw, inp=inp, pre=pre, prd=prd, valid=valid, lat=lat, lon=lon,
             lr_lat=lr_ds.lat.values, lr_lon=lr_ds.lon.values, st_lat=st_lat, st_lon=st_lon,
             st_asos=st_asos, ep_ft=ep_f, ep_pre=ep_b)
    print("saved", npz)
