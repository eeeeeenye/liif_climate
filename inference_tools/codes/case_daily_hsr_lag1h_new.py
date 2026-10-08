"""
집중호우 case 일누적 지도 (val 2021 JJA, KST 하루 24시간):
  열 = Raw(HSR 500m) | Input(2x, 1km LR) | Radar pretrain 예측 | MLP ft(MSE, lag1h) 예측   (전부 500m 격자, 일합 mm/day)
  행 = case 날짜.  ASOS 일합(원)을 같은 색으로 덧그림.

시간 정렬 (1h lag):
  ASOS 매시값[T] 는 직전 1시간 (T-1h, T] 누적, HSR[H] = [H, H+1) -> ASOS[T] <-> HSR[T-1h].
  lag1h obs csv 는 이미 라벨을 T-1h 로 옮겨 놓았으므로, KST 하루 D 의 라벨 00~23시 합 = ASOS 달력일합 = HSR 00~23시 합 과 같은 시간창.

추론: model.gen_feat(전체 LR) 한 번 -> HR 격자(1326x1234) 를 청크로 query_rgb (cell=2/H,2/W; infer_hsr_radar_2x_sr_2021_08_23.py 와 동일).
      ft 는 blend 후 correction_layer(adapter) 를 적용 (forward 와 동일).

사용:  python case_daily_hsr_lag1h.py 2021-07-06 2021-08-23
출력:  inference_tools/case_studies/daily/case_daily_<날짜들>_lag1h.png,  inference_tools/case_studies/daily/case_daily_<날짜>_lag1h_new.npz (일합 배열)
env: liif, 물리 GPU 2 (CUDA_VISIBLE_DEVICES=2 -> cuda:0)
"""
import os, sys, copy
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm
import models
from utils import make_coord

dev = torch.device("cuda:0")
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid"
LR_NC = f"{M}/HSR_val_norm_2x_obscrop.nc"
HR_NC = f"{M}/HSR_val_500m_norm_obscrop.nc"
OBS = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv"
S = "/home/inhye_yoo/deeplearning/inr/liif/save"
BASE_CKPT = f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth"
FT_CKPT = os.environ.get("FT_CKPT", "/tmp/claude-1018/-home-inhye-yoo/80214217-fd09-4b86-ac2a-f405eaea02bd/scratchpad/ft_mse_new_best_loss.pth")
OUTDIR = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
PMAX = 74.8784
BSIZE = 500_000
dates = sys.argv[1:] or ["2021-07-06", "2021-08-23"]


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
lr_ds, hr_ds = xr.open_dataset(LR_NC), xr.open_dataset(HR_NC)
times = pd.to_datetime(hr_ds.time.values)
assert np.array_equal(times, pd.to_datetime(lr_ds.time.values))
H, W = hr_ds.sizes["lat"], hr_ds.sizes["lon"]
coord = make_coord((H, W)).unsqueeze(0).to(dev)
cell = torch.ones_like(coord); cell[:, :, 0] *= 2.0 / H; cell[:, :, 1] *= 2.0 / W
lat, lon = hr_ds.lat.values, hr_ds.lon.values

obs = pd.read_csv(OBS, encoding="utf-8-sig"); obs["t"] = pd.to_datetime(obs["time"])

for date in dates:
    npz = f"{OUTDIR}/case_studies/daily/case_daily_{date}_lag1h_new.npz"
    if os.path.exists(npz):
        print("이미 있음, 건너뜀:", npz); continue
    tidx = np.where(times.strftime("%Y-%m-%d") == date)[0]
    assert len(tidx) == 24, (date, len(tidx))
    raw = np.zeros((H, W)); inp = np.zeros((lr_ds.sizes["lat"], lr_ds.sizes["lon"]))
    pre = np.zeros((H, W)); prd = np.zeros((H, W)); valid = np.zeros((H, W), bool)
    for t in tqdm(tidx, desc=date):
        hr = hr_ds.precip_norm.isel(time=int(t)).values; lr = lr_ds.precip_norm.isel(time=int(t)).values
        valid |= np.isfinite(hr)
        raw += np.nan_to_num(hr) * PMAX; inp += np.nan_to_num(lr) * PMAX
        x = torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev)
        pre += predict(base, x, coord, cell, H, W, adapter=False)
        prd += predict(ft, x, coord, cell, H, W, adapter=False)   # 재학습 모델: adapter 는 query_rgb 안에서 방향별 1회 적용됨 -> 뒤에 또 부르면 2회
        torch.cuda.empty_cache()
    o = obs[(obs.t.dt.strftime("%Y-%m-%d") == date) & obs.v.notna()]
    st = o.groupby(["x", "y"]).v.sum().reset_index().rename(columns={"x": "lon", "y": "lat", "v": "asos"})
    np.savez(npz, raw=raw, inp=inp, pre=pre, prd=prd, valid=valid, lat=lat, lon=lon,
             lr_lat=lr_ds.lat.values, lr_lon=lr_ds.lon.values, st_lat=st.lat.values, st_lon=st.lon.values,
             st_asos=st.asos.values, ep_ft=ep_f, ep_pre=ep_b)
    print("saved", npz)
