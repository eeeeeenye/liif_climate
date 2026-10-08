"""
2x 사전학습(동결 imnet) 의 방향별 p_k(4) 와 local-ensemble 가중치 w_k(4)를 '전체 영상 입력' 방식으로 추출 (지점 최근접 500m 격자 중심 좌표 질의).
  (infer_hsr_2x_ft_daily_2021.py 의 nc 와 같은 계산: gen_feat(전체 LR 영상) -> 격자 중심 좌표 query. collect_pk_weights_2x.py 는 지점 주변 48x48 크롭 방식)
  YEAR=2021(val) | 2019(train 2019 JJA).  지점·시각 목록/obs/hsr_hr 은 기존 station_preds_*.npz 의 순서를 따름 (tid = 데이터셋 시간 index).
사용: YEAR=2021 python collect_pk_weights_fullimage_2x.py     (HOURS=N: 앞 N 시간만. 시간당 약 1.8초, 2021 1729시간 ~52분, 2019 1610시간 ~48분; GPU 공유 시 더 걸림)
출력: station_eval/preds_npz/pk_w_fullimage_<YEAR>_2x_pre2x.npz   (P[N,4] mm/h, W[N,4], obs, hsr_hr, tid, lat, lon)
env: liif (GPU 2)
"""
import os, sys, copy, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"; sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
import numpy as np, torch, xarray as xr, models
from utils import make_coord
YEAR = int(os.environ.get("YEAR", "2021")); HOURS = int(os.environ.get("HOURS", "0"))
H0 = "/home/inhye_yoo/deeplearning/inr/liif"; M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge"; E = f"{H0}/inference_tools/station_eval/preds_npz"
LR_NC, HR_NC = (f"{M}/HSR_2019JJA_norm_2x_obscrop.nc", f"{M}/HSR_2019JJA_500m_norm_obscrop.nc") if YEAR == 2019 else (f"{M}/regrid/HSR_val_norm_2x_obscrop.nc", f"{M}/regrid/HSR_val_500m_norm_obscrop.nc")
REF = f"{E}/" + ("station_preds_train2019_8x_ft_epoch-best-loss.npz" if YEAR == 2019 else "station_preds_val2021_lag1h_mse_new.npz")
OUT = f"{E}/pk_w_fullimage_{YEAR}_2x_pre2x.npz"; PMAX = 74.8784; dev = torch.device("cuda:0")
sv = torch.load(f"{H0}/save/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth", map_location="cpu", weights_only=False)
spec = copy.deepcopy(sv["model"]); model = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
store = []; model.imnet.register_forward_hook(lambda m, i, o: store.append(o.detach().float().cpu()))
hr, lr = xr.open_dataset(HR_NC), xr.open_dataset(LR_NC); H, W = hr.sizes["lat"], hr.sizes["lon"]; lat, lon = hr.lat.values, hr.lon.values
coord_all = make_coord((H, W)).to(dev)
ref = np.load(REF); tid, la, lo, obs, hsr = ref["tid"], ref["lat"], ref["lon"], ref["obs"], ref["hsr_hr"]
ii = np.abs(lat[None, :] - la[:, None]).argmin(1); jj = np.abs(lon[None, :] - lo[:, None]).argmin(1)
P = np.full((len(tid), 4), np.nan); Wt = np.full((len(tid), 4), np.nan); ut = np.unique(tid)
if HOURS: ut = ut[:HOURS]
print(YEAR, "rows", len(tid), "시간", len(ut), flush=True); t0 = time.time()
with torch.no_grad():
    for n, t in enumerate(ut):
        r = np.where(tid == t)[0]
        x = torch.from_numpy(np.nan_to_num(lr.precip_norm.isel(time=int(t)).values).astype(np.float32))[None, None].to(dev)
        model.gen_feat(x)
        q = coord_all[ii[r] * W + jj[r]][None]; cell = torch.ones_like(q); cell[..., 0] *= 2.0 / H; cell[..., 1] *= 2.0 / W
        store.clear(); model.query_rgb(q, cell); assert len(store) == 4
        P[r] = torch.stack([s.reshape(-1) for s in store], 1).numpy() * PMAX
        Wt[r] = torch.stack([w.reshape(-1).cpu() for w in model.le_weights], 1).numpy()
        if n % 100 == 0: print(f"  {n}/{len(ut)} 시간, {time.time()-t0:.0f}초", flush=True)
print("소요", round(time.time() - t0), "초")
np.savez(OUT, P=P, W=Wt, obs=obs, hsr_hr=hsr, tid=tid, lat=la, lon=lo); print("saved", OUT, flush=True)
