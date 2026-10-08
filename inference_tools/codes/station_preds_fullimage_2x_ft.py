"""
2x ft 모델(RUN, epoch-best-loss)을 '전체 영상 입력' 방식으로 지점 위치(최근접 500m 격자)만 질의해 예측을 만든다.
  (infer_hsr_2x_ft_daily_2021.py 가 만드는 nc 에서 지점 최근접 격자 값을 읽는 것과 같은 계산: gen_feat(전체 LR) -> 격자 중심 좌표 query_rgb)
  YEAR=2019 : 학습 2019 JJA (HSR_2019JJA_norm_2x_obscrop.nc) / YEAR=2021 : val.   지점·시각 목록과 obs/hsr_hr 은 기존 station_preds_*.npz 사용(tid = 데이터셋 시간 index).
  PRE=2x : 2x 사전학습 -> 2x ft (..._2x_minmax_liif_ft_mlp_1e-3_lag1h)   PRE=8x : ..._from8x
  HOURS=N : 앞 N 시간만 (속도/검증용)
env: liif (GPU 2)
출력: inference_tools/station_eval/preds_npz/station_preds_fullimage_<YEAR>_<TAG>.npz  (tid,lat,lon,obs,hsr_hr,pf_full)
"""
import os, sys, copy, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"; sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
import numpy as np, torch, xarray as xr, models
from utils import make_coord
YEAR = int(os.environ.get("YEAR", "2019")); PRE = os.environ.get("PRE", "2x"); HOURS = int(os.environ.get("HOURS", "0"))
TAG = "2x_pre2x" if PRE == "2x" else "2x_from8x"
H0 = "/home/inhye_yoo/deeplearning/inr/liif"; M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge"
RUN = f"{H0}/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h" + ("" if PRE == "2x" else "_from8x")
LR_NC, HR_NC = (f"{M}/HSR_2019JJA_norm_2x_obscrop.nc", f"{M}/HSR_2019JJA_500m_norm_obscrop.nc") if YEAR == 2019 else (f"{M}/regrid/HSR_val_norm_2x_obscrop.nc", f"{M}/regrid/HSR_val_500m_norm_obscrop.nc")
REF = f"{H0}/inference_tools/station_eval/preds_npz/" + ("station_preds_train2019_8x_ft_epoch-best-loss.npz" if YEAR == 2019 else "station_preds_val2021_lag1h_mse_new.npz")
OUT = f"{H0}/inference_tools/station_eval/preds_npz/station_preds_fullimage_{YEAR}_{TAG}.npz"
PMAX = 74.8784; dev = torch.device("cuda:0")
ck = torch.load(f"{RUN}/epoch-best-loss.pth", map_location="cpu", weights_only=False); print("epoch", ck["epoch"], flush=True)
spec = copy.deepcopy(ck["model"]); model = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
hr, lr = xr.open_dataset(HR_NC), xr.open_dataset(LR_NC); H, W = hr.sizes["lat"], hr.sizes["lon"]; lat, lon = hr.lat.values, hr.lon.values
coord_all = make_coord((H, W)).to(dev)
ref = np.load(REF); tid, la, lo, obs, hsr = ref["tid"], ref["lat"], ref["lon"], ref["obs"], ref["hsr_hr"]
ii = np.abs(lat[None, :] - la[:, None]).argmin(1); jj = np.abs(lon[None, :] - lo[:, None]).argmin(1)
print("LR", lr.precip_norm.shape, "HR", hr.precip_norm.shape, "rows", len(tid), "시간 수", len(np.unique(tid)), flush=True)
pf = np.full(len(tid), np.nan); hs_chk = []; t0 = time.time(); ut = np.unique(tid)
if HOURS: ut = ut[:HOURS]
with torch.no_grad():
    for n, t in enumerate(ut):
        r = np.where(tid == t)[0]
        x = torch.from_numpy(np.nan_to_num(lr.precip_norm.isel(time=int(t)).values).astype(np.float32))[None, None].to(dev)
        model.gen_feat(x)
        q = coord_all[ii[r] * W + jj[r]][None]; cell = torch.ones_like(q); cell[..., 0] *= 2.0 / H; cell[..., 1] *= 2.0 / W
        pf[r] = (model.query_rgb(q, cell)[0, :, 0].clamp(0, 1) * PMAX).clamp(min=0).cpu().numpy()
        hv = hr.precip_norm.isel(time=int(t)).values[ii[r], jj[r]] * PMAX; hs_chk.append(np.nanmax(np.abs(hv - hsr[r])))
        if n % 200 == 0: print(f"  {n}/{len(ut)} 시간, {time.time()-t0:.0f}초", flush=True)
print("HR nc 지점값 vs ref hsr_hr 최대차:", np.nanmax(hs_chk), " 소요", round(time.time() - t0), "초")
np.savez(OUT, tid=tid, lat=la, lon=lo, obs=obs, hsr_hr=hsr, pf_full=pf); print("saved", OUT)
