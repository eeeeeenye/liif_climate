"""
2019 JJA 에서 'HSR>=20, pred<=0.6xHSR, ASOS>=0.8xHSR' 인 4점 (case_big_drop_*.csv) 의 입력을 그린다.
  열 1: 8x LR 입력 (~4 km 격자, 모델이 보는 입력), 지점 주변 +-8 LR 픽셀 (= +-64 HR 픽셀 ~ +-32 km)
  열 2: 같은 범위의 원래 HSR 500m
  열 3: 지점 주변 500m 확대 (+-16 HR 픽셀 ~ +-8 km)
  모든 패널은 같은 색 눈금(mm/h). 지점 = 빨간 마커. LR 열에는 imnet 방향별 p_k 와 가중치도 표기.
입력: station_preds_train2019_8x_ft_epoch-best-loss.npz, pk_w_cache_8x_train2019.npz, train_dataset(config) 의 LR/HR nc
출력: station_eval/figures/hexbin_hsr_preds_vs_asos/case_big_drop_inputs_2019.png
env: liif (CPU)
"""
import os, sys
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd, yaml, xarray as xr, torch, datasets
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False})
V = 74.8784; H = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/"
cfg = yaml.load(open("/home/inhye_yoo/deeplearning/inr/liif/configs/train_climate/climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h.yaml"), Loader=yaml.FullLoader)
meta = __import__("json").load(open("/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/meta_HSR_500m_2019JJA_obscrop_lag1h_train.json")); lat_max, lat_min = meta["lat_range"]; lon_min, lon_max = meta["lon_range"]
HRNC = xr.open_dataset("/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/HSR_2019JJA_500m_norm_obscrop.nc"); NLAT, NLON = HRNC.lat.values, HRNC.lon.values
ds = datasets.make(cfg["train_dataset"]["dataset"])
d = np.load(H + "station_eval/preds_npz/station_preds_train2019_8x_ft_epoch-best-loss.npz"); pk = np.load(H + "station_eval/preds_npz/pk_w_cache_8x_train2019.npz")
cases = pd.read_csv("/home/inhye_yoo/slide_csv/case_big_drop_hsr20_pred_le_0.6hsr_asos_ge_0.8hsr_2019.csv")
fig, axes = plt.subplots(len(cases), 3, figsize=(15, 4.9 * len(cases)))
for ri, c in cases.iterrows():
    row = int(np.where((d["tid"] == c.tid) & (abs(d["lat"] - c.lat) < 0.01) & (abs(d["lon"] - c.lon) < 0.01))[0][0])
    lat0, lon0 = float(d["lat"][row]), float(d["lon"][row]); img_lr, img_hr, _, _ = ds[int(c.tid)]; lr = img_lr[0].numpy() * V; hr = img_hr[0].numpy() * V
    y_ = 2 * (lat0 - lat_min) / (lat_max - lat_min) - 1; x_ = 2 * (lon0 - lon_min) / (lon_max - lon_min) - 1
    ia = (y_ + 1) / 2 * (lr.shape[0] - 1); ja = (x_ + 1) / 2 * (lr.shape[1] - 1)
    r_ = int(np.abs(NLAT - lat0).argmin()); c_ = int(np.abs(NLON - lon0).argmin()); i0, j0 = int(round(ia)), int(round(ja))
    R = 8; sl = (slice(max(i0 - R, 0), i0 + R + 1), slice(max(j0 - R, 0), j0 + R + 1)); hs = (slice(r_ - 64, r_ + 65), slice(c_ - 64, c_ + 65)); hz = (slice(r_ - 16, r_ + 17), slice(c_ - 16, c_ + 17))
    vmax = max(35, np.nanmax(hr[hs]), np.nanmax(lr[sl]))
    ext_lr = (sl[1].start - 0.5, sl[1].stop - 0.5, sl[0].stop - 0.5, sl[0].start - 0.5)
    P = pk["P"][row]; W = pk["W"][row]
    a = axes[ri, 0]; im = a.imshow(lr[sl], vmin=0, vmax=vmax, cmap="Blues", extent=ext_lr); a.plot(ja, ia, "r+", ms=16, mew=2)
    a.set_title(f"8x LR 입력 (~4 km)  LR 최근접 픽셀 {lr[i0, j0]:.1f}", fontsize=10, loc="left")
    a.text(0.02, 0.02, "imnet p_k × 가중치\n" + "  ".join(f"{p:.1f}×{w:.2f}" for p, w in zip(P, W)) + f"\n→ 가중평균 {float((P * W).sum()):.1f}", transform=a.transAxes, fontsize=8, va="bottom", bbox=dict(fc="white", alpha=0.8, ec="none"))
    b = axes[ri, 1]; b.imshow(hr[hs], vmin=0, vmax=vmax, cmap="Blues", extent=(c_ - 64.5, c_ + 64.5, r_ + 64.5, r_ - 64.5)); b.plot(c_, r_, "r+", ms=16, mew=2)
    b.set_title(f"원래 HSR 500 m (같은 범위 ±32 km)  지점값 {hr[r_, c_]:.1f}", fontsize=10, loc="left")
    z = axes[ri, 2]; z.imshow(hr[hz], vmin=0, vmax=vmax, cmap="Blues", extent=(c_ - 16.5, c_ + 16.5, r_ + 16.5, r_ - 16.5)); z.plot(c_, r_, "r+", ms=16, mew=2)
    z.set_title(f"500 m 확대 (±8 km)   최대 {np.nanmax(hr[hz]):.1f}", fontsize=10, loc="left")
    axes[ri, 0].set_ylabel(f"tid {int(c.tid)}  {c.asos_time_KST}\n({c.lat:.2f}N, {c.lon:.2f}E)\nHSR {c.hsr:.1f} → imnet {c.imnet:.1f} → pred {c.pred:.1f}\nASOS {c.asos:.1f}", fontsize=10)
    for a_ in (a, b, z): a_.set_xticks([]); a_.set_yticks([])
    fig.colorbar(im, ax=list(axes[ri]), fraction=0.012, pad=0.01, label="mm/h")
fig.suptitle("2019 JJA · HSR≥20 인데 pred 가 0.6배 이하로 내려가고 ASOS 는 HSR 수준이던 4점의 입력 (+ = 지점)", fontsize=13, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 0.98, 0.985)); out = H + "station_eval/figures/hexbin_hsr_preds_vs_asos/case_big_drop_inputs_2019.png"; fig.savefig(out, dpi=110); print("saved", out)
