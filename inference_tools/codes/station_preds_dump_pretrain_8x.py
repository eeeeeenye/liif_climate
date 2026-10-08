"""
사전학습 8x LIIF (_climate_edsr_radar_hsr_8x_minmax_liif, 어댑터 없음, 2019~2020 JJA 로 학습) 를
2019 JJA (학습 구간) / 2021 JJA (검증 구간) 관측소 지점에서 그대로 추론해 HSR 500m 와 비교용으로 덤프한다.
station_preds_dump_{train2019,val2021}_8x_ft.py 와 동일한 지점/시각/crop 처리, 모델만 사전학습 체크포인트.
사용:  python station_preds_dump_pretrain_8x.py            # CKPT=epoch-last (기본)
출력:  station_eval/preds_npz/station_preds_pretrain_8x_{train2019,val2021}_<CKPT>.npz  (tid, lat, lon, obs, pf, hsr_hr : mm/h)
env: liif, 물리 GPU 2
"""
import sys, os, copy, json
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import torch
import yaml
import xarray as xr
import datasets, models

device = torch.device("cuda:0")
RUN = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_8x_minmax_liif"
CKPT_TAG = os.environ.get("CKPT", "epoch-last")
CFG = "/home/inhye_yoo/deeplearning/inr/liif/configs/train_climate/climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h.yaml"
OBS = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/"
HR = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/"
OUTD = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/station_eval/preds_npz/"
PMAX = 74.8784
CROP = 48
JOBS = {"train2019": ("train_dataset", OBS + "meta_HSR_500m_2019JJA_obscrop_lag1h_train.json", HR + "HSR_2019JJA_500m_norm_obscrop.nc"),
        "val2021": ("val_dataset", OBS + "meta_HSR_500m_2019JJA_obscrop_lag1h_val.json", HR + "regrid/HSR_val_500m_norm_obscrop.nc")}

config = yaml.load(open(CFG), Loader=yaml.FullLoader)
sv = torch.load(f"{RUN}/{CKPT_TAG}.pth", map_location="cuda:0", weights_only=False)
spec = copy.deepcopy(sv["model"])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(device)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
print(f"pretrain {CKPT_TAG} (epoch {sv.get('epoch')}) add_layer={hasattr(model, 'add_layer')}", flush=True)

for tag, (key, META, HR_NC) in JOBS.items():
    meta = json.load(open(META)); lat_max, lat_min = meta["lat_range"]; lon_min, lon_max = meta["lon_range"]
    val_ds = datasets.make(config[key]["dataset"])
    _g = xr.open_dataset(HR_NC); NC_LAT = _g.lat.values; NC_LON = _g.lon.values
    crops, coords, gts, lats, lons, tids, hsr_hr = [], [], [], [], [], [], []
    for idx in range(len(val_ds)):
        img_lr, img_hr, obs_, _ = val_ds[idx]
        H_lr, W_lr = img_lr.shape[-2:]
        v = obs_["gt"]
        if v.ndim == 1: v = v.unsqueeze(-1)
        mask = v.squeeze(-1) != -2.0
        if mask.sum() == 0: continue
        c = obs_["coord"][mask].clone(); v = v[mask]
        c[:, 0] = -c[:, 0]
        for k in range(c.shape[0]):
            y_, x_ = float(c[k, 0]), float(c[k, 1])
            lats.append(lat_min + (y_ + 1) / 2 * (lat_max - lat_min))
            lons.append(lon_min + (x_ + 1) / 2 * (lon_max - lon_min))
            ia = (y_ + 1) / 2 * (H_lr - 1); ja = (x_ + 1) / 2 * (W_lr - 1)
            x0 = min(max(int(round(ia - CROP / 2)), 0), H_lr - CROP)
            y0 = min(max(int(round(ja - CROP / 2)), 0), W_lr - CROP)
            crops.append(img_lr[:, x0:x0 + CROP, y0:y0 + CROP])
            coords.append(torch.tensor([((ia - x0) / (CROP - 1)) * 2 - 1, ((ja - y0) / (CROP - 1)) * 2 - 1], dtype=torch.float32))
            gts.append(float(v[k, 0])); tids.append(idx)
            r_ = int(np.abs(NC_LAT - lats[-1]).argmin()); c_ = int(np.abs(NC_LON - lons[-1]).argmin())
            hsr_hr.append(float(img_hr[0, r_, c_]))
    N = len(crops); print(tag, "(시각,관측소) 쌍:", N, flush=True)
    crops_t = torch.stack(crops); coords_t = torch.stack(coords).unsqueeze(1); out = []
    with torch.no_grad():
        for s_ in range(0, N, 512):
            inp = crops_t[s_:s_ + 512].to(device); co = coords_t[s_:s_ + 512].to(device)
            cell = torch.ones_like(co); cell[:, :, 0] *= 2 / 1000; cell[:, :, 1] *= 2 / 1000
            out.append(model(inp, None, co, cell).squeeze(-1).squeeze(-1).cpu())
    pf = torch.cat(out).numpy() * PMAX
    np.savez(f"{OUTD}station_preds_pretrain_8x_{tag}_{CKPT_TAG}.npz", tid=np.array(tids), lat=np.array(lats), lon=np.array(lons),
             obs=np.array(gts) * PMAX, pf=pf, hsr_hr=np.array(hsr_hr) * PMAX)
    print(tag, "saved", flush=True)
