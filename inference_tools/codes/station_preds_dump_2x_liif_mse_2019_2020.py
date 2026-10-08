"""
2x LIIF (1 km 입력) ASOS 미세조정 L2-MSE-1920 (_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020, epoch-best-loss) 의
지점 예측을 2021 JJA(검증) / 2019+2020 JJA(학습) 에서 덤프한다. station_preds_dump_val2021_wmae_plainadapter.py 와 같은 지점/crop 처리.
  pb = 2x 사전학습(어댑터 없음, _climate_edsr_radar_hsr_2x_minmax_liif/epoch-last), pf = 미세조정 모델 최종 출력  (mm/h, clamp(0,1))
사용: python station_preds_dump_2x_liif_mse_2019_2020.py
출력: station_eval/preds_npz/station_preds_{val2021,train2019_2020}_2x_ft_2019_2020_epoch-best-loss.npz  (tid, lat, lon, obs, pb, pf, hsr_hr)
env: liif, 물리 GPU 2
"""
import sys, os, copy, json
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif"); os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np, torch, yaml, xarray as xr, datasets, models
device = torch.device("cuda:0")
L = "/home/inhye_yoo/deeplearning/inr/liif"; O = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/"; N_ = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/"
CFG = f"{L}/configs/train_climate/climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020.yaml"
BASE_CKPT = f"{L}/save/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth"
FT_CKPT = f"{L}/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020/epoch-best-loss.pth"
OUTD = f"{L}/inference_tools/station_eval/preds_npz/"
JOBS = {"val2021": ("val_dataset", O + "meta_HSR_500m_2019JJA_obscrop_lag1h_val.json", N_ + "regrid/HSR_val_500m_norm_obscrop.nc"),
        "train2019_2020": ("train_dataset", O + "meta_HSR_500m_2019_2020JJA_obscrop_lag1h_train.json", N_ + "HSR_2019_2020JJA_500m_norm_obscrop.nc")}
PMAX = 74.8784; CROP = 48
config = yaml.load(open(CFG), Loader=yaml.FullLoader)
def load_model(path):
    sv = torch.load(path, map_location='cuda:0', weights_only=False); spec = copy.deepcopy(sv['model'])
    m = models.make({"name": spec["name"], "args": spec["args"]}).to(device); m.load_state_dict(spec["sd"], strict=True); m.eval(); m.topo = None; return m, sv.get("epoch")
base, _ = load_model(BASE_CKPT); ft, ep = load_model(FT_CKPT); print("ft epoch:", ep, flush=True)
for tag, (key, META, HR_NC) in JOBS.items():
    meta = json.load(open(META)); lat_max, lat_min = meta['lat_range']; lon_min, lon_max = meta['lon_range']
    ds = datasets.make(config[key]['dataset']); _g = xr.open_dataset(HR_NC); NC_LAT = _g.lat.values; NC_LON = _g.lon.values
    crops, coords, gts, lats, lons, tids, hsr_hr = [], [], [], [], [], [], []
    for idx in range(len(ds)):
        img_lr, img_hr, obs_, _ = ds[idx]; H_lr, W_lr = img_lr.shape[-2:]; v = obs_['gt']
        if v.ndim == 1: v = v.unsqueeze(-1)
        mask = v.squeeze(-1) != -2.0
        if mask.sum() == 0: continue
        c = obs_['coord'][mask].clone(); v = v[mask]; c[:, 0] = -c[:, 0]
        for k in range(c.shape[0]):
            y_, x_ = float(c[k, 0]), float(c[k, 1])
            lats.append(lat_min + (y_ + 1) / 2 * (lat_max - lat_min)); lons.append(lon_min + (x_ + 1) / 2 * (lon_max - lon_min))
            ia = (y_ + 1) / 2 * (H_lr - 1); ja = (x_ + 1) / 2 * (W_lr - 1)
            x0 = min(max(int(round(ia - CROP / 2)), 0), H_lr - CROP); y0 = min(max(int(round(ja - CROP / 2)), 0), W_lr - CROP)
            crops.append(img_lr[:, x0:x0 + CROP, y0:y0 + CROP]); coords.append(torch.tensor([((ia - x0) / (CROP - 1)) * 2 - 1, ((ja - y0) / (CROP - 1)) * 2 - 1], dtype=torch.float32))
            gts.append(float(v[k, 0])); tids.append(idx)
            r_ = int(np.abs(NC_LAT - lats[-1]).argmin()); c_ = int(np.abs(NC_LON - lons[-1]).argmin()); hsr_hr.append(float(img_hr[0, r_, c_]))
    N = len(crops); print(tag, "(시각,관측소) 쌍:", N, flush=True); crops_t = torch.stack(crops); coords_t = torch.stack(coords).unsqueeze(1)
    def run(model):
        out = []
        with torch.no_grad():
            for s in range(0, N, 512):
                inp = crops_t[s:s + 512].to(device); co = coords_t[s:s + 512].to(device)
                cell = torch.ones_like(co); cell[:, :, 0] *= 2 / 1000; cell[:, :, 1] *= 2 / 1000
                out.append(model(inp, None, co, cell).clamp(0, 1).squeeze(-1).squeeze(-1).cpu())
        return torch.cat(out).numpy() * PMAX
    np.savez(f"{OUTD}station_preds_{tag}_2x_ft_2019_2020_epoch-best-loss.npz", tid=np.array(tids), lat=np.array(lats), lon=np.array(lons), obs=np.array(gts) * PMAX, pb=run(base), pf=run(ft), hsr_hr=np.array(hsr_hr) * PMAX)
    print(tag, "saved", flush=True)
