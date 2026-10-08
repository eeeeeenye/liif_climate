"""[새 파일] 학습한 체크포인트를 2021 JJA 지점 평가(48x48 크롭 방식, 기존 station_preds_dump_*.py 와 같은 규약)로 돌려 표를 출력/저장.
  - 'lte'(기존 plain 어댑터) / 'lte_ctx'(문맥 어댑터) 체크포인트 모두 지원 (체크포인트에 저장된 모델 spec 으로 생성, models_ctx import).
  - 출력: (1) ASOS>=0.1 / 필터 없음 지표 (r, bias, RMSE, ratio)  (2) 큰 비 구간(ASOS>=10, >=20) ratio  (3) ASOS 구간별 예측평균/ASOS평균  (4) ASOS=0 시각의 예측
  - 기준선(기존 8x LTE wmae ep564) 과 같은 형식으로 나란히 출력해 바로 비교 가능.
사용 (liif 루트에서):
    RUN=save/_ctxC_lte_ctx_wmae_binloss CKPT=epoch-best-loss python exp_ctx_binloss/eval_station_2021.py
    (RUN 기본 = 기존 8x LTE wmae run, 평가 파이프라인 검증용)   CFG=<yaml> 로 val_dataset 설정을 따로 줄 수 있음 (기본: RUN/config.yaml)
출력 npz: exp_ctx_binloss/results/station_preds_val2021_<run이름>_<CKPT>.npz  (tid, lat, lon, obs, pf, hsr_hr : mm/h)
env: liif, 물리 GPU 2
"""
import sys, os, copy, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np, torch, yaml, xarray as xr
import datasets, models
import models_ctx  # noqa: F401  (lte_ctx 등록)

device = torch.device("cuda:0")
RUN = os.environ.get("RUN", f"{ROOT}/save/_climate_edsr_radar_hsr_8x_minmax_lte_ft_mlp_1e-3_lag1h_2019_2020_wmae")
RUN = RUN if os.path.isabs(RUN) else os.path.join(ROOT, RUN)
CKPT_TAG = os.environ.get("CKPT", "epoch-best-loss")
CFG = os.environ.get("CFG", f"{RUN}/config.yaml")
META = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/meta_HSR_500m_2019JJA_obscrop_lag1h_val.json"
HR_NC = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid/HSR_val_500m_norm_obscrop.nc"
BASE_NPZ = f"{ROOT}/inference_tools/station_eval/preds_npz/station_preds_val2021_8x_lte_ft_2019_2020_wmae_ep564_epoch-best.npz"
OUT = f"{ROOT}/exp_ctx_binloss/results/station_preds_val2021_{os.path.basename(RUN.rstrip('/'))}_{CKPT_TAG}.npz"
PMAX, CROP = 74.8784, 48

config = yaml.load(open(CFG), Loader=yaml.FullLoader); meta = json.load(open(META))
lat_max, lat_min = meta['lat_range']; lon_min, lon_max = meta['lon_range']
val_ds = datasets.make(config['val_dataset']['dataset'])
sv = torch.load(f"{RUN}/{CKPT_TAG}.pth", map_location='cuda:0', weights_only=False); spec = copy.deepcopy(sv['model'])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(device)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
print(f"{os.path.basename(RUN)} / {CKPT_TAG} (epoch {sv.get('epoch')}, model={spec['name']})", flush=True)

g = xr.open_dataset(HR_NC); NC_LAT, NC_LON = g.lat.values, g.lon.values
crops, coords, gts, lats, lons, tids, hsr = [], [], [], [], [], [], []
for idx in range(len(val_ds)):
    img_lr, img_hr, obs_, _ = val_ds[idx]; H_lr, W_lr = img_lr.shape[-2:]; v = obs_['gt']
    if v.ndim == 1: v = v.unsqueeze(-1)
    mask = v.squeeze(-1) != -2.0
    if mask.sum() == 0: continue
    c = obs_['coord'][mask].clone(); v = v[mask]; c[:, 0] = -c[:, 0]
    for k in range(c.shape[0]):
        y_, x_ = float(c[k, 0]), float(c[k, 1])
        lats.append(lat_min + (y_ + 1) / 2 * (lat_max - lat_min)); lons.append(lon_min + (x_ + 1) / 2 * (lon_max - lon_min))
        ia = (y_ + 1) / 2 * (H_lr - 1); ja = (x_ + 1) / 2 * (W_lr - 1)
        x0 = min(max(int(round(ia - CROP / 2)), 0), H_lr - CROP); y0 = min(max(int(round(ja - CROP / 2)), 0), W_lr - CROP)
        crops.append(img_lr[:, x0:x0 + CROP, y0:y0 + CROP])
        coords.append(torch.tensor([((ia - x0) / (CROP - 1)) * 2 - 1, ((ja - y0) / (CROP - 1)) * 2 - 1], dtype=torch.float32))
        gts.append(float(v[k, 0])); tids.append(idx)
        hsr.append(float(img_hr[0, int(np.abs(NC_LAT - lats[-1]).argmin()), int(np.abs(NC_LON - lons[-1]).argmin())]))
N = len(crops); print("(시각,관측소) 쌍:", N, flush=True)
crops_t = torch.stack(crops); coords_t = torch.stack(coords).unsqueeze(1); out = []
with torch.no_grad():
    for s in range(0, N, 512):
        inp = crops_t[s:s + 512].to(device); co = coords_t[s:s + 512].to(device)
        cell = torch.ones_like(co); cell[:, :, 0] *= 2 / 1000; cell[:, :, 1] *= 2 / 1000
        out.append(model(inp, None, co, cell).clamp(0, 1).squeeze(-1).squeeze(-1).cpu())
pf = torch.cat(out).numpy() * PMAX; ob = np.array(gts) * PMAX; hs = np.array(hsr) * PMAX
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, tid=np.array(tids), lat=np.array(lats), lon=np.array(lons), obs=ob, pf=pf, hsr_hr=hs); print("npz 저장:", OUT)

BINS = [(0, 0.001, "=0"), (0.001, 0.5, "0~.5"), (0.5, 1, ".5~1"), (1, 2, "1~2"), (2, 5, "2~5"), (5, 10, "5~10"), (10, 999, "10+")]
def report(name, a, p):
    m1 = a >= 0.1; r = []
    for flt, mm in (("ASOS>=0.1", m1), ("필터 없음", np.ones_like(m1))):
        x, y = p[mm], a[mm]; r.append("%s r=%.3f bias=%+.2f RMSE=%.2f ratio=%.3f" % (flt, np.corrcoef(x, y)[0, 1], (x - y).mean(), np.sqrt(((x - y) ** 2).mean()), x.mean() / y.mean()))
    h10, h20 = a >= 10, a >= 20
    print("\n[%s]\n  %s\n  %s\n  큰 비 ratio(예측평균/ASOS평균): ASOS>=10 %.2f (n=%d) | ASOS>=20 %.2f (n=%d)" % (name, r[0], r[1], p[h10].mean() / a[h10].mean(), h10.sum(), p[h20].mean() / a[h20].mean(), h20.sum()))
    row = ["%s:%s" % (nm, ("%.2fmm" % p[(a >= lo) & (a < hi)].mean()) if lo == 0 else "%.2f" % (p[(a >= lo) & (a < hi)].mean() / a[(a >= lo) & (a < hi)].mean())) for lo, hi, nm in BINS]
    print("  ASOS 구간별 예측평균/ASOS평균 ( =0 은 예측 평균 mm/h ):", "  ".join(row))
    z = a == 0; print("  ASOS=0 시각: 예측>=0.1 비율 %.1f%%, 예측 평균 %.2f mm/h" % (100 * (p[z] >= 0.1).mean(), p[z].mean()))
print("\n===== 2021 JJA 지점 평가 (48x48 크롭 방식) =====")
if os.path.exists(BASE_NPZ):
    d = np.load(BASE_NPZ); assert np.abs(d["obs"] - ob).max() < 1e-3, "기준선과 (지점,시각) 순서 불일치"
    report("기준선: 기존 8x LTE wmae ep564 (저장된 npz)", d["obs"], d["pf"].astype(float))
report("HSR 원본", ob, hs)
report(f"이번 모델: {os.path.basename(RUN)} / {CKPT_TAG} (ep {sv.get('epoch')})", ob, pf)
