"""
[DEPRECATED] slope/relief 채널 기능이 TopoInMemory에서 제거됨(효과 없음).
옛 3채널 체크포인트를 elevation-only 모델에 strict=False로 부분 로드하므로
결과가 무의미하다. 기록용으로만 남겨둠.

climate_edsr_topo_80x100_topo_slope_relief_ver 실험의 epoch-last.pth 로
2019-07-20 (train/val 기간 밖, out-of-sample) 하루치 예측을 뽑고 cartopy 지도로 저장.

- topo 채널: [elevation, slope, relief]  (add_slope / add_relief = true)
- adapter 미사용 (layer_spec 제거)
- 정규화: minmax, mm/day = value * 506.01996  (min = 0)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # 무조건 GPU 2번 사용

import sys
import copy
from collections import OrderedDict

import numpy as np
import torch
import yaml
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.append("/home/inhye_yoo/deeplearning/inr/liif")
from models.models import make as make_model
from datasets.image_folder import TopoInMemory
from utils import make_coord

# ------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------
RUN_DIR  = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_slope_relief_ver"
CKPT     = os.path.join(RUN_DIR, "epoch-last.pth")
CFG_PATH = os.path.join(RUN_DIR, "config.yaml")
OUT_DIR  = os.path.join(RUN_DIR, "preds")
os.makedirs(OUT_DIR, exist_ok=True)

DATE = "2019-07-20"
LR_RAW = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_02deg_1998_202504_noleaf.nc"
HR_RAW = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_01deg_1998_202504_noleaf.nc"

DEVICE   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA_VISIBLE_DEVICES=2 -> cuda:0 == 물리 GPU 2
TOPO_LAT = (33, 43)
TOPO_LON = (124, 132)          # PairedImageFolders 와 동일

# ------------------------------------------------------------------
# config
# ------------------------------------------------------------------
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

topo_path = cfg["val_dataset"]["dataset"]["args"]["topo_nc_path"]
topo_kw   = cfg["val_dataset"]["dataset"]["args"].get("topo_kwargs", {})
pn        = cfg["physical_norm"]
VMAX, VMIN = float(pn["max"]), float(pn["min"])          # 506.01996, 0.0

def to_mm(x):
    return x * (VMAX - VMIN) + VMIN

# ------------------------------------------------------------------
# 데이터 (raw mm/day -> minmax 정규화)
# ------------------------------------------------------------------
lr_da = xr.open_dataset(LR_RAW)["precipitation"].sel(time=DATE)
hr_da = xr.open_dataset(HR_RAW)["precipitation"].sel(time=DATE)

lat_hr, lon_hr = hr_da["lat"].values, hr_da["lon"].values
lat_lr, lon_lr = lr_da["lat"].values, lr_da["lon"].values
H, W = hr_da.shape[-2:]                                  # 100, 80

lr_mm = np.nan_to_num(lr_da.values).astype(np.float32)
gt_mm = np.nan_to_num(hr_da.values).astype(np.float32)
lr_norm = lr_mm / VMAX                                   # 학습 입력 스케일

d_hr = float(lat_hr[1] - lat_hr[0]); d_lr = float(lat_lr[1] - lat_lr[0])
ext_hr = [lon_hr[0]-d_hr/2, lon_hr[-1]+d_hr/2, lat_hr[0]-d_hr/2, lat_hr[-1]+d_hr/2]
ext_lr = [lon_lr[0]-d_lr/2, lon_lr[-1]+d_lr/2, lat_lr[0]-d_lr/2, lat_lr[-1]+d_lr/2]

# ------------------------------------------------------------------
# topo (elevation + slope + relief)
# ------------------------------------------------------------------
topo_ds = TopoInMemory(path_1=topo_path, lat_range=TOPO_LAT, lon_range=TOPO_LON, **topo_kw)
topo = topo_ds[0].unsqueeze(0).to(DEVICE)               # (1, C, Ht, Wt)
print("topo tensor:", tuple(topo.shape))

# ------------------------------------------------------------------
# 모델 로드
# ------------------------------------------------------------------
ckpt = torch.load(CKPT, map_location="cpu")
model = make_model({"name": "liif", "args": copy.deepcopy(ckpt["model"]["args"])}).to(DEVICE)
sd, own = ckpt["model"]["sd"], model.state_dict()
compat = OrderedDict((k, v) for k, v in sd.items() if k in own and v.shape == own[k].shape)
model.load_state_dict(compat, strict=False)
model.eval()
print(f"ckpt epoch={ckpt.get('epoch')}  loaded {len(compat)}/{len(own)} tensors")

coord = make_coord((H, W)).unsqueeze(0).to(DEVICE)
cell = torch.ones_like(coord)
cell[..., 0] *= 2.0 / H
cell[..., 1] *= 2.0 / W

with torch.no_grad():
    inp = torch.from_numpy(lr_norm)[None, None].to(DEVICE)   # (1,1,50,40)
    model.gen_feat(inp)
    model.topo = topo
    pred_norm = model.query_rgb(coord, cell).view(H, W).clamp_(0.0, None).cpu().numpy()

pred_mm = to_mm(pred_norm)
err = pred_mm - gt_mm
rmse = float(np.sqrt(np.mean(err ** 2)))
bias = float(np.mean(err))
print(f"{DATE}  RMSE={rmse:.2f}  bias={bias:+.2f} mm/day  pred_max={pred_mm.max():.1f}  gt_max={gt_mm.max():.1f}")

# ------------------------------------------------------------------
# 그림
# ------------------------------------------------------------------
proj = ccrs.PlateCarree()
levels = [0, 1, 2, 5, 10, 20, 30, 50, 80, 120, 180, 250]
cmap = plt.get_cmap("YlGnBu", len(levels))
norm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")

def add_map(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), lw=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), lw=0.5, ls=":")
    ax.set_extent(ext_hr, crs=proj)
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}

fig, axes = plt.subplots(1, 4, figsize=(16, 4.6),
                         subplot_kw={"projection": proj}, constrained_layout=True)
titles = ["LR 입력 (0.2°)", "예측 (0.1°)", "정답 IMERG (0.1°)", "오차 (예측−정답)"]
data = [(lr_mm, ext_lr), (pred_mm, ext_hr), (gt_mm, ext_hr)]

for c, (arr, ext) in enumerate(data):
    ax = axes[c]; add_map(ax)
    im = ax.imshow(arr, extent=ext, origin="lower", transform=proj,
                   cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(titles[c], fontsize=10)

emax = float(np.nanpercentile(np.abs(err), 99)) or 1.0
add_map(axes[3])
imd = axes[3].imshow(err, extent=ext_hr, origin="lower", transform=proj,
                     cmap="RdBu_r", vmin=-emax, vmax=emax, interpolation="nearest")
axes[3].set_title(titles[3], fontsize=10)
axes[3].text(0.02, 0.02, f"RMSE={rmse:.1f}  bias={bias:+.1f} mm/d",
             transform=axes[3].transAxes, fontsize=8,
             bbox=dict(fc="white", alpha=0.75, ec="none"))

fig.colorbar(im, ax=axes[:3], shrink=0.7, location="bottom", ticks=levels,
             label="강수량 (mm/day)", pad=0.02, aspect=40)
fig.colorbar(imd, ax=axes[3], shrink=0.7, location="bottom", label="mm/day", pad=0.02)
fig.suptitle(f"LIIF + topo(elev/slope/relief), adapter 미사용 — {DATE} (epoch {ckpt.get('epoch')})",
             fontsize=12)

out_png = os.path.join(OUT_DIR, f"pred_{DATE}_epoch_last.png")
fig.savefig(out_png, dpi=140)
print("saved:", out_png)

# ------------------------------------------------------------------
# NetCDF 저장
# ------------------------------------------------------------------
xr.Dataset(
    {
        "precip_pred": (("lat", "lon"), pred_mm.astype(np.float32)),
        "precip_gt":   (("lat", "lon"), gt_mm.astype(np.float32)),
    },
    coords={"lat": lat_hr, "lon": lon_hr},
    attrs={"date": DATE, "rmse_mm_day": rmse, "bias_mm_day": bias,
           "norm": f"minmax min={VMIN} max={VMAX}"},
).to_netcdf(os.path.join(OUT_DIR, f"pred_{DATE}_epoch_last.nc"))
print("saved:", os.path.join(OUT_DIR, f"pred_{DATE}_epoch_last.nc"))
