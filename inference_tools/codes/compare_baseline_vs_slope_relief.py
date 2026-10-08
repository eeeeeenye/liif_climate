"""
[DEPRECATED] slope/relief 채널 기능이 TopoInMemory에서 제거됨(효과 없음).
이 스크립트는 옛 slope/relief run(_climate_edsr_topo_80x100_topo_slope_relief_ver,
extra_in_dim=3)을 참조하므로 지금은 elevation-only로만 로드되어 결과가 무의미하다.
기록용으로만 남겨둠.

2019-07-20 (out-of-sample) 하루치 예측 비교  (체크포인트: epoch-best-loss)
  - baseline      : topo = elevation only   (_climate_edsr_topo_80x100_topo_precip_minmax)
  - slope/relief  : topo = elev+slope+relief (_climate_edsr_topo_80x100_topo_slope_relief_ver)

정규화: minmax, mm/day = value * 506.01996 (min = 0)

주의: 두 run 은 학습기간/topo 소스 파일이 달라 완전한 ablation 은 아님.
  baseline  : train 2000-2002, ETOPO1_on_IMERG_Land.nc (var z)
  s/r       : train 1998-2009, ETOPO1_Ice_g_gmt4_kor.nc (var elev)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"          # 무조건 GPU 2번

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
SAVE_ROOT = "/home/inhye_yoo/deeplearning/inr/liif/save"
RUNS = [
    # (label, run_dir, checkpoint 파일명)
    ("baseline (elev only)",  "_climate_edsr_topo_80x100_topo_precip_minmax",     "epoch-best-loss.pth"),
    ("elev + slope + relief", "_climate_edsr_topo_80x100_topo_slope_relief_ver",  "epoch-last.pth"),
]
DATE = "2019-07-20"
LR_RAW = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_02deg_1998_202504_noleaf.nc"
HR_RAW = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_01deg_1998_202504_noleaf.nc"
OUT_DIR = os.path.join(SAVE_ROOT, "_climate_edsr_topo_80x100_topo_slope_relief_ver", "preds")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # == 물리 GPU 2
TOPO_LAT = (33, 43)
TOPO_LON = (124, 132)
VMAX, VMIN = 506.01995849609375, 0.0

def to_mm(x):
    return x * (VMAX - VMIN) + VMIN

# ------------------------------------------------------------------
# 입력/정답 (raw mm/day -> /VMAX 로 정규화)
# ------------------------------------------------------------------
lr_da = xr.open_dataset(LR_RAW)["precipitation"].sel(time=DATE)
hr_da = xr.open_dataset(HR_RAW)["precipitation"].sel(time=DATE)
lat_hr, lon_hr = hr_da["lat"].values, hr_da["lon"].values
lat_lr, lon_lr = lr_da["lat"].values, lr_da["lon"].values
H, W = hr_da.shape[-2:]

lr_mm = np.nan_to_num(lr_da.values).astype(np.float32)
gt_mm = np.nan_to_num(hr_da.values).astype(np.float32)
lr_norm = lr_mm / VMAX

d_hr = float(lat_hr[1] - lat_hr[0]); d_lr = float(lat_lr[1] - lat_lr[0])
ext_hr = [lon_hr[0]-d_hr/2, lon_hr[-1]+d_hr/2, lat_hr[0]-d_hr/2, lat_hr[-1]+d_hr/2]
ext_lr = [lon_lr[0]-d_lr/2, lon_lr[-1]+d_lr/2, lat_lr[0]-d_lr/2, lat_lr[-1]+d_lr/2]

coord = make_coord((H, W)).unsqueeze(0).to(DEVICE)
cell = torch.ones_like(coord)
cell[..., 0] *= 2.0 / H
cell[..., 1] *= 2.0 / W


@torch.no_grad()
def run_model(run_dir, ckpt_name):
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    da  = cfg["val_dataset"]["dataset"]["args"]
    topo_ds = TopoInMemory(path_1=da["topo_nc_path"], lat_range=TOPO_LAT, lon_range=TOPO_LON,
                           **da.get("topo_kwargs", {}))
    topo = topo_ds[0].unsqueeze(0).to(DEVICE)

    ckpt = torch.load(os.path.join(run_dir, ckpt_name), map_location="cpu")
    model = make_model({"name": "liif", "args": copy.deepcopy(ckpt["model"]["args"])}).to(DEVICE)
    own = model.state_dict()
    compat = OrderedDict((k, v) for k, v in ckpt["model"]["sd"].items()
                         if k in own and v.shape == own[k].shape)
    model.load_state_dict(compat, strict=False)
    model.eval()

    inp = torch.from_numpy(lr_norm)[None, None].to(DEVICE)
    model.gen_feat(inp)
    model.topo = topo
    pred = model.query_rgb(coord, cell).view(H, W).clamp_(0.0, None).cpu().numpy()
    return to_mm(pred), int(ckpt.get("epoch", -1)), topo.shape[1]


results = []
for label, name, ckpt_name in RUNS:
    pred_mm, ep, nch = run_model(os.path.join(SAVE_ROOT, name), ckpt_name)
    err = pred_mm - gt_mm
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(pred_mm.ravel(), gt_mm.ravel())[0, 1])
    results.append(dict(label=label, pred=pred_mm, err=err, rmse=rmse, bias=bias,
                        corr=corr, epoch=ep, nch=nch, ckpt=ckpt_name))
    print(f"{label:24s} {ckpt_name:20s} epoch={ep:4d}  topoCh={nch}  "
          f"RMSE={rmse:.2f}  bias={bias:+.2f}  r={corr:.4f}")

# ------------------------------------------------------------------
# 그림  (행: LR/GT | baseline | s/r ,  열: 예측 / 오차)
# ------------------------------------------------------------------
proj = ccrs.PlateCarree()
levels = [0, 1, 2, 5, 10, 20, 30, 50, 80, 120, 180, 250]
cmap = plt.get_cmap("YlGnBu", len(levels))
pnorm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")
emax = float(np.nanpercentile(np.abs(np.concatenate([r["err"].ravel() for r in results])), 99))

def add_map(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), lw=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), lw=0.5, ls=":")
    ax.set_extent(ext_hr, crs=proj)
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 6}

fig, axes = plt.subplots(3, 2, figsize=(9, 13),
                         subplot_kw={"projection": proj}, constrained_layout=True)

add_map(axes[0, 0])
im = axes[0, 0].imshow(lr_mm, extent=ext_lr, origin="lower", transform=proj,
                       cmap=cmap, norm=pnorm, interpolation="nearest")
axes[0, 0].set_title("LR 입력 (0.2°)", fontsize=10)

add_map(axes[0, 1])
axes[0, 1].imshow(gt_mm, extent=ext_hr, origin="lower", transform=proj,
                  cmap=cmap, norm=pnorm, interpolation="nearest")
axes[0, 1].set_title("정답 IMERG (0.1°)", fontsize=10)

for i, r in enumerate(results):
    ax_p, ax_e = axes[i + 1, 0], axes[i + 1, 1]
    add_map(ax_p); add_map(ax_e)
    ax_p.imshow(r["pred"], extent=ext_hr, origin="lower", transform=proj,
                cmap=cmap, norm=pnorm, interpolation="nearest")
    ax_p.set_title(f"예측 — {r['label']}\n({r['ckpt']}, epoch {r['epoch']})", fontsize=9)
    imd = ax_e.imshow(r["err"], extent=ext_hr, origin="lower", transform=proj,
                      cmap="RdBu_r", vmin=-emax, vmax=emax, interpolation="nearest")
    ax_e.set_title("오차 (예측−정답)", fontsize=9)
    ax_e.text(0.02, 0.02,
              f"RMSE={r['rmse']:.2f}  bias={r['bias']:+.2f}\nr={r['corr']:.4f}",
              transform=ax_e.transAxes, fontsize=7,
              bbox=dict(fc="white", alpha=0.8, ec="none"))

fig.colorbar(im, ax=axes[:, 0], shrink=0.5, location="bottom",
             ticks=levels, label="강수량 (mm/day)", pad=0.02, aspect=35)
fig.colorbar(imd, ax=axes[:, 1], shrink=0.5, location="bottom",
             label="오차 (mm/day)", pad=0.02, aspect=35)
fig.suptitle(f"baseline vs slope/relief — {DATE}", fontsize=13)

out_png = os.path.join(OUT_DIR, f"compare_{DATE}_baseline_vs_slope_relief.png")
fig.savefig(out_png, dpi=140)
print("saved:", out_png)

xr.Dataset(
    {
        "pred_baseline":   (("lat", "lon"), results[0]["pred"].astype(np.float32)),
        "pred_slope_relief": (("lat", "lon"), results[1]["pred"].astype(np.float32)),
        "gt":              (("lat", "lon"), gt_mm.astype(np.float32)),
    },
    coords={"lat": lat_hr, "lon": lon_hr},
    attrs={"date": DATE,
           "baseline_rmse": results[0]["rmse"], "baseline_bias": results[0]["bias"],
           "sr_rmse": results[1]["rmse"], "sr_bias": results[1]["bias"]},
).to_netcdf(os.path.join(OUT_DIR, f"compare_{DATE}.nc"))
print("saved:", os.path.join(OUT_DIR, f"compare_{DATE}.nc"))
