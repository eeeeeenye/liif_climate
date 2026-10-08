"""
빵꾸 났던 13일 —  ft_mlp_1e-3_flipped (epoch-best, adapter/correction ON) 예측만.
(hole_dates_3col_flipped.py 에서 오른쪽 열만 따로 저장)

env: /home/inhye_yoo/anaconda3/envs/liif/bin/python
"""
import sys, copy, argparse
from collections import OrderedDict

import numpy as np
import torch
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
from models.models import make
from datasets.image_folder import TopoInMemory
from utils import make_coord

MAXV = 506.01995849609375
RAW02 = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_02deg_1998_202504_noleaf.nc"
TOPO = "/home/inhye_yoo/data/level/NOAA_ETOPO1_topo/ETOPO1_Ice_g_gmt4_kor.nc"
CKPT = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-3_flipped/epoch-best.pth"
OUT = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/adapter_diagnostics/hole_dates_flipped_only.png"

H_HR, W_HR = 100, 80
LAT_HR = 33.0 + 0.1 * np.arange(H_HR)
LON_HR = 124.0 + 0.1 * np.arange(W_HR)
DATES = ["1998-08-05", "2003-07-20", "2012-08-23", "2015-07-07", "2016-09-16",
         "2017-10-21", "2018-07-01", "2018-08-26", "2021-06-10", "2023-08-09",
         "2023-08-31", "2024-09-12", "2024-09-21"]

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="2")
args = ap.parse_args()
dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

ck = torch.load(CKPT, map_location=dev, weights_only=False)
margs = copy.deepcopy(ck["model"]["args"])
margs.get("encoder_spec", {}).get("args", {}).pop("freeze", None)
model = make({"name": "liif", "args": margs}).to(dev)
sd = ck["model"]["sd"]; nsd = model.state_dict()
model.load_state_dict(OrderedDict((k, sd[k]) for k in nsd
                                  if k in sd and sd[k].shape == nsd[k].shape), strict=False)
model.eval()

topo = TopoInMemory(path_1=TOPO, lat_range=(33, 43), lon_range=(124, 132),
                    normalize="standard_tanh").level.unsqueeze(0).to(dev)
coord = make_coord((H_HR, W_HR)).unsqueeze(0).to(dev)
cell = torch.ones_like(coord); cell[..., 0] *= 2 / H_HR; cell[..., 1] *= 2 / W_HR
raw02 = xr.open_dataset(RAW02)


@torch.no_grad()
def infer(date):
    lr = raw02["precipitation"].sel(time=date).values.astype("float32") / MAXV
    inp = torch.from_numpy(lr)[None, None].to(dev)
    pred = model(inp, topo, coord, cell).view(H_HR, W_HR).clamp(0, 1) * MAXV
    return pred.clamp(min=0).cpu().numpy()


cmap = plt.get_cmap("turbo")
ncol = 4
nrow = int(np.ceil(len(DATES) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.6 * nrow),
                         subplot_kw={"projection": ccrs.PlateCarree()},
                         constrained_layout=True)
axes = axes.ravel()

for i, date in enumerate(DATES):
    arr = infer(date)
    ax = axes[i]
    # 컬러바 스타일은 mlp_hole_dates_viz.py 의 minmax_mlp 패널과 동일:
    # turbo, 10 mm/day 간격, 20 마다 눈금, extend="max"
    vmax = float(np.ceil(np.nanmax(arr) / 10) * 10) or 10
    bounds = list(range(0, int(vmax) + 10, 10))
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="max")
    ax.set_extent([124, 132, 33, 43], crs=ccrs.PlateCarree())
    mp = ax.pcolormesh(LON_HR, LAT_HR, arr, transform=ccrs.PlateCarree(),
                       cmap=cmap, norm=norm, shading="auto")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    ax.set_title(f"{date}   max {np.nanmax(arr):.0f} mm", fontsize=10)
    plt.colorbar(mp, ax=ax, shrink=0.8, pad=0.03, extend="max",
                 ticks=[b for b in bounds if b % 20 == 0], label="Precipitation (mm/day)")

for j in range(len(DATES), len(axes)):
    axes[j].axis("off")

fig.suptitle("ft_mlp_1e-3_flipped (epoch-best, adapter ON) — hole dates", fontsize=13)
plt.savefig(OUT, dpi=120, bbox_inches="tight")
print("saved", OUT)
