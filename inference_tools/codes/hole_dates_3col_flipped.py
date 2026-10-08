"""
빵꾸(hole) 났던 13일:  raw(GT) | 구멍났던 모델(ft_mlp_1e-4) | flipped 재학습(ft_mlp_1e-3_flipped)  3열 지도.

- 모델을 직접 로드해 13일만 추론 (correction_layer=adapter 활성화: model(inp, topo, coord, cell) 로 forward 호출).
- LR 입력  : raw 0.2deg IMERG / 506.02  (minmax, data_norm 없음)
- topo     : ETOPO1_Ice_g_gmt4_kor.nc, standard_tanh, lat(33,43) lon(124,132)
- HR 출력  : 100x80 (0.1deg, lat 33~42.9 asc, lon 124~131.9)  -> clamp(0,1) * 506.02
- GT       : imerg_precip_korea_daily_01deg_..._noleaf.nc

검증: --check 로 기존 _ft_mlp_1e-4 예측 nc(epoch-last)와 자기 추론을 비교.
env: /home/inhye_yoo/anaconda3/envs/liif/bin/python
"""
import os, sys, copy, argparse, yaml
from collections import OrderedDict

import numpy as np
import pandas as pd
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
GT01 = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_01deg_1998_202504_noleaf.nc"
TOPO = "/home/inhye_yoo/data/level/NOAA_ETOPO1_topo/ETOPO1_Ice_g_gmt4_kor.nc"
SAVE = "/home/inhye_yoo/deeplearning/inr/liif/save"
OUT = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/adapter_diagnostics/hole_dates_3col_flipped.png"

H_HR, W_HR = 100, 80
LAT_HR = 33.0 + 0.1 * np.arange(H_HR)          # ascending
LON_HR = 124.0 + 0.1 * np.arange(W_HR)

DATES = ["1998-08-05", "2003-07-20", "2012-08-23", "2015-07-07", "2016-09-16",
         "2017-10-21", "2018-07-01", "2018-08-26", "2021-06-10", "2023-08-09",
         "2023-08-31", "2024-09-12", "2024-09-21"]

MODELS = [
    ("precip_minmax_ft_mlp (holed)",
     f"{SAVE}/_climate_edsr_topo_80x100_topo_precip_minmax_ft_mlp/epoch-best.pth"),
    ("ft_mlp_1e-3_flipped best",
     f"{SAVE}/_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-3_flipped/epoch-best.pth"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="2")
ap.add_argument("--check", action="store_true", help="기존 1e-4 예측 nc와 자기추론 비교")
args = ap.parse_args()
dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    margs = copy.deepcopy(ck["model"]["args"])
    # 일부 ckpt 는 model.args 가 불완전(예: layer_spec.args 에 hidden_dims 누락)
    # -> 같은 run 의 config.yaml model.args 로 보정
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    if os.path.exists(cfg_path):
        cfg_args = yaml.safe_load(open(cfg_path))["model"]["args"]
        if "layer_spec" in cfg_args:
            margs["layer_spec"] = cfg_args["layer_spec"]
    if "encoder_spec" in margs and "args" in margs["encoder_spec"]:
        margs["encoder_spec"]["args"].pop("freeze", None)
    m = make({"name": "liif", "args": margs}).to(dev)
    sd_ck = ck["model"]["sd"]
    new_sd = m.state_dict()
    compat = OrderedDict((k, sd_ck[k]) for k in new_sd
                         if k in sd_ck and sd_ck[k].shape == new_sd[k].shape)
    m.load_state_dict(compat, strict=False)
    m.eval()
    print(f"  loaded {ckpt_path.split('/')[-2]}  ({len(compat)}/{len(new_sd)} params)")
    return m


# --- 공통 텐서 ---
topo_ds = TopoInMemory(path_1=TOPO, lat_range=(33, 43), lon_range=(124, 132),
                       normalize="standard_tanh")
topo = topo_ds.level.unsqueeze(0).to(dev)          # (1,1,Ht,Wt)
coord = make_coord((H_HR, W_HR)).unsqueeze(0).to(dev)
cell = torch.ones_like(coord)
cell[..., 0] *= 2 / H_HR
cell[..., 1] *= 2 / W_HR
print(f"topo {tuple(topo.shape)}  coord {tuple(coord.shape)}")

raw02 = xr.open_dataset(RAW02)
gt01 = xr.open_dataset(GT01)


@torch.no_grad()
def infer(model, date):
    lr = raw02["precipitation"].sel(time=date).values.astype("float32") / MAXV  # (50,40) asc
    inp = torch.from_numpy(lr)[None, None].to(dev)
    pred = model(inp, topo, coord, cell)              # forward -> correction_layer 적용
    pred = pred.view(H_HR, W_HR).clamp(0, 1) * MAXV
    return pred.clamp(min=0).cpu().numpy()


# ================= 검증 =================
if args.check:
    ref = xr.open_dataset(f"{SAVE}/_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-4/"
                          "preds_nc/preds_check_epoch-last/pred_liif_mlp_1e-4_epoch-last_01deg_1998_2024.nc")
    mchk = load_model(f"{SAVE}/_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-4/epoch-last.pth")
    print("\n[check] 자기추론(epoch-last) vs 기존 nc")
    for d in DATES[:5]:
        mine = infer(mchk, d)
        rr = ref["precip"].sel(time=d).values
        for tag, cand in [("asis", mine), ("flipud", mine[::-1])]:
            c = np.corrcoef(cand.ravel(), rr.ravel())[0, 1]
            mae = np.abs(cand - rr).mean()
            print(f"  {d} {tag:7s} corr {c:.4f}  MAE {mae:.3f}  (mine max {cand.max():.0f} / ref {rr.max():.0f})")
    sys.exit(0)

# ================= 추론 + 그림 =================
cmap = plt.get_cmap("turbo").copy()
fig, axes = plt.subplots(len(DATES), 3, figsize=(16, 3.7 * len(DATES)),
                         subplot_kw={"projection": ccrs.PlateCarree()},
                         constrained_layout=True)
models = [(name, load_model(p)) for name, p in MODELS]

for r, date in enumerate(DATES):
    gt = gt01["precipitation"].sel(time=date).values
    preds = [(name, infer(m, date)) for name, m in models]

    # ── 행 전체(3패널) 공통 컬러스케일 = GT 기준 ── 억제/붕괴가 눈에 보이게
    # 컬러바 스타일: turbo, 10 mm/day 간격, 20 마다 눈금
    vmax = float(np.ceil(np.nanmax(gt) / 10) * 10)
    bounds = list(range(0, int(vmax) + 10, 10))
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="max")

    wet = gt > 5.0                       # GT 기준 '비 오는' 셀
    panels = [("raw (0.1 deg GT)", gt)] + preds
    for c, (title, arr) in enumerate(panels):
        ax = axes[r, c]
        ax.set_extent([124, 132, 33, 43], crs=ccrs.PlateCarree())
        mp = ax.pcolormesh(LON_HR, LAT_HR, arr, transform=ccrs.PlateCarree(),
                           cmap=cmap, norm=norm, shading="auto")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        gl.top_labels = gl.right_labels = False
        if c == 0:
            t = f"{date}   raw GT   max {np.nanmax(arr):.0f}"
        else:
            r_gt = np.corrcoef(arr[wet], gt[wet])[0, 1]
            dry = float((arr[wet] < 1.0).mean() * 100)
            t = f"{title}   max {np.nanmax(arr):.0f}   r|wet {r_gt:.2f}   ~0 in GT-rain {dry:.0f}%"
        ax.set_title(t, fontsize=8)
    plt.colorbar(mp, ax=axes[r, :].tolist(), shrink=0.9, pad=0.01, extend="max",
                 ticks=[b for b in bounds if b % 20 == 0], label="Precipitation (mm/day)")

fig.suptitle("Hole dates (adapter ON):  raw GT  |  precip_minmax_ft_mlp (holed)  |  ft_mlp 1e-3 flipped"
             "   — color scale shared per row", fontsize=13)
plt.savefig(OUT, dpi=115, bbox_inches="tight")
print("saved", OUT)
