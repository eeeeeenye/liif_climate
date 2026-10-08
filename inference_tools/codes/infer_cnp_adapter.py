""" 확률적 CNP adapter 추론.

    학습된 cnp-adapter (train_cnp_adapter.py) 로, 주어진 날짜에 대해
      - ASOS 전 지점을 context 로
      - 100x80 격자 전체를 target 으로
    넣어 아래 map 을 만든다.

      precip_base        Y0(x)                = 얼린 LTE 다운스케일 (mm/day)
      precip_correction  mu(x)                = adapter 보정량 기대값 (mm/day)
      precip_uncertainty sigma(x)             = 보정 불확실성 (mm/day)
      precip_final       clip(Y0(x)+mu(x), 0) = 최종 강수 (mm/day)
      precip_true        IMERG 0.1deg GT      (mm/day)

    좌표/정규화 레시피는 save/.../infer_cases.py 와 동일:
      LR  = imerg 0.2deg raw(mm/day) / 506.01995849609375  -> (1,1,50,40), lat 오름차순(row0=South)
      topo= ETOPO1_Ice_g_gmt4_kor.nc, TopoInMemory(lat(33,43), lon(124,132), standard_tanh)
      out = 100 x 80 (lat 33.0..42.9, lon 124.0..131.9)
    context/target 모두 cell = 2/1000 (학습 시 wrapper 가 ASOS 에 준 값과 동일).
    ASOS 좌표는 wrapper 와 똑같이 y 부호를 뒤집어 위경도 flip 버그를 교정한다.

    실행:
      python inference_tools/infer_cnp_adapter.py \
        --adapter_ckpt save/_cnp_adapter_lte_minmax/epoch-best-nll.pth \
        --dates 2020-08-01 2021-07-05 --gpu 2
"""

import os
import sys
import copy
import argparse

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.append("/home/inhye_yoo/deeplearning/inr/liif")
import models                       # noqa: E402
import models.cnp_adapter           # noqa: E402,F401  ('cnp-adapter' 등록)
from datasets.image_folder import TopoInMemory, ObsInMemory   # noqa: E402
from utils import make_coord        # noqa: E402

LR_NC = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_02deg_1998_202504_noleaf.nc"
HR_NC = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_01deg_1998_202504_noleaf.nc"
TOPO = "/home/inhye_yoo/data/level/NOAA_ETOPO1_topo/ETOPO1_Ice_g_gmt4_kor.nc"
OBS_CSV = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/preprocessing_codes/preprocessed_obs_daily_precip_2020_2022_utc_4380.csv"
META = "/home/inhye_yoo/deeplearning/inr/liif/load/climate/meta/train_imerg80x100_meta.json"

PMAX = 506.01995849609375
H_HR, W_HR = 100, 80
LAT0, LAT1 = 33.0, 42.9
LON0, LON1 = 124.0, 131.9
POINT_CELL = 2.0 / 1000.0          # 학습 시 ASOS cell 과 동일


def build_lte(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    spec = copy.deepcopy(ck["model"])
    m = models.make({"name": spec["name"], "args": spec["args"]}).to(device)
    m.load_state_dict(spec["sd"], strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m, ck.get("epoch", -1)


def build_adapter(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    spec = copy.deepcopy(ck["model"])
    m = models.make(spec, load_sd=True).to(device)
    m.eval()
    return m, ck.get("epoch", -1), ck.get("lte_from", None)


def load_topo(device):
    t = TopoInMemory(path_1=TOPO, lat_range=(33, 43), lon_range=(124, 132),
                     normalize="standard_tanh")
    lvl = t.level
    if lvl.dim() == 3:
        lvl = lvl.unsqueeze(0)
    return lvl.to(device).float()


@torch.no_grad()
def lte_query(lte, inp, level, coord, cell, chunk=20000):
    lte.gen_feat(inp)
    lte.topo = level
    n = coord.shape[1]
    outs = []
    for ql in range(0, n, chunk):
        qr = min(ql + chunk, n)
        outs.append(lte.query_rgb(coord[:, ql:qr], cell[:, ql:qr]))
    return torch.cat(outs, dim=1)


@torch.no_grad()
def infer_one(lte, adapter, date, level, grid_coord, grid_cell, obs_ds, device):
    # ---- LR (mm/day -> [0,1]) --------------------------------------------
    lr_raw = xr.open_dataset(LR_NC)["precipitation"].sel(time=date).values.astype(np.float32)
    inp = torch.from_numpy(lr_raw / PMAX).float()[None, None].to(device)   # (1,1,50,40)

    # ---- ASOS context (wrapper 와 동일하게 y 부호 반전) --------------------
    s = obs_ds[date]
    m = s["mask"].bool()
    obs_coord = s["coord"][m].clone().to(device)
    obs_coord[:, 0] = -obs_coord[:, 0]                                     # flip 교정
    obs_gt = s["gt"][m].to(device).view(1, -1, 1)                          # [0,1] minmax
    obs_coord = obs_coord.view(1, -1, 2)
    obs_cell = torch.full_like(obs_coord, POINT_CELL)

    y0_ctx = lte_query(lte, inp, level, obs_coord, obs_cell)              # (1,Nc,1)
    e_ctx = obs_gt - y0_ctx
    context = torch.cat([obs_coord, y0_ctx, e_ctx], dim=-1)              # (1,Nc,4)

    # ---- target = 전체 격자 --------------------------------------------
    y0_tgt = lte_query(lte, inp, level, grid_coord, grid_cell)           # (1,N,1)
    target = torch.cat([grid_coord, y0_tgt], dim=-1)                     # (1,N,3)

    mu, sigma = adapter(context, target)                                # (1,N)
    mu = mu.view(H_HR, W_HR).cpu().numpy()
    sigma = sigma.view(H_HR, W_HR).cpu().numpy()
    y0 = y0_tgt.view(H_HR, W_HR).cpu().numpy()

    base_mm = np.clip(y0, 0.0, None) * PMAX
    corr_mm = mu * PMAX
    sigma_mm = sigma * PMAX
    final_mm = np.clip(y0 + mu, 0.0, None) * PMAX

    gt = xr.open_dataset(HR_NC)["precipitation"].sel(time=date).values.astype(np.float32)

    # ASOS 지점 (원본 위경도, mm/day) - quicklook overlay 용
    rows = np.array(obs_ds.data[date], dtype=np.float64)          # (n,3) lat,lon,val
    return dict(base=base_mm, corr=corr_mm, sigma=sigma_mm, final=final_mm,
                gt=gt, obs_lat=rows[:, 0], obs_lon=rows[:, 1], obs_val=rows[:, 2],
                n_ctx=int(m.sum().item()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_ckpt", required=True)
    ap.add_argument("--lte_ckpt", default=None,
                    help="없으면 adapter 체크포인트의 lte_from 사용")
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--obs_csv", default=OBS_CSV)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--gpu", default="2")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    adapter, a_ep, lte_from = build_adapter(args.adapter_ckpt, device)
    lte_ckpt = args.lte_ckpt or lte_from
    assert lte_ckpt, "lte_ckpt 를 지정하거나 adapter 체크포인트에 lte_from 이 있어야 함"
    lte, l_ep = build_lte(lte_ckpt, device)
    print(f"adapter epoch={a_ep}  lte epoch={l_ep}  ({lte_ckpt})")

    obs_ds = ObsInMemory(root_path=args.obs_csv, meta_path=META)
    level = load_topo(device)

    grid_coord = make_coord((H_HR, W_HR)).unsqueeze(0).to(device)        # (1,8000,2)
    grid_cell = torch.full_like(grid_coord, POINT_CELL)

    lat_out = np.linspace(LAT0, LAT1, H_HR)
    lon_out = np.linspace(LON0, LON1, W_HR)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.adapter_ckpt)), "preds_cnp")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for date in args.dates:
        r = infer_one(lte, adapter, date, level, grid_coord, grid_cell, obs_ds, device)
        results[date] = r
        mae_f = float(np.nanmean(np.abs(r["final"] - r["gt"])))
        mae_b = float(np.nanmean(np.abs(r["base"] - r["gt"])))
        print(f"  {date}  n_ctx={r['n_ctx']:3d}  "
              f"MAE final={mae_f:.3f}  base={mae_b:.3f} mm/day  "
              f"| corr [{r['corr'].min():.1f},{r['corr'].max():.1f}]  "
              f"sigma mean={r['sigma'].mean():.2f}")

        xr.Dataset(
            {
                "precip_base": (("lat", "lon"), r["base"].astype(np.float32)),
                "precip_correction": (("lat", "lon"), r["corr"].astype(np.float32)),
                "precip_uncertainty": (("lat", "lon"), r["sigma"].astype(np.float32)),
                "precip_final": (("lat", "lon"), r["final"].astype(np.float32)),
                "precip_true": (("lat", "lon"), r["gt"].astype(np.float32)),
                "obs_lat": (("station",), r["obs_lat"].astype(np.float32)),
                "obs_lon": (("station",), r["obs_lon"].astype(np.float32)),
                "obs_val": (("station",), r["obs_val"].astype(np.float32)),
            },
            coords={"lat": lat_out, "lon": lon_out, "time": pd.Timestamp(date)},
            attrs={"pmax": PMAX, "adapter_epoch": int(a_ep), "lte_epoch": int(l_ep),
                   "units": "mm/day"},
        ).to_netcdf(os.path.join(out_dir, f"cnp_{date}.nc"))

    try:
        make_fig(results, lat_out, lon_out, out_dir)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("fig skipped:", e)


def make_fig(results, lat_out, lon_out, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = list(results)
    panels = [("precip_base", "Y0 (LTE)", "jet", None),
              ("precip_correction", "correction mu", "RdBu_r", "sym"),
              ("precip_uncertainty", "uncertainty sigma", "viridis", None),
              ("precip_final", "final = Y0+mu", "jet", None),
              ("precip_true", "GT (0.1deg)", "jet", None),
              ("diff", "final - GT", "RdBu_r", "sym")]
    key = {"precip_base": "base", "precip_correction": "corr",
           "precip_uncertainty": "sigma", "precip_final": "final",
           "precip_true": "gt"}

    nrow, ncol = len(dates), len(panels)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.4 * nrow), squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.1)

    for r_i, date in enumerate(dates):
        res = results[date]
        for c_i, (var, name, cmap, mode) in enumerate(panels):
            ax = axs[r_i][c_i]
            arr = (res["final"] - res["gt"]) if var == "diff" else res[key[var]]
            if mode == "sym":
                v = float(np.nanmax(np.abs(arr))) or 1.0
                kw = dict(cmap=cmap, vmin=-v, vmax=v)
            elif var in ("precip_base", "precip_final", "precip_true"):
                vmax = max(1.0, float(np.nanmax([res["gt"], res["final"], res["base"]])))
                kw = dict(cmap=cmap, vmin=0.0, vmax=vmax)
            else:
                kw = dict(cmap=cmap)
            im = ax.pcolormesh(lon_out, lat_out, arr, shading="auto", **kw)
            ax.scatter(res["obs_lon"], res["obs_lat"], s=6, c="k", marker="o",
                       linewidths=0, alpha=0.5)
            ax.set_title(f"{date}\n{name}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)

    out = os.path.join(out_dir, "quicklook.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
