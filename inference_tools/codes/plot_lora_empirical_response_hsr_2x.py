"""Plot an empirical response curve for the 2x LTE+LoRA ASOS finetune.

Unlike the MLP-ASOS adapter, LoRA changes several hidden-vector projections in
LTE's imnet.  It therefore has no standalone scalar ``p_k -> output`` transfer
function.  This script plots the directly comparable empirical relationship:
base LTE prediction -> LoRA prediction / observed ASOS, using the aligned 2019
JJA training station-hour pairs.

Output: /home/inhye_yoo/lora_empirical_response_hsr_2x_r16_nofirst_lag1h.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
import datasets
import models
from models.lora import build_lora_lte

RUN = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_lte_lora_r16_nofirst_asos_lag1h"
CKPT = f"{RUN}/epoch-best-loss.pth"
OUT = "/home/inhye_yoo/lora_empirical_response_hsr_2x_r16_nofirst_lag1h.png"
PMAX = 74.8784
CROP = 48
BATCH = int(os.environ.get("BATCH", "128"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def station_query_batches(dataset):
    """Yield bounded batches of valid station queries without retaining all crops."""
    crops, coords, obs = [], [], []
    for idx in range(len(dataset)):
        img_lr, _, station_obs, _ = dataset[idx]
        height, width = img_lr.shape[-2:]
        values = station_obs["gt"]
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        mask = values.squeeze(-1) != -2.0
        if not mask.any():
            continue
        coord = station_obs["coord"][mask].clone()
        values = values[mask]
        coord[:, 0] = -coord[:, 0]  # dataset-to-geographic latitude convention
        for point, value in zip(coord, values):
            y, x = float(point[0]), float(point[1])
            i = (y + 1) / 2 * (height - 1)
            j = (x + 1) / 2 * (width - 1)
            i0 = min(max(int(round(i - CROP / 2)), 0), height - CROP)
            j0 = min(max(int(round(j - CROP / 2)), 0), width - CROP)
            crops.append(img_lr[:, i0:i0 + CROP, j0:j0 + CROP])
            coords.append(torch.tensor([
                ((i - i0) / (CROP - 1)) * 2 - 1,
                ((j - j0) / (CROP - 1)) * 2 - 1,
            ], dtype=torch.float32))
            obs.append(float(value[0]))
            if len(obs) == BATCH:
                yield torch.stack(crops), torch.stack(coords).unsqueeze(1), np.asarray(obs)
                crops, coords, obs = [], [], []
    if obs:
        yield torch.stack(crops), torch.stack(coords).unsqueeze(1), np.asarray(obs)


with open(f"{RUN}/config.yaml") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
meta_path = cfg["train_dataset"]["dataset"]["args"]["meta_path"]
with open(meta_path) as file:
    meta = json.load(file)

# The full train array is unnecessarily large for a diagnostic pass.
cfg["train_dataset"]["dataset"]["args"]["load_into_memory"] = False
train_ds = datasets.make(cfg["train_dataset"]["dataset"])
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
base_ck = torch.load(ck["base_ckpt"], map_location="cpu", weights_only=False)
base = models.make(base_ck["model"], load_sd=True).to(DEVICE).eval()
ft, _ = build_lora_lte(ck["base_ckpt"], ck["rank"], ck["layers"], ck["lora"], device=DEVICE)
ft.eval()
base.topo = ft.topo = None
print(f"device={DEVICE}; LoRA epoch={ck['epoch']}, rank={ck['rank']}, layers={ck['layers']}", flush=True)

base_pred, ft_pred, obs_parts = [], [], []
pair_count = 0
with torch.inference_mode():
    for batch_no, (crops, coords, obs_part) in enumerate(station_query_batches(train_ds), start=1):
        inp = crops.to(DEVICE)
        coord = coords.to(DEVICE)
        cell = torch.ones_like(coord)
        cell[:, :, 0] *= 2 / 1000
        cell[:, :, 1] *= 2 / 1000
        base_pred.append(base(inp, None, coord, cell).clamp(0, 1).flatten().cpu())
        ft_pred.append(ft(inp, None, coord, cell).clamp(0, 1).flatten().cpu())
        obs_parts.append(obs_part)
        pair_count += len(obs_part)
        if batch_no % 20 == 0:
            print(f"  predicted {pair_count} station-hour pairs", flush=True)

x = torch.cat(base_pred).numpy() * PMAX
y = torch.cat(ft_pred).numpy() * PMAX
obs = np.concatenate(obs_parts) * PMAX
print(f"aligned 2019 train station-hour pairs: {len(obs)}", flush=True)
upper = max(np.percentile(np.r_[x, y, obs], 99.8), 8.0)
upper = min(np.ceil(upper / 5) * 5, PMAX)
edges = np.arange(0, upper + 1.0, 1.0)
bin_id = np.digitize(x, edges) - 1
bx, by_ft, by_obs, count = [], [], [], []
for i in range(len(edges) - 1):
    selected = bin_id == i
    if selected.sum() >= 10:
        bx.append(x[selected].mean())
        by_ft.append(y[selected].mean())
        by_obs.append(obs[selected].mean())
        count.append(selected.sum())
bx, by_ft, by_obs, count = map(np.asarray, (bx, by_ft, by_obs, count))
slope, intercept = np.polyfit(x, obs, 1)
corr_base = np.corrcoef(x, obs)[0, 1]
corr_ft = np.corrcoef(y, obs)[0, 1]

fig, (ax, hist) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[4, 1], sharex=True)
ax.plot([0, upper], [0, upper], "r--", lw=1.2, label="y = x (no LoRA change)")
ax.plot(bx, by_obs, "o", color="#31548f", ms=3.5,
        label="E[ASOS | base LTE prediction] (aligned 2019 train pairs)")
ax.plot(bx, by_ft, "o-", color="tab:blue", ms=3.2, lw=2.4,
        label=f"E[LoRA output | base LTE prediction] (best-loss ep {ck['epoch']})")
line_x = np.array([0, upper])
ax.plot(line_x, slope * line_x + intercept, color="k", lw=1.8,
        label=f"OLS ASOS~base LTE (y={slope:.2f}x{intercept:+.2f}, r={corr_base:.2f}, n={len(x)})")
ax.set_ylim(-max(1, upper * .04), upper * 1.05)
ax.set_ylabel("LoRA output / ASOS (mm/hr)")
ax.set_title("2x HSR LTE + LoRA empirical response (r=16, no-first; lag1h pairs, 2019 train)")
ax.legend(loc="upper left", fontsize=8)
ax.text(.98, .04, "LoRA changes hidden-vector projections;\nthis is an empirical response curve, not\na standalone scalar adapter transfer function.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.35")
hist.hist(x[x > 0.001], bins=edges, color="navy", alpha=.6)
hist.set_yscale("log")
hist.set_xlabel("base LTE prediction at ASOS station (mm/hr)")
hist.set_ylabel("# pairs\n(log)")
fig.tight_layout()
fig.savefig(OUT, dpi=135)

def metrics(pred):
    return np.abs(pred - obs).mean(), (pred - obs).mean(), np.sqrt(np.mean((pred - obs) ** 2)), np.corrcoef(pred, obs)[0, 1]

print("base  MAE=%.3f bias=%+.3f RMSE=%.3f r=%.3f" % metrics(x))
print("LoRA  MAE=%.3f bias=%+.3f RMSE=%.3f r=%.3f" % metrics(y))
print(f"saved {OUT}")
