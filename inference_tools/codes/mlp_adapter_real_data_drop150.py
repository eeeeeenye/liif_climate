"""
mlp_adapter_real_data_scatter.py 스타일 그림 — _ft_mlp_1e-4_drop150 / epoch-last.

주의: 이 체크포인트는 adapter 활성화가 ReLU (옛 adapter.py). 현재 models/adapter.py 는
Tanh 이므로 ASOSAdapter 를 import 하지 않고 여기서 ReLU MLP forward 를 직접 구현한다.

real x = 이전(base) 모델 _minmax_topo_raw 예측 nc 픽셀값(정규화). 합성 입력 없음.
출력:
  mlp_adapter_real_data_scatter_drop150.png
  mlp_adapter_zoom_0.15_0.45_drop150.png
"""

import numpy as np
import torch
from netCDF4 import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path_base = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax_topo_raw/preds_nc/preds_not_merged_test/pred_liif_minmax_ft_01deg_1998_2024.nc"
path_ckpt = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-4_drop150/epoch-last.pth"

OUT_SCATTER = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/adapter_diagnostics/mlp_adapter_real_data_scatter_drop150.png"
OUT_ZOOM = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/adapter_diagnostics/mlp_adapter_zoom_0.15_0.45_drop150.png"

MAXV = 506.01995849609375
HIGH_X_THRESH = 0.30
BULK_SAMPLE_N = 200000
ZOOM_LO, ZOOM_HI = 0.15, 0.45

# ---- adapter (ReLU MLP 1-32-16-32-1) ----
ckpt = torch.load(path_ckpt, map_location="cpu", weights_only=False)
sd = ckpt["model"]["sd"] if "sd" in ckpt.get("model", {}) else ckpt["model"]
ad = {k.replace("add_layer.", ""): v for k, v in sd.items() if k.startswith("add_layer.")}
W = [ad[f"asos_adapter.{i}.weight"] for i in (0, 2, 4, 6)]
B = [ad[f"asos_adapter.{i}.bias"] for i in (0, 2, 4, 6)]


def adapter_relu(x):  # x: (N,1) tensor -> (N,1)
    h = x
    for k in range(4):
        h = h @ W[k].T + B[k]
        if k < 3:
            h = torch.relu(h)
    return h


w1 = W[0].squeeze(1).numpy()
b1 = B[0].numpy()
bp = -b1 / w1
bp_zoom = np.sort(bp[(bp >= ZOOM_LO) & (bp <= ZOOM_HI)])
print(f"ckpt epoch={ckpt.get('epoch')}  layer-1 ReLU breakpoints in {ZOOM_LO}~{ZOOM_HI}: {np.round(bp_zoom,4)}")

# ---- real x ----
base_f = Dataset(path_base)
x_flat = (np.ma.filled(base_f.variables["precip"][:], np.nan) / MAXV).ravel()
base_f.close()
x_flat = x_flat[np.isfinite(x_flat)]
print(f"실제 x 개수: {x_flat.size}")

rng = np.random.default_rng(0)
high_mask = x_flat > HIGH_X_THRESH
bulk_idx = rng.choice(np.where(~high_mask)[0], size=BULK_SAMPLE_N, replace=False)
sel_idx = np.concatenate([np.where(high_mask)[0], bulk_idx])
x_sel = x_flat[sel_idx]
with torch.no_grad():
    y_sel = adapter_relu(torch.tensor(x_sel, dtype=torch.float32).unsqueeze(1)).squeeze(1).numpy()
print(f"scatter 샘플 {x_sel.size}  (x>{HIGH_X_THRESH} real {int(high_mask.sum())}개 전부 포함)  x_max={x_sel.max():.4f} (~{x_sel.max()*MAXV:.0f} mm)")

# ---- scatter ----
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x_sel, y_sel, s=2, alpha=0.15, color="#1f77b4",
           label="actual (x, adapter raw output) pairs")
ax.axhline(0, color="gray", lw=1, ls="--")
ax.plot([0, x_sel.max()], [0, x_sel.max()], color="green", lw=1, ls=":",
        label="y = x (input unchanged)")
ax.set_xlabel("x (normalized precip, actual base-model prediction)")
ax.set_ylabel("y (normalized precip, adapter raw output, before clamp)")
ax.set_title("mlp adapter (ReLU, drop150) input-output: ACTUAL data only (1998-2024, epoch-last.pth)\n"
             "no synthetic input - every point is a real (date, pixel) occurrence")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(-0.7, 0.5)
secax = ax.secondary_xaxis("top", functions=(lambda x: x * MAXV, lambda mm: mm / MAXV))
secax.set_xlabel("raw precip (mm/day)")
plt.tight_layout()
plt.savefig(OUT_SCATTER, dpi=150, bbox_inches="tight")
print("saved:", OUT_SCATTER)

# ---- zoom 0.15~0.45 ----
zmask = (x_flat >= ZOOM_LO) & (x_flat <= ZOOM_HI)
xz = np.sort(x_flat[zmask])
with torch.no_grad():
    yz = adapter_relu(torch.tensor(xz, dtype=torch.float32).unsqueeze(1)).squeeze(1).numpy()
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(xz, yz, lw=2, color="#1f77b4", label="real x, sorted, connected (no downsampling)")
for i, c in enumerate(bp_zoom):
    ax.axvline(c, color="red", ls="--", lw=1,
               label=f"layer-1 breakpoints ({', '.join(f'{v:.4f}' for v in bp_zoom)})" if i == 0 else None)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"ReLU drop150 adapter  zoom-in: x={ZOOM_LO}~{ZOOM_HI} (all real x, no sampling artifact)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_ZOOM, dpi=150, bbox_inches="tight")
print("saved:", OUT_ZOOM)
