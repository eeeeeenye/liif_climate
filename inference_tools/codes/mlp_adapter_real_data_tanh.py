"""
mlp_adapter_real_data_scatter.py / mlp_adapter_zoom_0.15_0.45.py 의 tanh 버전.

대상: _climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-3_tanh / epoch-best.pth
      (adapter 활성화가 ReLU -> Tanh 로 바뀐 run. 구조는 동일: 1-32-16-32-1)

real x = 이전(base) 모델 예측 nc 의 픽셀값(정규화). 합성 입력 없음.
출력:
  mlp_adapter_real_data_scatter_tanh.png   전체 scatter (x, adapter raw output)
  mlp_adapter_zoom_0.15_0.45_tanh.png      x=0.15~0.45 확대, 정렬-연결 곡선
"""

import sys
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import torch
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from models.adapter import ASOSAdapter

path_base = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax_topo_raw/preds_nc/preds_not_merged_test/pred_liif_minmax_ft_01deg_1998_2024.nc"
path_mlp_ckpt = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-3_tanh/epoch-best.pth"

OUT_SCATTER = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/adapter_diagnostics/mlp_adapter_real_data_scatter_tanh.png"
OUT_ZOOM = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/adapter_diagnostics/mlp_adapter_zoom_0.15_0.45_tanh.png"

MAXV = 506.01995849609375
HIGH_X_THRESH = 0.30
BULK_SAMPLE_N = 200000
ZOOM_LO, ZOOM_HI = 0.15, 0.45

# =============================================================================
# 1) 실제 x값 추출
# =============================================================================
base_f = Dataset(path_base)
x_all = np.ma.filled(base_f.variables["precip"][:], np.nan)
base_f.close()

x_flat = (x_all / MAXV).ravel()
x_flat = x_flat[np.isfinite(x_flat)]
print(f"실제 (x) 개수 (결측 제외): {x_flat.size}")

# =============================================================================
# 2) adapter 로드 (tanh, strict)
# =============================================================================
ckpt = torch.load(path_mlp_ckpt, map_location="cpu", weights_only=False)
sd = ckpt["model"]["sd"] if "sd" in ckpt.get("model", {}) else ckpt["model"]
adapter_sd = {k.replace("add_layer.", ""): v for k, v in sd.items() if k.startswith("add_layer.")}

model = ASOSAdapter(1, 1, hidden_dims=[32, 16, 32])
model.load_state_dict(adapter_sd, strict=True)
model.eval()

# layer-1 (첫 Linear) pre-activation 이 0 이 되는 x = -b/w  (Tanh 중심점; ReLU 버전의 breakpoint 대응)
w1 = adapter_sd["asos_adapter.0.weight"].squeeze(1).numpy()
b1 = adapter_sd["asos_adapter.0.bias"].numpy()
centers = -b1 / w1
centers_zoom = np.sort(centers[(centers >= ZOOM_LO) & (centers <= ZOOM_HI)])
print(f"layer-1 tanh 중심점 (x in {ZOOM_LO}~{ZOOM_HI}): {np.round(centers_zoom, 4)}")

# =============================================================================
# 3) scatter 용 샘플
# =============================================================================
rng = np.random.default_rng(0)
high_mask = x_flat > HIGH_X_THRESH
bulk_idx = rng.choice(np.where(~high_mask)[0], size=BULK_SAMPLE_N, replace=False)
sel_idx = np.concatenate([np.where(high_mask)[0], bulk_idx])
x_sel = x_flat[sel_idx]

with torch.no_grad():
    y_raw_sel = model(torch.tensor(x_sel, dtype=torch.float32).unsqueeze(1)).squeeze(1).numpy()

print(f"scatter 샘플 개수: {x_sel.size}  (x>{HIGH_X_THRESH} real {high_mask.sum()}개 전부 포함)")
print(f"실제 x 최댓값: {x_sel.max():.4f}  (raw precip 약 {x_sel.max()*MAXV:.1f} mm/day)")

# =============================================================================
# 4) scatter 그림
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x_sel, y_raw_sel, s=2, alpha=0.15, color="#1f77b4",
           label="actual (x, adapter raw output) pairs")
ax.axhline(0, color="gray", linewidth=1, linestyle="--")
ax.plot([0, x_sel.max()], [0, x_sel.max()], color="green", linewidth=1, linestyle=":",
        label="y = x (input unchanged)")
ax.set_xlabel("x (normalized precip, actual base-model prediction)")
ax.set_ylabel("y (normalized precip, adapter raw output, before clamp)")
ax.set_title("mlp adapter (TANH) input-output: ACTUAL data only (1998-2024, epoch-best.pth)\n"
             "no synthetic input - every point is a real (date, pixel) occurrence")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(-0.7, 0.5)
secax = ax.secondary_xaxis("top", functions=(lambda x: x * MAXV, lambda mm: mm / MAXV))
secax.set_xlabel("raw precip (mm/day)")
plt.tight_layout()
plt.savefig(OUT_SCATTER, dpi=150, bbox_inches="tight")
print("saved:", OUT_SCATTER)

# =============================================================================
# 5) zoom 그림 (x=0.15~0.45, real x 전부 정렬-연결, 다운샘플 없음)
# =============================================================================
zmask = (x_flat >= ZOOM_LO) & (x_flat <= ZOOM_HI)
xz = np.sort(x_flat[zmask])
with torch.no_grad():
    yz = model(torch.tensor(xz, dtype=torch.float32).unsqueeze(1)).squeeze(1).numpy()

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(xz, yz, lw=2, color="#1f77b4",
        label="real x, sorted, connected (no downsampling)")
for i, c in enumerate(centers_zoom):
    ax.axvline(c, color="red", ls="--", lw=1,
               label="layer-1 tanh centers" if i == 0 else None)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"TANH adapter  zoom-in: x={ZOOM_LO}~{ZOOM_HI} (all real x, no sampling artifact)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_ZOOM, dpi=150, bbox_inches="tight")
print("saved:", OUT_ZOOM)
