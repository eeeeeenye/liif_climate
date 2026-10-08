import sys
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import torch
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from models.adapter import ASOSAdapter

# ---- mlp adapter 이전(base) 체크포인트의 예측 nc: adapter의 실제 입력 x가 여기서 나옴 ----
path_base = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax_topo_raw/preds_nc/preds_not_merged_test/pred_liif_minmax_ft_01deg_1998_2024.nc"
path_mlp_ckpt = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_topo_80x100_topo_precip_minmax_ft_mlp/epoch-best.pth"

MAXV = 506.01995849609375  # train_imerg80x100_meta.json의 min-max 정규화 기준값

HIGH_X_THRESH = 0.30   # 이 이상은 희소한 고강수 구간이라 전부 포함
BULK_SAMPLE_N = 200000  # 나머지 저강수 구간은 랜덤 샘플링

# =============================================================================
# 1) 실제 x값 추출 (1998-2024 전체 기간 x 전체 격자, 합성 입력 없음)
# =============================================================================
base_f = Dataset(path_base)
x_all = np.ma.filled(base_f.variables["precip"][:], np.nan)  # (time, lat, lon), mm
base_f.close()

x_flat = (x_all / MAXV).ravel()
x_flat = x_flat[np.isfinite(x_flat)]
print(f"실제 (x) 개수 (결측 제외): {x_flat.size}")

# =============================================================================
# 2) 실제 x를 실제 학습된 mlp adapter에 통과시켜 y(clamp 이전 raw 출력) 계산
# =============================================================================
ckpt = torch.load(path_mlp_ckpt, map_location="cpu", weights_only=False)
sd = ckpt["model"]["sd"] if "sd" in ckpt.get("model", {}) else ckpt["model"]
adapter_sd = {k.replace("add_layer.", ""): v for k, v in sd.items() if k.startswith("add_layer.")}

model = ASOSAdapter(1, 1, hidden_dims=[32, 16, 32])
model.load_state_dict(adapter_sd, strict=True)

rng = np.random.default_rng(0)
high_mask = x_flat > HIGH_X_THRESH
bulk_idx = rng.choice(np.where(~high_mask)[0], size=BULK_SAMPLE_N, replace=False)
sel_idx = np.concatenate([np.where(high_mask)[0], bulk_idx])
x_sel = x_flat[sel_idx]

with torch.no_grad():
    y_raw_sel = model(torch.tensor(x_sel, dtype=torch.float32).unsqueeze(1)).squeeze(1).numpy()

print(f"플롯에 쓴 샘플 개수: {x_sel.size}  (x>{HIGH_X_THRESH}인 real 데이터 {high_mask.sum()}개 전부 포함)")
print(f"실제 x 최댓값: {x_sel.max():.4f}  (raw precip 약 {x_sel.max()*MAXV:.1f} mm/day)")

# =============================================================================
# 3) 시각화
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x_sel, y_raw_sel, s=2, alpha=0.15, color="#1f77b4",
           label="actual (x, adapter raw output) pairs")
ax.axhline(0, color="gray", linewidth=1, linestyle="--")
ax.plot([0, x_sel.max()], [0, x_sel.max()], color="green", linewidth=1, linestyle=":",
        label="y = x (input unchanged)")

ax.set_xlabel("x (normalized precip, actual base-model prediction)")
ax.set_ylabel("y (normalized precip, adapter raw output, before clamp)")
ax.set_title("mlp adapter input-output: ACTUAL data only (1998-2024, epoch-best.pth)\n"
              "no synthetic input — every point is a real (date, pixel) occurrence")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(-0.7, 0.5)

secax = ax.secondary_xaxis("top", functions=(lambda x: x * MAXV, lambda mm: mm / MAXV))
secax.set_xlabel("raw precip (mm/day)")

plt.tight_layout()
plt.savefig("/home/inhye_yoo/deeplearning/inr/liif/inference_tools/adapter_diagnostics/mlp_adapter_real_data_scatter.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("saved")
