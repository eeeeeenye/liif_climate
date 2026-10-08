import sys
sys.path.append("/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from models.adapter import ASOSAdapter

path = "/home/inhye_yoo/data/analysis/results/csv/pairs_imerg_asos_nearest_mse_linear_1998_2009_after_before_adapter_norm.csv"

df = pd.read_csv(path)

# 숫자형 변환
for col in ["ASOS", "PRED_gt", "ASOS_gt"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# inf -> NaN
df[["ASOS", "PRED_gt", "ASOS_gt"]] = (
    df[["ASOS", "PRED_gt", "ASOS_gt"]]
    .replace([np.inf, -np.inf], np.nan)
)

# ASOS 200 mm 이상만 선택
df = df[df["ASOS"] >= 200.0].copy()

# PRED_gt, ASOS_gt 유효값만 사용
df = df.dropna(subset=["PRED_gt", "ASOS_gt"]).copy()

# -2 sentinel이 있으면 제거
df = df[
    (df["PRED_gt"] != -2.0) &
    (df["ASOS_gt"] != -2.0)
].copy()

print("ASOS >= 200 mm 샘플 수:", len(df))

x_data = df["PRED_gt"].to_numpy(dtype=np.float64)
y_data = df["ASOS_gt"].to_numpy(dtype=np.float64)


# ============================================================
# 1. 닫힌 해
# ============================================================

X = np.column_stack([
    x_data,
    np.ones_like(x_data)
])

beta, _, _, _ = np.linalg.lstsq(
    X,
    y_data,
    rcond=None
)

w_closed = float(beta[0])
b_closed = float(beta[1])

y_pred_closed = w_closed * x_data + b_closed

ss_res = np.sum((y_data - y_pred_closed) ** 2)
ss_tot = np.sum((y_data - y_data.mean()) ** 2)

r2 = 1.0 - ss_res / ss_tot

print("\n[Closed-form: ASOS >= 200 mm]")
print("N:", len(df))
print("w:", w_closed)
print("b:", b_closed)
print("R2:", r2)


# ============================================================
# 2. 입력 중심화 (실제 ASOSAdapter가 쓰는 input_mean)
# ============================================================

x_mean = float(x_data.mean())

x = torch.tensor(
    x_data,
    dtype=torch.float32
).view(-1, 1)

y = torch.tensor(
    y_data,
    dtype=torch.float32
).view(-1, 1)


# ============================================================
# 3. Gradient descent (models/adapter.py의 ASOSAdapter를 그대로 사용)
# ============================================================

torch.manual_seed(0)

model = ASOSAdapter(in_dim=1, out_dim=1)
model.set_input_mean(x_mean)   # 실제 파이프라인에서 빠져있던 중심화를 여기선 제대로 세팅

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2
)

loss_fn = nn.MSELoss()

for epoch in range(10000):
    optimizer.zero_grad()

    pred = model(x)
    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        weight, bias_centered, bias_original = model.get_original_parameters()
        w_model = weight.item()
        b_original = bias_original.item()

        print(
            f"epoch={epoch:5d} | "
            f"loss={loss.item():.10f} | "
            f"w={w_model:.10f} | "
            f"b={b_original:.10f}"
        )


# ============================================================
# 4. 최종 비교
# ============================================================

weight, bias_centered, bias_original = model.get_original_parameters()
w_model = weight.item()
b_model = bias_original.item()

print("\n[Gradient descent, ASOSAdapter]")
print("final weight:", w_model)
print("final bias:", b_model)

print("\n[Closed-form]")
print("weight:", w_closed)
print("bias:", b_closed)