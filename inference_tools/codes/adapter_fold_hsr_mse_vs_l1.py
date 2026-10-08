"""
HSR adapter transfer function 비교: 기존(ASOS 1h 어긋난 짝으로 학습) vs lag1h(정렬 교정 짝으로 학습, 진행 중 체크포인트).
adapter_fold_phenomenon_hsr.py 와 같은 형식/축. 기준선으로 정렬된 HSR-ASOS 시간별 짝의 bin 평균 E[ASOS | HSR] 를 같이 그림
(x 가 모델 p_k 가 아니라 HSR 관측이라 경향 비교용).

출력: /home/inhye_yoo/adapter_fold_phenomenon_hsr_mse_vs_l1.png
env:  liif (CPU 만 사용)
"""
import time
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

V = 74.8784
S = "/home/inhye_yoo/deeplearning/inr/liif/save"
OLD = f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3"
NEW = f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h"
CACHE = "/tmp/claude-1018/-home-inhye-yoo/fa366d51-be66-48da-b7e8-102eabca7ed9/scratchpad/adapter_pk_cache_hsr.npz"
PAIRS = "/home/inhye_yoo/data/radar_HSR/analysis/results/pairs_hourly_jja_2019_2021.csv"

L1 = f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h_l1"
RUNS = [("MSE(lag1h) best-loss", f"{NEW}/epoch-best-loss.pth", "tab:blue", "-", 2.6),
        ("MSE(lag1h) last", f"{NEW}/epoch-last.pth", "tab:green", "-", 2.6),
        ("L1(lag1h) best-loss", f"{L1}/epoch-best-loss.pth", "tab:orange", "--", 2.6),
        ("L1(lag1h) last", f"{L1}/epoch-last.pth", "tab:red", "--", 2.6)]


def load_ckpt(path):
    for _ in range(5):                       # 학습 중 덮어쓰는 파일은 잠깐 깨져 있을 수 있어 재시도
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            time.sleep(3)
    raise RuntimeError(path)


def build(ck):
    sd = ck["model"]["sd"]
    ad = {k.replace("add_layer.asos_adapter.", ""): v for k, v in sd.items() if "add_layer.asos_adapter" in k}
    dims, L = [1, 32, 16, 32], []
    for a, b in zip(dims[:-1], dims[1:]):
        L += [nn.Linear(a, b), nn.ReLU()]
    m = nn.Sequential(*L, nn.Linear(dims[-1], 1)); m.load_state_dict(ad); m.eval()
    return m


pk = np.load(CACHE)["P_train"].ravel()
x = torch.linspace(0, 1.3, 4000).view(-1, 1)
xin = x.view(-1).numpy() * V

oh = pd.read_csv(PAIRS)
hx, hy = oh["radar_mm_h"].to_numpy(float), oh["asos_mm_h"].to_numpy(float)
edges = np.arange(0, hx.max() + 2, 2.0); ix = np.digitize(hx, edges) - 1
bx, by = zip(*[(hx[ix == i].mean(), hy[ix == i].mean()) for i in range(len(edges) - 1) if (ix == i).sum() >= 5])

fig, (ax, axh) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[4, 1], sharex=True)
ax.plot([0, V], [0, V], color="k", alpha=.3, lw=1, label="y = x (no correction)")
ax.axvspan(V, 1.3 * V, color="0.9")
ax.plot(bx, by, "o", color="#31548f", ms=3.5, label="E[ASOS | HSR] (aligned pairs, x=HSR obs)")
print(f"{'run':26s} ep   f(5)   f(10)  f(20)  f(30)  f(34)  mono")
for name, path, c, ls, lw in RUNS:
    ck = load_ckpt(path); m = build(ck)
    with torch.no_grad():
        y = m(x).view(-1).numpy() * V
    yc = np.clip(y, 0, None)
    ax.plot(xin, yc, color=c, ls=ls, lw=lw, label=f"{name} (ep {ck.get('epoch')})")
    f = lambda t: yc[np.argmin(abs(xin - t))]
    print(f"{name:26s} {ck.get('epoch')}  {f(5):5.1f}  {f(10):5.1f}  {f(20):5.1f}  {f(30):5.1f}  {f(34):5.1f}  {bool((np.diff(y) >= -0.01).all())}")
ax.axvline(pk.max(), color="0.5", lw=0.8, ls=":")
ax.text(V + 1, 5, f"extrapolation\n(train max ~{pk.max():.0f}mm/hr)", fontsize=8, color="0.4")
ax.set_xlim(0, 1.3 * V); ax.set_ylim(-3, V * 1.05)
ax.set_ylabel("adapter output / ASOS (mm/hr)")
ax.set_title("HSR MLP adapter transfer function (in training): MSE vs L1 loss, lag1h pairs")
ax.legend(loc="upper left", fontsize=8)

axh.hist(pk[pk > 0.1], bins=np.arange(0, 1.3 * V, 1), color="navy", alpha=.6)
axh.set_yscale("log"); axh.set_xlabel("adapter input = per-direction imnet output p_k (mm/hr)")
axh.set_ylabel("# p_k (log)")
plt.tight_layout()
out = "/home/inhye_yoo/adapter_fold_phenomenon_hsr_mse_vs_l1.png"
plt.savefig(out, dpi=135); print("saved", out)
