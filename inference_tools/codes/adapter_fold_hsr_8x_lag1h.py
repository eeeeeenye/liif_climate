"""
8x LIIF ASOS-finetune MLP adapter transfer function, 2019 만 학습한 run (..._8x_minmax_liif_ft_mlp_1e-3_lag1h), adapter_fold_hsr_lag1h_compare.py 와 같은 형식/축. CPU 만 사용.
  - 파란 실선 : 이 run best-loss 의 adapter 변환
  - 점/빨간 점선 : 정렬된 HSR-ASOS 시간별 짝 (2019 = 이 run 의 학습 해) 의 E[ASOS|HSR] 와 OLS  (x 는 HSR 500m 관측; 모델 p_k 아님)
  - 아래 히스토그램 : adapter 입력 p_k (frozen imnet 8x base 출력). 캐시(adapter_fold_hsr_8x_2019_2020.py 가 만든 것) 중 앞 28468행(2019 블록) 만 사용 = 이 run 의 학습 짝. 주황 점 = E[ASOS | p_k].
출력: /home/inhye_yoo/adapter_fold_phenomenon_hsr_8x_lag1h.png
env:  liif
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

V = 74.8784
RUN = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h"
CACHE = "/tmp/claude-1018/-home-inhye-yoo/7e1a34f7-7351-4f0c-b04a-fc2f8340a769/scratchpad/adapter_pk_cache_hsr_8x_2019_2020.npz"
PAIRS = "/home/inhye_yoo/data/radar_HSR/analysis/results/pairs_hourly_jja_2019_2021.csv"
OUT = "/home/inhye_yoo/adapter_fold_phenomenon_hsr_8x_lag1h.png"

ck = torch.load(f"{RUN}/epoch-best-loss.pth", map_location="cpu", weights_only=False)
ad = {k.replace("add_layer.asos_adapter.", ""): v for k, v in ck["model"]["sd"].items() if "add_layer.asos_adapter" in k}
dims, L = [1, 32, 16, 32], []
for a, b in zip(dims[:-1], dims[1:]):
    L += [nn.Linear(a, b), nn.ReLU()]
m = nn.Sequential(*L, nn.Linear(dims[-1], 1)); m.load_state_dict(ad); m.eval()

_c = np.load(CACHE)
N19 = 28468                                             # 2019_2020 학습 csv 의 앞 2019 블록 = 이 run(2019 단독) 의 학습 짝 (순서/값 일치 확인함)
Pk, Ok = _c["P_train"][:N19], _c["O_train"][:N19]
pk = Pk.ravel(); ok_ = np.repeat(Ok, Pk.shape[1])       # p_k 4방향 각각에 같은 ASOS 를 짝지음
x = torch.linspace(0, 1.3, 4000).view(-1, 1); xin = x.view(-1).numpy() * V
with torch.no_grad():
    y = m(x).view(-1).numpy() * V
yc = np.clip(y, 0, None)

oh = pd.read_csv(PAIRS); oh = oh[pd.to_datetime(oh["t"]).dt.year == 2019]
hx = oh["radar_mm_h"].to_numpy(float)
edges = np.arange(0, hx.max() + 2, 2.0)
pix = np.digitize(pk, edges) - 1                       # E[ASOS | p_k]: 같은 2 mm/h bin, n>=5
px, py = zip(*[(pk[pix == i].mean(), ok_[pix == i].mean()) for i in range(len(edges) - 1) if (pix == i).sum() >= 5])

fig, (ax, axh) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[4, 1], sharex=True)
ax.plot([0, V], [0, V], color="red", ls="--", lw=1.2, label="y = x (no correction)")
ax.axvspan(V, 1.3 * V, color="0.9")
ax.plot(px, py, "o", color="tab:orange", ms=3.5, label="E[ASOS | p_k] (train pairs 2019, x=p_k, per-direction)")
ax.plot(xin, yc, color="tab:blue", lw=2.6, label=f"8x lag1h best-loss (ep {ck.get('epoch')})")
ax.axvline(pk.max(), color="0.5", lw=0.8, ls=":")
ax.text(V + 1, 5, f"extrapolation\n(p_k max ~{pk.max():.0f}mm/hr,\n2019 train)", fontsize=8, color="0.4")
ax.set_xlim(0, 1.3 * V); ax.set_ylim(-3, V * 1.05)
ax.set_ylabel("adapter output / ASOS (mm/hr)")
ax.set_title("8x HSR MLP adapter transfer function (MSE, lag1h pairs, 2019 train)")
ax.legend(loc="upper left", fontsize=8)
axh.hist(pk[pk > 0.1], bins=np.arange(0, 1.3 * V, 1), color="navy", alpha=.6)
axh.set_yscale("log"); axh.set_xlabel("adapter input = per-direction imnet output p_k (mm/hr)"); axh.set_ylabel("# p_k (log)")
plt.tight_layout(); plt.savefig(OUT, dpi=135)
f = lambda t: yc[np.argmin(abs(xin - t))]
print(f"ep {ck.get('epoch')}  f(5)={f(5):.1f} f(10)={f(10):.1f} f(20)={f(20):.1f} f(30)={f(30):.1f} f(40)={f(40):.1f} mono={bool((np.diff(y) >= -0.01).all())}")
print(f"saved {OUT}")
