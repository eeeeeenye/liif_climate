"""
8x LIIF ASOS-finetune (_climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h, epoch-best-loss, ep 772) 의
'점별 보정 전후 이동' 을 그린다. 서로 다른 두 비교를 따로 보여준다.
  (1) 원래 HSR  -> 최종 pred          (move_hsr_to_pred_<연도>.png)   : 점별 짝, ASOS 높이에서 화살표
  (2) 어댑터 입력 p_k -> 어댑터 출력 g(p_k) (adapter_in_out_pk_2019.png) : 방향별 p_k 가 전달함수 위에 놓임
  (3) HSR -> imnet 출력(p_k 가중평균) -> 최종 pred -> ASOS (three_stage_<연도>.png) : 두 비교가 한 점에서 어떻게 동시에 성립하는지
입력:
  - station_preds_{train2019,val2021}_8x_ft_epoch-best-loss.npz   (HSR, 최종 pred, ASOS)
  - station_preds_pretrain_8x_{train2019,val2021}_epoch-last.npz  (어댑터 없는 imnet 출력 = p_k 면적가중평균; 2019 에서 p_k 평균과 corr 0.993 확인)
  - adapter p_k 캐시 (2019, 방향별 4개) : adapter_pk_cache_hsr_8x_2019_2020.npz 앞 28468행 (=2019 학습 짝)
출력: inference_tools/station_eval/figures/hexbin_hsr_preds_vs_asos/
env: liif (CPU)
"""
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "0.4",
                     "axes.grid": True, "grid.color": "0.92", "grid.linewidth": 0.8, "font.size": 10})
UP, DOWN, NEU, INK, SUB = "#2b6cb0", "#d9731a", "0.65", "0.15", "0.4"   # 올라감=파랑, 내려감=주황, 변화 작음=회색

V = 74.8784
P = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/station_eval/"
OUT = P + "figures/hexbin_hsr_preds_vs_asos/"
RUN = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h"
CACHE = "/tmp/claude-1018/-home-inhye-yoo/7ba25758-455d-42df-8513-9f96417dc540/scratchpad/adapter_pk_cache_hsr_8x_2019_2020.npz"
SRC = {2019: ("station_preds_train2019_8x_ft_epoch-best-loss.npz", "station_preds_pretrain_8x_train2019_epoch-last.npz"),
       2021: ("station_preds_val2021_8x_ft_epoch-best-loss.npz", "station_preds_pretrain_8x_val2021_epoch-last.npz")}

ck = torch.load(f"{RUN}/epoch-best-loss.pth", map_location="cpu", weights_only=False)
ad = {k.replace("add_layer.asos_adapter.", ""): v for k, v in ck["model"]["sd"].items() if "add_layer.asos_adapter" in k}
L = []
for a, b in zip([1, 32, 16, 32][:-1], [1, 32, 16, 32][1:]): L += [nn.Linear(a, b), nn.ReLU()]
net = nn.Sequential(*L, nn.Linear(32, 1)); net.load_state_dict(ad); net.eval()
def g(p):
    with torch.no_grad(): return np.clip(net(torch.tensor(np.asarray(p) / V, dtype=torch.float32).reshape(-1, 1)).numpy().ravel() * V, 0, None)

def load(y):
    f, p = SRC[y]; d, q = np.load(P + "preds_npz/" + f), np.load(P + "preds_npz/" + p)
    return d["hsr_hr"], q["pf"].astype(float), d["pf"].astype(float), d["obs"]      # HSR, imnet(p_k blend), 최종 pred, ASOS

# ---------------------------------------------------------------- (1) HSR -> 최종 pred
def fig_move(y, thr=10, lim=50):
    h, pb, pf, o = load(y); m = h >= thr; r = pf[m] / h[m]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1, 1.15]})
    hb = a1.hexbin(h, pf, gridsize=60, bins="log", extent=(0, lim, 0, lim), mincnt=1, cmap="Greys", linewidths=0.2)
    a1.plot([0, lim], [0, lim], color=INK, lw=1, ls="--"); a1.set_xlim(0, lim); a1.set_ylim(0, lim); a1.set_aspect("equal")
    lo, hi = (pf < 0.8 * h) & (h >= thr), (pf > 1.2 * h) & (h >= thr)
    a1.text(0.97, 0.03, f"HSR≥{thr}: 내려감 {lo.sum()}점 · 올라감 {hi.sum()}점 · 그 외 {m.sum()-lo.sum()-hi.sum()}점\n(내려감 pred<0.8×HSR, 올라감 pred>1.2×HSR)", transform=a1.transAxes, ha="right", va="bottom", fontsize=8.5, color=SUB)
    a1.set_xlabel("원래 HSR 500 m (mm/h)"); a1.set_ylabel("최종 pred (mm/h)"); a1.set_title("점별 짝: x = HSR, y = 그 점의 최종 pred", fontsize=10.5, loc="left", color=INK)
    fig.colorbar(hb, ax=a1, fraction=0.04, pad=0.02, label="count (log)")
    cl = np.where(r < 0.8, DOWN, np.where(r > 1.2, UP, NEU))
    segs = [[(h[m][i], o[m][i]), (pf[m][i], o[m][i])] for i in range(m.sum())]
    a2.add_collection(LineCollection(segs, colors=cl, linewidths=1.0, alpha=0.75))
    a2.scatter(h[m], o[m], s=14, facecolors="white", edgecolors=cl, linewidths=0.9, zorder=3)
    a2.scatter(pf[m], o[m], s=14, c=cl, zorder=4)
    a2.plot([0, 70], [0, 70], color=INK, lw=1, ls="--"); a2.set_xlim(0, lim); a2.set_ylim(0, 70)
    a2.set_xlabel("HSR (○) → 최종 pred (●), mm/h"); a2.set_ylabel("ASOS (mm/h)  — 점마다 같은 높이")
    a2.set_title(f"HSR ≥ {thr} 점의 이동 (n={m.sum()}): 같은 ASOS 높이에서 HSR→pred", fontsize=10.5, loc="left", color=INK, pad=16)
    a2.text(0.0, 1.0, "주황 = 내려감, 파랑 = 올라감, 회색 = 변화 ±20% 이내", transform=a2.transAxes, ha="left", va="bottom", fontsize=8.5, color=SUB)
    fig.suptitle(f"{y} JJA 시간별 · 8x LIIF ASOS 미세조정 (epoch-best-loss)", fontsize=12, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{OUT}move_hsr_to_pred_{y}.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------- (2) 어댑터 입력 p_k -> 출력 g(p_k)
def fig_adapter():
    pk = np.load(CACHE)["P_train"][:28468]; o = np.load(P + "preds_npz/" + SRC[2019][0])["obs"]
    x = np.linspace(0, 45, 800); fx = g(x); pkf = pk.ravel(); rng = np.random.default_rng(0); idx = rng.choice(pkf.size, 4000, replace=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
    a1.plot([0, 45], [0, 45], color=INK, lw=1, ls="--", label="y = x")
    a1.plot(x, fx, color=UP, lw=2.2, label="어댑터 전달함수 g (ep 772)")
    a1.scatter(pkf[idx], g(pkf[idx]), s=6, c="0.35", alpha=0.35, label="실제 입력 p_k (방향별 4개, 4000개 표본)", zorder=3)
    for v in (10, 20, 30):
        a1.annotate(f"{v} → {g([v])[0]:.1f}", (v, g([v])[0]), xytext=(v + 3.0, g([v])[0] - 7.5), fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8))
    a1.axvline(pkf.max(), color="0.5", ls=":", lw=0.9); a1.text(pkf.max() + 0.5, 2, f"학습 p_k 최대 {pkf.max():.0f}", fontsize=8.5, color=SUB)
    a1.set_xlim(0, 45); a1.set_ylim(0, 50); a1.set_xlabel("어댑터 입력 p_k (imnet 방향별 출력, mm/h)"); a1.set_ylabel("어댑터 출력 g(p_k) (mm/h)")
    a1.set_title("p_k → g(p_k): 모든 점이 같은 곡선 위에 놓임", fontsize=10.5, loc="left", color=INK); a1.legend(loc="upper left", fontsize=8.5, frameon=False)
    d = g(pkf) - pkf; bins = np.arange(0, 41, 2); ii = np.digitize(pkf, bins) - 1
    c = 0.5 * (bins[:-1] + bins[1:]); med = [np.median(d[ii == k]) for k in range(len(c))]
    a2.axhline(0, color=INK, lw=1); a2.plot(x, fx - x, color=UP, lw=2.2)
    a2.fill_between(x, fx - x, 0, where=(fx - x) < 0, color=DOWN, alpha=0.35, label="g(p_k) < p_k (어댑터가 낮춤)")
    a2.fill_between(x, fx - x, 0, where=(fx - x) >= 0, color=UP, alpha=0.2, label="g(p_k) > p_k (어댑터가 올림)")
    a2.set_xlim(0, 45); a2.set_xlabel("어댑터 입력 p_k (mm/h)"); a2.set_ylabel("g(p_k) − p_k (mm/h)")
    a2.set_title("어댑터가 바꾸는 양은 p_k 만의 함수", fontsize=10.5, loc="left", color=INK); a2.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax2 = a2.twinx(); ax2.hist(pkf[pkf > 0.1], bins=np.arange(0, 45, 1), color="0.5", alpha=0.25); ax2.set_yscale("log"); ax2.set_ylabel("p_k 개수 (log, 회색 막대)", color=SUB); ax2.grid(False); ax2.spines["right"].set_visible(True)
    fig.suptitle("2019 JJA 학습 짝 · 어댑터 입력 p_k 와 출력 g(p_k) 의 비교 (HSR 과는 다른 비교)", fontsize=12, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}adapter_in_out_pk_2019.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------- (3) HSR -> imnet -> 최종 pred -> ASOS (점별 평행좌표)
def fig_stage(y, thr=15):
    h, pb, pf, o = load(y); m = h >= thr; H, B, F, O = h[m], pb[m], pf[m], o[m]
    better = abs(F - O) < abs(H - O); col = np.where(better, UP, DOWN)
    fig, ax = plt.subplots(figsize=(9.5, 6.2)); X = np.arange(4)
    for i in np.argsort(~better):
        ax.plot(X, [H[i], B[i], F[i], O[i]], color=col[i], lw=0.9, alpha=0.5)
    for k, v in enumerate((H, B, F, O)): ax.plot(k, np.median(v), marker="_", ms=26, color=INK, mew=2.2, zorder=5); ax.text(k + 0.06, np.median(v), f"중앙 {np.median(v):.1f}", fontsize=8.5, color=INK, va="center")
    ax.set_xticks(X); ax.set_xticklabels(["원래 HSR", "imnet 출력\n(어댑터 전, p_k 가중평균)", "최종 pred\n(어댑터 후)", "ASOS"]); ax.set_xlim(-0.2, 3.5)
    ax.set_ylabel("mm/h"); ax.grid(axis="x", color="0.8")
    ax.set_title(f"{y} JJA · HSR≥{thr} 인 {m.sum()}점의 단계별 값 — 파랑: 최종 pred 가 HSR 보다 ASOS 에 가까움 ({better.sum()}점), 주황: 더 멂 ({(~better).sum()}점)", fontsize=10, loc="left", color=INK)
    fig.text(0.01, 0.005, "imnet 단계(HSR→imnet 출력)와 어댑터 단계(imnet 출력→최종 pred)는 서로 다른 비교다. 검은 막대 = 중앙값.", fontsize=8.5, color=SUB)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{OUT}three_stage_{y}.png", dpi=140); plt.close(fig)

for y in (2019, 2021):
    fig_move(y); fig_stage(y)
fig_adapter()
print("g(10,20,30) =", np.round(g([10, 20, 30]), 2))
print("done ->", OUT)
