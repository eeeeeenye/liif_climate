"""
MLP adapter transfer function 위에 "IMERG(p_k) bin별 ASOS mean/median"을 겹쳐 그린다.

목적: 고강수 영역에서 adapter 기울기가 완만해지는 게 activation function 자체의 문제인지,
아니면 IMERG(p_k)-ASOS 관계 자체가 원래 거기서 완만해지고(+ sample 희소성) 그걸 MLP가 학습한
결과인지 구분하기 위함.

- x축(adapter input) = per-direction imnet output p_k (mm/day)  [adapter_fold_phenomenon.py 와 동일]
- 각 row 의 asos 값을 p1,p2,p3,p4 네 개 모두에 동일하게 짝지어(=histogram 과 동일한 p_k 모집단) 사용.
  (진짜 per-direction ASOS는 없으므로, "이 p_k 값이 나온 station-day 의 asos" 라는 근사.)
- IMERG(p_k) 를 bin 으로 나눠 각 bin 의 asos mean/median 을 transfer function 위에 겹쳐 그림.
- 진짜 piecewise(segmented) linear regression: breakpoint c 를 임의로 정하지 않고 SSE 를 최소화하는
  값을 grid search 로 데이터에서 추정, c 에서 연속(jump 없음)인 2-구간 직선을 원점 통과로 fit.
  (이전 버전은 임의 threshold 로 자른 뒤 독립적으로 2번 회귀한 것이라 진짜 piecewise regression이
  아니었음 — 이번에 수정.)

출력: /home/inhye_yoo/adapter_fold_vs_asos_binning.png
env:  /home/inhye_yoo/anaconda3/envs/liif/bin/python
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "Noto Sans CJK JP"

V = 506.01995849609375
CSV = "/home/inhye_yoo/data/observation/IMERG_ASOS_Paired/codes/inference_results/station_base_pred_2000_2002_ftmlp_epochbest.csv"
S = "/home/inhye_yoo/deeplearning/inr/liif/save"

RUNS = [
    ("old  _ft_mlp (folded, ep3136)", f"{S}/_climate_edsr_topo_80x100_topo_precip_minmax_ft_mlp/epoch-best.pth", nn.ReLU, "gray"),
    ("new  ReLU flipped (best)",       f"{S}/_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-3_flipped/epoch-best.pth", nn.ReLU, "seagreen"),
    ("new  tanh flipped (best)",       f"{S}/_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-3_flipped_tanh/epoch-best.pth", nn.Tanh, "royalblue"),
    ("new  linear flipped (best)",     f"{S}/_climate_edsr_imerg_asos_ft_adapter_80x100_linear_flipped/epoch-best.pth", None, "darkorange"),
]

d = pd.read_csv(CSV)

# p_k 와 asos 를 짝지어 long-format 으로 (histogram 과 동일한 p_k 모집단)
pk = d[["p1", "p2", "p3", "p4"]].values.ravel()
asos_rep = np.repeat(d["asos"].values, 4)
mask = pk > 1  # 원본 histogram 과 동일하게 1mm 미만 제외
pk, asos_rep = pk[mask], asos_rep[mask]

# ---------- IMERG(p_k) bin 별 ASOS mean/median ----------
edges = np.arange(0, 420, 20)
edges = edges[edges <= pk.max()]
edges = np.append(edges, [pk.max() + 1])
bin_idx = np.digitize(pk, edges) - 1
rows = []
for i in range(len(edges) - 1):
    sel = bin_idx == i
    n = sel.sum()
    if n < 5:
        continue
    rows.append(dict(lo=edges[i], hi=edges[i + 1], n=n,
                      pk_mean=pk[sel].mean(),
                      asos_mean=asos_rep[sel].mean(),
                      asos_median=np.median(asos_rep[sel])))
binned = pd.DataFrame(rows)
print("=== IMERG(p_k) bin 별 ASOS ===")
print(binned.to_string(index=False))

# ---------- true piecewise (segmented) linear regression ----------
# model: asos = s1 * min(pk,c) + s2 * max(pk-c, 0)
#   - 원점을 지나고(c 이전엔 asos=s1*pk), breakpoint c 에서 연속으로 이어짐(jump 없음)
#   - c 는 고정하지 않고 SSE 를 최소화하는 값을 grid search 로 데이터에서 직접 추정
def fit_two_segment(x, y, c):
    b1 = np.minimum(x, c)
    b2 = np.maximum(x - c, 0.0)
    A = np.stack([b1, b2], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    sse = float(np.sum((y - A @ coef) ** 2))
    return coef, sse

def slope_no_intercept(x, y):
    return float((x @ y) / (x @ x))

candidates = np.arange(40, pk.max() - 20, 2.0)
results = [(c, *fit_two_segment(pk, asos_rep, c)) for c in candidates]
c_star, coef_star, sse_star = min(results, key=lambda r: r[2])
s1_star, s2_star = coef_star

s_single = slope_no_intercept(pk, asos_rep)
sse_single = float(np.sum((asos_rep - s_single * pk) ** 2))

print("\n=== single-line fit (no break): slope=%.3f  SSE=%.3e ===" % (s_single, sse_single))
print("=== true piecewise regression (breakpoint estimated from data, continuous at c) ===")
print(f"breakpoint c* = {c_star:.1f} mm/day   slope1(p_k<c*)={s1_star:.3f}   "
      f"slope2(p_k>=c*)={s2_star:.3f}   SSE={sse_star:.3e}  "
      f"(SSE reduction vs single line: {100*(1-sse_star/sse_single):.1f}%)")

# breakpoint 후보에 따라 SSE 가 얼마나 뚜렷하게 꺾이는지도 같이 출력 (u자형이어야 신뢰할 만함)
print("\nc(mm) -> SSE (sanity check, u-shape 이어야 breakpoint 가 의미있음):")
for c, coef, sse in results[::10]:
    print(f"  c={c:6.1f}  SSE={sse:.3e}")

# ---------- adapter transfer function (원본과 동일) ----------
def load_adapter(path, act):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]["sd"]
    ad = {k.replace("add_layer.asos_adapter.", ""): v
          for k, v in sd.items() if "add_layer.asos_adapter" in k}
    if act is None:
        m = nn.Sequential(nn.Linear(1, 1, bias=False))
    else:
        dims = [1, 32, 16, 32]; L = []
        for a, b in zip(dims[:-1], dims[1:]):
            L += [nn.Linear(a, b), act()]
        L += [nn.Linear(dims[-1], 1)]
        m = nn.Sequential(*L)
    m.load_state_dict(ad)
    m.eval()
    return m, ck.get("epoch")

x = torch.linspace(0, 1.3, 4000).view(-1, 1)
xin = x.view(-1).numpy() * V

fig, (ax, axh) = plt.subplots(2, 1, figsize=(10, 8.5), height_ratios=[4, 1], sharex=True)
ax.plot([0, 520], [0, 520], color="k", alpha=.3, lw=1, label="y = x (no correction)")
ax.axvspan(506, 650, color="0.9")
for name, path, act, c in RUNS:
    try:
        m, ep = load_adapter(path, act)
    except Exception as e:
        print(name, "skip:", e); continue
    with torch.no_grad():
        y = m(x).view(-1).numpy() * V
    yc = np.clip(y, 0, None)
    lw = 2.0 if name.startswith("old") else 2.6
    ax.plot(xin, yc, color=c, lw=lw, label=f"{name} (ep {ep})")

# IMERG(p_k) bin 별 실측 ASOS mean/median 겹쳐 그리기
ax.plot(binned["pk_mean"], binned["asos_mean"], "o-", color="crimson", lw=2.2, ms=6,
        label="observed ASOS mean per IMERG(p_k) bin", zorder=5)
ax.plot(binned["pk_mean"], binned["asos_median"], "s--", color="crimson", lw=1.4, ms=5,
        alpha=.6, label="observed ASOS median per IMERG(p_k) bin", zorder=5)
for _, r in binned.iterrows():
    ax.annotate(f"n={int(r['n'])}", (r["pk_mean"], r["asos_mean"]),
                textcoords="offset points", xytext=(0, 8), fontsize=6.5, color="crimson", ha="center")

# 진짜 piecewise regression (breakpoint 데이터에서 추정, c 에서 연속) 겹쳐 그리기
xs1 = np.array([0, c_star])
ys1 = s1_star * xs1
xs2 = np.array([c_star, pk.max()])
ys2 = s1_star * c_star + s2_star * (xs2 - c_star)
ax.plot(xs1, ys1, color="magenta", lw=2.2, ls="-",
        label=f"piecewise fit: slope1={s1_star:.2f} (p_k<{c_star:.0f})")
ax.plot(xs2, ys2, color="magenta", lw=2.2, ls=":",
        label=f"piecewise fit: slope2={s2_star:.2f} (p_k>={c_star:.0f})")
ax.axvline(c_star, color="magenta", alpha=.3, lw=1)
ax.annotate(f"c*={c_star:.0f}mm", (c_star, 5), color="magenta", fontsize=8, ha="center")

ax.text(508, 30, "extrapolation\n(train max ~382mm)", fontsize=8, color="0.4")
ax.set_xlim(0, 650); ax.set_ylim(-20, 520)
ax.set_ylabel("output / ASOS (mm/day)")
ax.set_title("Adapter transfer function vs. observed IMERG(p_k)-ASOS relationship")
ax.legend(loc="upper left", fontsize=8)

axh.hist(pk, bins=np.arange(0, 650, 5), color="navy", alpha=.6)
axh.set_yscale("log")
axh.set_xlabel("adapter input = per-direction imnet output p_k (mm/day)")
axh.set_ylabel("# p_k (log)")
axh.text(360, axh.get_ylim()[1] * .2,
         f"training p_k:  >=150:{(pk>=150).sum()}  >=200:{(pk>=200).sum()}  >=326:{(pk>=326).sum()}  max:{pk.max():.0f}",
         fontsize=9, color="navy")

plt.tight_layout()
out = "/home/inhye_yoo/adapter_fold_vs_asos_binning.png"
plt.savefig(out, dpi=135)
print("\nsaved", out)
