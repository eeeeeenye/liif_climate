"""
plot_bias_by_category_new.py 와 같은 형식(강수 구간별 평균 bias 막대)을, 전체년/JJA 통계가 아니라
case_3h_2021-07-06_2x_multires_pow2_ft.npz (2021-07-06 케이스, 호우 창, ep393 = _ft_mlp_1e-3_lag1h 와 동일 체크포인트) 로 그린다.
  표본 = 그 창 안 72개 지점 x 8개 3시간 구간 (최대 576쌍, ASOS 기록 있는 것만) -> 표본이 아주 적어 참고용.
사용: python plot_bias_by_category_2021-07-06_case.py
출력: inference_tools/station_eval/figures/bias_by_category/bias_by_category_2021-07-06_case_vs_asos.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
NPZ = f"{HERE}/case_studies/3hourly/case_3h_2021-07-06_2x_multires_pow2_ft.npz"
EDGES = [0, 1, 2, 5, 10, 20, 50, np.inf]
LABELS = ["0–1", "1–2", "2–5", "5–10", "10–20", "20–50", "≥50"]
C_RAW, C_FT = "#2e9b2e", "#4f78e0"

z = np.load(NPZ)
raw, m500, lat, lon = z["raw"], z["m500"], z["lat500"], z["lon500"]
st_lat, st_lon, st_asos = z["st_lat"], z["st_lon"], z["st_asos"]      # (8, nstation)
print(f"ep {int(z['ep'])}, 창 {raw.shape[1]}x{raw.shape[2]}, 지점 {len(st_lat)}")


def at(field, b, slat, slon):
    r = np.abs(lat[None, :] - slat[:, None]).argmin(1); c = np.abs(lon[None, :] - slon[:, None]).argmin(1)
    return field[b, r, c]


obs_l, hsr_l, pf_l = [], [], []
for b in range(8):
    a = st_asos[b]; ok = np.isfinite(a)
    if ok.sum() == 0:
        continue
    obs_l.append(a[ok]); hsr_l.append(at(raw, b, st_lat, st_lon)[ok]); pf_l.append(at(m500, b, st_lat, st_lon)[ok])
obs, hsr, pf = np.concatenate(obs_l), np.concatenate(hsr_l), np.concatenate(pf_l)
print(f"표본(지점 x 3시간구간) {len(obs)}")


def bars(ref, series, title, xlabel, out, ylabel, pct=False):
    ok = np.isfinite(ref)
    for _, v, _ in series:
        ok &= np.isfinite(v)
    cat = np.digitize(ref, EDGES) - 1
    nb, k = len(LABELS), len(series)
    means = np.full((k, nb), np.nan); ns = np.zeros(nb, int); refm = np.full(nb, np.nan)
    for b in range(nb):
        m = ok & (cat == b); ns[b] = m.sum()
        if m.sum():
            refm[b] = ref[m].mean()
        for i, (_, v, _) in enumerate(series):
            if m.sum():
                means[i, b] = (v[m] - ref[m]).mean()
    fig, ax = plt.subplots(figsize=(19, 10))
    w = 0.34
    x = np.arange(nb)
    lo = np.nanmin(means) if np.isfinite(means).any() else -1; hi = np.nanmax(means) if np.isfinite(means).any() else 1
    span = max(hi - min(lo, 0), 0.5)
    for i, (name, _, col) in enumerate(series):
        off = (i - (k - 1) / 2) * w
        ax.bar(x + off, means[i], w, color=col, edgecolor="k", linewidth=1.2, label=name, alpha=0.95)
        for xi, mv, rm in zip(x + off, means[i], refm):
            if np.isnan(mv):
                continue
            lab = f"{mv / rm * 100:+.1f}%" if pct and rm else f"{mv:+.2f}"
            ax.text(xi, mv + (0.02 * span if mv >= 0 else -0.02 * span), lab, rotation=70,
                    ha="left" if mv >= 0 else "right", va="bottom" if mv >= 0 else "top",
                    fontsize=15, fontweight="bold", style="italic")
    ax.axhline(0, color="k", ls="--", lw=2.5)
    ymin = min(lo, 0) - 0.42 * span; ymax = max(hi, 0) + 0.18 * span
    ax.set_ylim(ymin, ymax)
    for xi, n in zip(x, ns):
        ax.text(xi, ymin + 0.02 * (ymax - ymin), f"n={n}", rotation=90, ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=20, fontweight="bold")
    ax.tick_params(axis="y", labelsize=18)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.set_xlabel(xlabel, fontsize=26, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=24, fontweight="bold")
    ax.set_title(title, fontsize=28, fontweight="bold", pad=14)
    ax.grid(axis="y", ls=":", alpha=0.6)
    ax.legend(loc="lower left", bbox_to_anchor=(0.005, 0.15), prop={"weight": "bold", "size": 17})
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print("saved", out)
    print(f"  {'구간':7s}" + "".join(f"{s[0][:18]:>20s}" for s in series) + "        n")
    for b in range(nb):
        print(f"  {LABELS[b]:7s}" + "".join(f"{means[i, b]:20.3f}" for i in range(k)) + f"  {ns[b]:8d}")


bars(obs, [("Before - Raw HSR", hsr, C_RAW), ("MLP ft (MSE)", pf, C_FT)],
     "Mean Bias Relative to ASOS by ASOS Precipitation Category (2021-07-06 case, ep393)",
     "ASOS precipitation category (mm/3h)", f"{HERE}/station_eval/figures/bias_by_category/bias_by_category_2021-07-06_case_vs_asos.png",
     "Mean Bias [Prediction - ASOS] (mm/3h)", pct=True)
