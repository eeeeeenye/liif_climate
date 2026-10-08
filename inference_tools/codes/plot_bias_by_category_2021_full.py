"""
plot_bias_by_category_new.py 의 2021 전체년판 (원래는 Val 2021 JJA npz 만 사용).
table_hsr_2x_scales_2021.py 가 만든 날짜별 캐시(table_2021_scales/days/*.npz, sid/hour/obs/raw/m500/...)에서
1~11월(11/24까지, 12월 미준비) 전부 모아 강수 구간별 평균 bias 막대 그림.  GPU 불필요(캐시만 사용).
사용: python plot_bias_by_category_2021_full.py
출력: inference_tools/station_eval/figures/bias_by_category/bias_by_category_2021_full_vs_asos.png
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
DAYS = f"{HERE}/station_eval/tables/table_2021_scales/days"
EDGES = [0, 1, 2, 5, 10, 20, 50, np.inf]
LABELS = ["0–1", "1–2", "2–5", "5–10", "10–20", "20–50", "≥50"]
C_RAW, C_FT = "#2e9b2e", "#4f78e0"

files = sorted(glob.glob(f"{DAYS}/2021-*.npz"))
print(f"파일 {len(files)}개")
obs_l, hsr_l, pf_l = [], [], []
for f in files:
    z = np.load(f)
    if len(z["obs"]):
        ok = np.isfinite(z["raw"])                      # 레이더 밖 제외 (원본 스크립트와 동일 취지)
        obs_l.append(z["obs"][ok]); hsr_l.append(z["raw"][ok]); pf_l.append(z["m500"][ok])
obs, hsr, pf = np.concatenate(obs_l), np.concatenate(hsr_l), np.concatenate(pf_l)
print(f"표본 {len(obs)}")


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
    lo = np.nanmin(means); hi = np.nanmax(means)
    span = max(hi - min(lo, 0), 0.5)
    for i, (name, _, col) in enumerate(series):
        off = (i - (k - 1) / 2) * w
        ax.bar(x + off, means[i], w, color=col, edgecolor="k", linewidth=1.2, label=name, alpha=0.95)
        for xi, mv, rm in zip(x + off, means[i], refm):
            if np.isnan(mv):
                continue
            lab = f"{mv / rm * 100:+.1f}%" if pct else f"{mv:+.2f}"
            ax.text(xi, mv + (0.02 * span if mv >= 0 else -0.02 * span), lab, rotation=70,
                    ha="left" if mv >= 0 else "right", va="bottom" if mv >= 0 else "top",
                    fontsize=15, fontweight="bold", style="italic")
    ax.axhline(0, color="k", ls="--", lw=2.5)
    ymin = min(lo, 0) - 0.42 * span; ymax = max(hi, 0) + 0.18 * span
    ax.set_ylim(ymin, ymax)
    for xi, n in zip(x, ns):
        ax.text(xi, ymin + 0.02 * (ymax - ymin), f"n={n:,}", rotation=90, ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=20, fontweight="bold")
    ax.tick_params(axis="y", labelsize=18)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.set_xlabel(xlabel, fontsize=26, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=24, fontweight="bold")
    ax.set_title(title, fontsize=30, fontweight="bold", pad=14)
    ax.grid(axis="y", ls=":", alpha=0.6)
    ax.legend(loc="lower left", bbox_to_anchor=(0.005, 0.15), prop={"weight": "bold", "size": 17})
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print("saved", out)
    print(f"  {'구간':7s}" + "".join(f"{s[0][:18]:>20s}" for s in series) + "        n")
    for b in range(nb):
        print(f"  {LABELS[b]:7s}" + "".join(f"{means[i, b]:20.3f}" for i in range(k)) + f"  {ns[b]:8d}" +
              (f"   (ref 평균 {refm[b]:.3f} -> " + ", ".join(f"{means[i, b] / refm[b] * 100:+.1f}%" for i in range(k)) + ")" if pct else ""))


bars(obs, [("Before - Raw HSR", hsr, C_RAW), ("MLP ft (MSE)", pf, C_FT)],
     "Mean Bias Relative to ASOS by ASOS Precipitation Category (2021, Full Year)",
     "ASOS precipitation category (mm/hour)", f"{HERE}/station_eval/figures/bias_by_category/bias_by_category_2021_full_vs_asos.png",
     "Mean Bias [Prediction - ASOS] (mm/hour)", pct=True)
