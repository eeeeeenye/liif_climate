"""
지점별 지표 지도 3x4 (Val 2021 JJA):  행 = mean bias(pred-obs) / MAE / corr,
열 = base / MLP ft(MSE) / BG(mean mu) / 변화(BG - base; 파랑 = 개선).
plot_station_metric_maps.py 의 BG 확장판.  입력: station_preds_dump_val2021_bg.py 의 npz.

표본 두 가지 (같은 npz 에서):
  recorded : 기존 lag1h csv(NaN 버전)에 기록이 있던 (시각,지점) 쌍만  -> 기존 그림(n=31533)과 같은 표본
  all      : 건조 시각까지 전 쌍(212k)                                 -> 실제 전체 모집단
사용:  python plot_station_metric_maps_bg.py [npz]
출력:  inference_tools/station_eval/figures/station_metric_maps/bg_station_metric_maps_val2021_{recorded,all}.png
env: liif (cartopy)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
NPZ = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/station_eval/preds_npz/station_preds_val2021_bg.npz"
MIN_N = 30
EXTENT = [124.2, 131.7, 32.7, 38.8]
d = np.load(NPZ)
ep_f, ep_b = int(d["ep_ft"]), int(d["ep_bg"])
MODELS = [("base", d["pb"]), (f"MLP ft (MSE, ep {ep_f})", d["pf"]), (f"BG mean mu (ep {ep_b})", d["mu"])]


def station_table(sel):
    rows = []
    for s in np.unique(d["sid"][sel]):
        m = sel & (d["sid"] == s)
        if m.sum() < MIN_N: continue
        o = d["obs"][m]; r = dict(lat=d["lat"][m][0], lon=d["lon"][m][0], n=int(m.sum()))
        for key, (_, p) in zip(["b", "f", "g"], MODELS):
            p = p[m]
            r[f"bias_{key}"] = (p - o).mean(); r[f"mae_{key}"] = np.abs(p - o).mean()
            r[f"corr_{key}"] = np.corrcoef(p, o)[0, 1] if p.std() > 0 and o.std() > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def plot(df, tag, title_extra, bias_lim, mae_lim):
    n_total = int(df["n"].sum())
    d_bias = df.bias_g.abs() - df.bias_b.abs(); d_mae = df.mae_g - df.mae_b; d_corr = df.corr_g - df.corr_b
    ROWS = [("bias", [df.bias_b, df.bias_f, df.bias_g], d_bias, np.linspace(-bias_lim, bias_lim, 12 + 1), "RdBu_r", "mm/hr", "RdBu_r",
             "|bias| change (BG-base), blue=better", "mean bias (pred-obs)"),
            ("mae", [df.mae_b, df.mae_f, df.mae_g], d_mae, np.linspace(0, mae_lim, 15 + 1), "YlOrRd", "mm/hr", "RdBu_r",
             "MAE change (BG-base), blue=better", "MAE"),
            ("corr", [df.corr_b, df.corr_f, df.corr_g], d_corr, np.arange(-1, 1.01, 0.1), "YlGnBu", "corr", "RdBu",
             "corr change (BG-base), blue=better", "corr with ASOS")]
    DL = {"bias": 0.5, "mae": 0.5, "corr": 0.5}
    fig, axes = plt.subplots(3, 4, figsize=(26, 17), subplot_kw={"projection": ccrs.PlateCarree()})
    for r, (key, vals, dv, lev, cmap, lab, dcmap, dtitle, vtitle) in enumerate(ROWS):
        for c in range(4):
            ax = axes[r, c]
            ax.set_extent(EXTENT, crs=ccrs.PlateCarree()); ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4); gl.top_labels = False; gl.right_labels = False
            if c < 3:
                v, lv, cm, ext = vals[c].clip(lev[0], lev[-1]), lev, cmap, "neither"; ttl = f"{MODELS[c][0]}: {vtitle}"
            else:
                lv = np.arange(-0.5, 0.51, 0.1); v, cm, ext, ttl = dv, dcmap, "both", dtitle
            sc = ax.scatter(df.lon, df.lat, c=v, cmap=cm, norm=BoundaryNorm(lv, plt.get_cmap(cm).N, extend=ext), s=85,
                            edgecolor="k", linewidth=0.6, transform=ccrs.PlateCarree(), zorder=5)
            ax.set_title(ttl, fontsize=13)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.72)
            cb.set_label(lab if c < 3 else ("corr" if key == "corr" else "mm/hr"), fontsize=11)
            if c < 3: cb.set_ticks(lv[::2])
    fig.suptitle(f"Val 2021 JJA, per-station metrics vs ASOS — {title_extra} (n={n_total} station-hours, station n>={MIN_N})", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = f"{HERE}/station_eval/figures/station_metric_maps/bg_station_metric_maps_val2021_{tag}.png"
    fig.savefig(out, dpi=100); plt.close(fig); print("saved", out)
    print(f"[{tag}] stations={len(df)}  평균 |bias| base {df.bias_b.abs().mean():.3f} / ft {df.bias_f.abs().mean():.3f} / BG {df.bias_g.abs().mean():.3f}"
          f" | MAE {df.mae_b.mean():.3f} / {df.mae_f.mean():.3f} / {df.mae_g.mean():.3f}"
          f" | corr {df.corr_b.mean():.3f} / {df.corr_f.mean():.3f} / {df.corr_g.mean():.3f}")
    print(f"[{tag}] BG 가 base 보다 나은 지점: |bias| {100*(d_bias<0).mean():.0f}%  MAE {100*(d_mae<0).mean():.0f}%  corr {100*(d_corr>0).mean():.0f}%"
          f" | BG 가 MLP ft 보다 나은 지점: |bias| {100*(df.bias_g.abs()<df.bias_f.abs()).mean():.0f}%  MAE {100*(df.mae_g<df.mae_f).mean():.0f}%  corr {100*(df.corr_g>df.corr_f).mean():.0f}%")


rec = d["rec"].astype(bool)
plot(station_table(rec), "recorded", "기록된 시각만(기존 그림과 같은 표본)", 2.2, 3.0)
plot(station_table(np.ones_like(rec)), "all", "건조 시각 포함 전체", 0.6, 1.5)
