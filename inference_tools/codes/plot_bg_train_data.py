"""
BG(Bernoulli-Gamma) 파인튜닝에 쓰는 데이터 한눈에 보기 (train 2019 JJA / val 2021 JJA, fillzero obs CSV + obscrop 500m 레이더).

  레이더값은 wrapper/meta 를 거치지 않고 obs csv 의 (경도,위도) 로 nc 격자에서 직접 최근접 픽셀을 읽는다
  -> meta 정규화 버그와 무관한 독립 확인. csv 시각 라벨은 이미 lag1h 로 옮겨져 있어 nc 시각과 같은 index.

  1행: 사례 지도 (train / val 각각 '비 오는 지점이 가장 많은 시각'; 배경=레이더, 원=ASOS 같은 색)
  2행: P(ASOS 강수 | 레이더값 구간) | 강수(wet) 강도 분포 + train gamma 적합 | 레이더 vs ASOS 산점도
사용: python plot_bg_train_data.py        env: liif (cartopy)
출력: inference_tools/station_eval/figures/bg_train_data/bg_train_data_overview.png
"""
import sys
import numpy as np, pandas as pd, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"   # 한글 표시 (기본 폰트는 한글이 네모로 깨짐)
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
from losses_bernoulli_gamma import fit_climatology

M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge"
O = "/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs"
SETS = {"train": (f"{M}/HSR_2019JJA_500m_norm_obscrop.nc", f"{O}/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_fillzero_train.csv", "2019 JJA"),
        "val": (f"{M}/regrid/HSR_val_500m_norm_obscrop.nc", f"{O}/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_fillzero_val.csv", "2021 JJA")}
PMAX, WET = 74.8784, 0.05
C = {"train": "#2a78d6", "val": "#eb6834"}          # dataviz 기본 팔레트 slot1/slot2 (blue/orange)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
LEV = [0.1, 0.5, 1, 2, 5, 10, 20, 40, 80]
cmap = ListedColormap(["white"] + list(plt.cm.turbo(np.linspace(0.08, 0.98, len(LEV) - 1))) + ["#4a0a2a"])
norm = BoundaryNorm(LEV, cmap.N, extend="both")

data = {}
for name, (nc, csv, yr) in SETS.items():
    ds = xr.open_dataset(nc); lat, lon = ds.lat.values, ds.lon.values
    obs = pd.read_csv(csv); obs["time"] = pd.to_datetime(obs["time"])
    times = pd.to_datetime(ds.time.values); tpos = {t: i for i, t in enumerate(times)}
    st = obs[["x", "y"]].drop_duplicates().reset_index(drop=True)
    ri = np.abs(lat[None, :] - st.y.values[:, None]).argmin(1); ci = np.abs(lon[None, :] - st.x.values[:, None]).argmin(1)
    T = ds.sizes["time"]; rad = np.empty((T, len(st)), np.float32)
    for t0 in range(0, T, 48):                          # 시간 chunk 통째로 읽고 지점 픽셀만 추출 (포인트 인덱싱은 매우 느림)
        blk = ds.precip_norm.isel(time=slice(t0, t0 + 48)).values
        rad[t0:t0 + 48] = blk[:, ri, ci] * PMAX
    key = {(x, y): k for k, (x, y) in enumerate(zip(st.x, st.y))}
    obs["ti"] = obs["time"].map(tpos); obs["si"] = [key[(x, y)] for x, y in zip(obs.x, obs.y)]
    obs["rad"] = rad[obs.ti.values, obs.si.values]
    obs = obs[np.isfinite(obs.rad)]                       # 레이더 NaN(우하단 모서리) 제외
    data[name] = dict(ds=ds, obs=obs, lat=lat, lon=lon, yr=yr)
    print(name, len(obs), "wet frac", (obs.v >= WET).mean().round(4), "corr", np.corrcoef(obs.rad, obs.v)[0, 1].round(3))

fig = plt.figure(figsize=(17, 11.5), facecolor="white")
gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1], hspace=0.28, wspace=0.75, left=0.05, right=0.985, top=0.93, bottom=0.07)

# ---- 1행: 사례 지도
for k, name in enumerate(["train", "val"]):
    d = data[name]; o = d["obs"]
    cnt = o[o.v >= WET].groupby("ti").size(); ti = int(cnt.idxmax())
    frame = d["ds"].precip_norm.isel(time=ti).values * PMAX
    ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3], projection=ccrs.PlateCarree())
    ax.set_extent([124.2, 131.4, 32.7, 38.8], crs=ccrs.PlateCarree())
    ax.pcolormesh(d["lon"][::2], d["lat"][::2], frame[::2, ::2], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor=INK2)
    s = o[o.ti == ti]
    sc = ax.scatter(s.x, s.y, c=s.v.clip(lower=0), cmap=cmap, norm=norm, s=46, edgecolor=INK, linewidth=0.6, transform=ccrs.PlateCarree(), zorder=5)
    r = np.corrcoef(s.rad, s.v)[0, 1]
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4); gl.top_labels = False; gl.right_labels = False
    t = pd.to_datetime(d["ds"].time.values[ti])
    ax.set_title(f"{name}  {t:%Y-%m-%d %H}시 KST — 배경=레이더, 원=ASOS (같은 시각, 지점 {len(s)}개)\n이 시각 지점간 r={r:.2f}, ASOS 강수 지점 {int((s.v >= WET).sum())}개", fontsize=12, color=INK, loc="left")
cb = fig.colorbar(sc, ax=fig.axes[:2], orientation="horizontal", fraction=0.03, pad=0.06, extend="both", ticks=LEV, aspect=60)
cb.set_label("hourly precipitation (mm/h)", color=INK2)

# ---- 2행 (a): P(wet | 레이더 구간)
bins = np.array([0, 0.05, 0.5, 1, 2, 5, 10, 1e9]); lab = ["<0.05", "0.05–0.5", "0.5–1", "1–2", "2–5", "5–10", "≥10"]
ax = fig.add_subplot(gs[1, 0:2])
for name in ["train", "val"]:
    o = data[name]["obs"]; b = np.digitize(o.rad, bins) - 1
    p = np.array([(o.v[b == i] >= WET).mean() if (b == i).sum() else np.nan for i in range(len(lab))])
    ax.plot(range(len(lab)), p, "-o", color=C[name], lw=2, ms=7, mfc=C[name], mec="white", mew=1.5, label=f"{name} ({data[name]['yr']})")
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=30, fontsize=9)
ax.set_ylim(0, 1.02); ax.set_xlabel("레이더 값 (mm/h, 지점 최근접 픽셀)", color=INK2); ax.set_ylabel("P(ASOS 강수 ≥ 0.05)", color=INK2)
ax.set_title("레이더가 이 정도일 때 ASOS는 얼마나 자주 비?", fontsize=12, loc="left", color=INK)
ax.legend(frameon=False, loc="upper left")

# ---- 2행 (b): 강수 강도 분포 + gamma 적합
ax = fig.add_subplot(gs[1, 2:4])
edges = np.logspace(np.log10(0.05), np.log10(80), 26); mid = np.sqrt(edges[:-1] * edges[1:])
for name in ["train", "val"]:
    w = data[name]["obs"].v; w = w[w >= WET]
    h, _ = np.histogram(np.log(w), bins=np.log(edges), density=True)
    ax.step(mid, h, where="mid", color=C[name], lw=2, label=f"{name} wet 관측 n={len(w):,}")
o = data["train"]["obs"]; w = o.v[o.v >= WET].values
p, a, b = fit_climatology(len(o), len(w), w.sum(), np.log(np.clip(w, WET, None)).sum())
from scipy.stats import gamma as G
xs = np.logspace(np.log10(0.05), np.log10(80), 300)
ax.plot(xs, xs * G.pdf(xs, a, scale=b), "--", color=INK2, lw=1.6, label=f"train gamma 적합 α={a:.2f}, β={b:.2f}mm")
ax.set_xscale("log"); ax.set_xlabel("ASOS wet 강도 (mm/h, log)", color=INK2); ax.set_ylabel("밀도 (log-강도 기준)", color=INK2)
ax.set_title("wet 강도 분포 vs gamma 가정", fontsize=12, loc="left", color=INK); ax.legend(frameon=False, fontsize=9)

# ---- 2행 (c): 산점도
ax = fig.add_subplot(gs[1, 4:6])
rng = np.random.default_rng(0)
for name in ["val", "train"]:
    o = data[name]["obs"]; m = (o.rad >= WET) | (o.v >= WET)
    s = o[m].sample(min(m.sum(), 6000), random_state=0)
    ax.scatter(s.rad.clip(lower=0.03), s.v.clip(lower=0.03), s=9, alpha=0.35, color=C[name], edgecolor="none",
               label=f"{name}  r={np.corrcoef(o.rad, o.v)[0, 1]:.2f} (전 지점·시각)")
ax.plot([0.03, 100], [0.03, 100], color=INK2, lw=1, ls=":")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.03, 100); ax.set_ylim(0.03, 100)
ax.set_xlabel("레이더 (mm/h, 0 → 0.03 로 표시)", color=INK2); ax.set_ylabel("ASOS (mm/h)", color=INK2)
ax.set_title("지점·시각 쌍 (둘 중 하나라도 ≥0.05)", fontsize=12, loc="left", color=INK); ax.legend(frameon=False, fontsize=9, loc="upper left")

for a_ in fig.axes[:2]: pass
for ax in fig.axes[3:]:
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2)

fig.suptitle("BG 파인튜닝 학습 데이터 (fillzero obs, lag1h, obscrop): train 2019 JJA / val 2021 JJA", fontsize=15, color=INK, y=0.985)
out = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/station_eval/figures/bg_train_data/bg_train_data_overview.png"
fig.savefig(out, dpi=90, bbox_inches="tight"); print("saved", out)
