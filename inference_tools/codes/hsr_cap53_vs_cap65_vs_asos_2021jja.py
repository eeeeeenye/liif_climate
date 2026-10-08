"""
2021 JJA 시간별: HSR cap53(학습 데이터) vs HSR cap65 가 ASOS 와 얼마나 차이나는지 비교.
  - 쌍: station_preds_val2021_8x_ft_2019_2020_wmae_epoch-best-loss.npz 의 (지점,시각) 31,533 쌍 (tid=val 시간 인덱스, lat/lon, obs=ASOS, hsr_hr=cap53 HSR[mm/h]).
        lag1h 정렬은 obs csv 에 이미 반영(ASOS[T] <-> radar[T-1h]).
  - cap65 : merge/HSR_2019_2023_JJA_merged_kst_hourly_2881x2305_cap65.nc 의 같은 시각(병합본 인덱스 = 4416 + tid), 지점 최근접 500m 격자.
            결측(-9.99e8)/NaN 은 0 으로 (정규화 파일 만들 때와 같은 처리).
  - 출력 : inference_tools/station_eval/preds_npz/hsr_cap53_cap65_pairs_2021jja.npz
           inference_tools/station_eval/figures/hexbin_hsr_preds_vs_asos/hexbin_hsr_cap53_vs_cap65_vs_asos_2021jja_nofilter.png
env: liif
"""
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
NPZ = f"{HERE}/station_eval/preds_npz/station_preds_val2021_8x_ft_2019_2020_wmae_epoch-best-loss.npz"
M65 = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/HSR_2019_2023_JJA_merged_kst_hourly_2881x2305_cap65.nc"
OFF = 4416   # 병합본에서 2021-06-01 00 KST 의 인덱스 (확인함)
OUT_NPZ = f"{HERE}/station_eval/preds_npz/hsr_cap53_cap65_pairs_2021jja.npz"
OUT_PNG = f"{HERE}/station_eval/figures/hexbin_hsr_preds_vs_asos/hexbin_hsr_cap53_vs_cap65_vs_asos_2021jja_nofilter.png"
LIM, GRID = 80, 60

d = np.load(NPZ)
tid, lat, lon, obs, h53 = d["tid"], d["lat"], d["lon"], d["obs"], d["hsr_hr"]
m = Dataset(M65)
LAT, LON = m["lat"][:], m["lon"][:]
r = np.abs(LAT[None, :] - lat[:, None]).argmin(1)
c = np.abs(LON[None, :] - lon[:, None]).argmin(1)
h65 = np.zeros(len(obs), dtype=np.float32)
for k, t in enumerate(np.unique(tid)):
    sel = np.where(tid == t)[0]
    sl = np.ma.filled(m["precip"][OFF + int(t)], np.nan)
    v = sl[r[sel], c[sel]]
    h65[sel] = np.where(np.isfinite(v) & (v > -1e6), v, 0.0)
    if (k + 1) % 400 == 0:
        print(f"  {k + 1}/{len(np.unique(tid))} 시각", flush=True)
np.savez(OUT_NPZ, tid=tid, lat=lat, lon=lon, obs=obs, hsr_cap53=h53, hsr_cap65=h65)

ok = np.isclose(h53, h65, atol=0.05)
print(f"cap53 와 cap65 가 같은 쌍(차이<0.05): {ok.mean() * 100:.2f}%  (다른 쌍 {int((~ok).sum())}개, 그중 cap65>cap53 {int((h65 > h53 + 0.05).sum())}개)")


def stats(x, y):
    b, a = np.polyfit(x, y, 1)
    return dict(r=np.corrcoef(x, y)[0, 1], bias=np.mean(x - y), rmse=np.sqrt(np.mean((x - y) ** 2)),
                mae=np.abs(x - y).mean(), ratio=x.mean() / y.mean(), a=a, b=b)


S = {"cap53": stats(h53, obs), "cap65": stats(h65, obs)}
for k, s in S.items():
    print(k, {kk: round(float(vv), 3) for kk, vv in s.items()})
print("\nASOS 강도   n     ASOS   cap53   cap65   (cap53/ASOS, cap65/ASOS)")
for lo, hi in [(0, .1), (.1, 5), (5, 10), (10, 20), (20, 30), (30, 1e9)]:
    mk = (obs >= lo) & (obs < hi)
    a = obs[mk].mean()
    print(f"{lo}-{hi:<7g} {mk.sum():6d} {a:6.2f} {h53[mk].mean():6.2f} {h65[mk].mean():6.2f}   " +
          (f"{h53[mk].mean() / a:.2f}, {h65[mk].mean() / a:.2f}" if a > 0 else "-"))
mk = obs >= 20
print(f"\nASOS>=20 MAE: cap53 {np.abs(h53[mk] - obs[mk]).mean():.2f}  cap65 {np.abs(h65[mk] - obs[mk]).mean():.2f}")
print(f"최대값: cap53 {h53.max():.1f}  cap65 {h65.max():.1f}  ASOS {obs.max():.1f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=130)
vmax = None
for ax, x, name in [(axes[0], h53, "HSR cap53 (current training data)"), (axes[1], h65, "HSR cap65")]:
    s = S["cap53" if x is h53 else "cap65"]
    hb = ax.hexbin(x, obs, gridsize=GRID, extent=(0, LIM, 0, LIM), mincnt=1, norm=LogNorm(1, 2e4), cmap="viridis")
    ax.plot([0, LIM], [0, LIM], color="0.5", ls="--", lw=1, label="1:1")
    xx = np.array([0, LIM]); ax.plot(xx, s["a"] + s["b"] * xx, color="red", lw=1.6, label=f"OLS: y = {s['a']:.2f} + {s['b']:.2f}x")
    ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel(f"{name} hourly precip (mm/h)"); ax.set_ylabel("ASOS hourly precip (mm/h)")
    ax.set_title(f"{name} vs ASOS\nn={len(x)}  r={s['r']:.2f}  bias={s['bias']:.2f}  RMSE={s['rmse']:.2f}  ratio={s['ratio']:.2f}", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
fig.suptitle("2021 JJA  hourly precip  (nearest 500m grid, all hours, no filter)", fontsize=13)
fig.tight_layout(rect=(0, 0, 0.92, 1))
cax = fig.add_axes([0.94, 0.15, 0.015, 0.7]); cb = fig.colorbar(hb, cax=cax); cb.set_label("count (log)")
fig.savefig(OUT_PNG); print("saved", OUT_PNG)
