"""
table_hsr_2x_scales_2020.py 가 만든 날짜별 npz 캐시(table_2020_scales/days/*.npz, sid/hour/obs/raw/m500/...)에서
2020 JJA(6~8월)만 모아 지점별 지표 CSV 생성 (plot_station_metric_maps_pm0.3.py 입력 형식). base = raw HSR, ft = MLP ft(2x) x2(500m).
GPU 불필요 (이미 계산된 캐시만 사용).
사용: python station_metrics_2020jja_from_table_cache.py
출력: inference_tools/station_eval/tables/station_metrics_2020jja_raw_vs_ft.csv
"""
import glob
import numpy as np
import pandas as pd

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
DAYS = f"{HERE}/station_eval/tables/table_2020_scales/days"
META = "/home/inhye_yoo/data/observation/raw/META_관측지점정보_20251115184700.csv"
OUT = f"{HERE}/station_eval/tables/station_metrics_2020jja_raw_vs_ft.csv"
MIN_N = 30

files = sorted(glob.glob(f"{DAYS}/2020-0[678]-*.npz"))
print(f"JJA 파일 {len(files)}개 (기대 92일)")
parts = []
for f in files:
    z = np.load(f)
    if len(z["obs"]):
        ok = np.isfinite(z["raw"])                                     # 레이더 밖 제외
        parts.append(pd.DataFrame({"sid": z["sid"][ok], "obs": z["obs"][ok], "hsr": z["raw"][ok], "pf": z["m500"][ok]}))
df = pd.concat(parts, ignore_index=True)
print(f"표본 {len(df)}")

m = pd.read_csv(META, encoding="cp949").dropna(subset=["지점", "위도", "경도"]).copy()
m["시작일"] = pd.to_datetime(m["시작일"], errors="coerce"); m["종료일"] = pd.to_datetime(m["종료일"], errors="coerce").fillna(pd.Timestamp("2100-01-01"))
ref = pd.Timestamp("2020-07-01"); coords = {}
for sid, g in m.groupby("지점"):
    g = g.sort_values("시작일"); hit = g[(g["시작일"] <= ref) & (g["종료일"] >= ref)]
    row = hit.iloc[-1] if len(hit) else g.iloc[(g["시작일"] - ref).abs().argsort().iloc[0]]
    coords[int(sid)] = (float(row["위도"]), float(row["경도"]))

rows = []
for sid, g in df.groupby("sid"):
    n = len(g)
    if n < MIN_N or sid not in coords:
        continue
    obs, hsr, pf = g["obs"].values, g["hsr"].values, g["pf"].values

    def corr(p):
        return np.corrcoef(p, obs)[0, 1] if p.std() > 0 and obs.std() > 0 else np.nan
    lat, lon = coords[sid]
    rows.append(dict(lat=lat, lon=lon, n=n,
                      bias_b=(hsr - obs).mean(), bias_f=(pf - obs).mean(),
                      mae_b=np.abs(hsr - obs).mean(), mae_f=np.abs(pf - obs).mean(),
                      corr_b=corr(hsr), corr_f=corr(pf)))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(f"저장: {OUT}  지점 {len(out)}개 (n>={MIN_N}), 표본 합 {out['n'].sum()}")
print(f"평균 |bias| raw {out.bias_b.abs().mean():.3f} -> ft {out.bias_f.abs().mean():.3f} | "
      f"MAE {out.mae_b.mean():.3f} -> {out.mae_f.mean():.3f} | corr {out.corr_b.mean():.3f} -> {out.corr_f.mean():.3f}")
