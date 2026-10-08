"""
station_preds_dump_2020.py 산출 npz(2020 전체년, sid/obs/pb(=raw HSR 더미)/pf/hsr_hr) -> 지점별 지표 CSV
(plot_station_metric_maps_pm0.3.py 입력 형식: lat, lon, n, bias_b/f, mae_b/f, corr_b/f).  base = raw HSR, ft = MLP ft(2x, lag1h).
사용: python station_metrics_from_npz_2020.py
출력: inference_tools/station_eval/tables/station_metrics_2020_raw_vs_ft.csv
"""
import numpy as np
import pandas as pd

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
NPZ = f"{HERE}/station_eval/preds_npz/station_preds_2020.npz"
META = "/home/inhye_yoo/data/observation/raw/META_관측지점정보_20251115184700.csv"
OUT = f"{HERE}/station_eval/tables/station_metrics_2020_raw_vs_ft.csv"
MIN_N = 30

z = np.load(NPZ)
df = pd.DataFrame({"sid": z["sid"], "obs": z["obs"], "hsr": z["hsr_hr"], "pf": z["pf"]})

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
