"""
JJA 밖 2021 (1~5월, 9~11월) 의 HSR 시간별 자료를 기존 val(HSR_val_*_obscrop.nc) 과 똑같은 방식으로 날짜별 nc 로 만든다.
  원자료 = hsr_nc_raw_all_kst (DBZ_CAP=53, KST 시간별 mm/h; 2021-07-06 06시가 val 과 최대 차 2e-6 으로 일치함을 확인)
  1) 24시간 모아 /PMAX(74.8784) 정규화, 결측 -> _FillValue(-999e6)   (cdo expr (x-0)/(74.8784-0) 와 동일, clip 없음)
  2) cdo remapbil,grid_2x.txt (전체 도메인, 학습 입력과 동일) -> HR/LR 을 val obscrop 격자(lat/lon)로 크롭
  출력: <OUTDIR>/<날짜>_hr.nc (500m, 1326x1234), <날짜>_lr.nc (1km, 663x617), 변수 precip_norm.  24시간이 다 있는 날만 만든다.
사용:  python prep_hsr_2021_nonjja.py 2021-01 2021-02 ... | 2021-07-06 ...     (월 'YYYY-MM' 또는 날짜 'YYYY-MM-DD', 이미 있으면 건너뜀)
       PROCS=6 (병렬 수, 기본 6)
env: liif (netCDF4, xarray), cdo = cdo_env
"""
import os, sys, glob, subprocess, shutil
from multiprocessing import Pool
import numpy as np
import xarray as xr
from netCDF4 import Dataset

R = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc"
RAW = f"{R}/hsr_nc_raw_all_kst"
GRID = f"{R}/merge/grid_2x.txt"
CDO = "/home/inhye_yoo/anaconda3/envs/cdo_env/bin/cdo"
OUTDIR = os.environ.get("OUTDIR", f"{R}/merge/regrid/hsr2021_nonjja")
TMP = os.environ.get("TMP", "/tmp/claude-1018/-home-inhye-yoo/360583e4-d141-44c9-9858-78608ac036d0/scratchpad/prep_tmp")
PMAX, FILL = 74.8784, np.float32(-999000000.0)
os.makedirs(OUTDIR, exist_ok=True); os.makedirs(TMP, exist_ok=True)

_hr = xr.open_dataset(f"{R}/merge/regrid/HSR_val_500m_norm_obscrop.nc"); _lr = xr.open_dataset(f"{R}/merge/regrid/HSR_val_norm_2x_obscrop.nc")
HR_LAT, HR_LON, LR_LAT, LR_LON = _hr.lat.values, _hr.lon.values, _lr.lat.values, _lr.lon.values
_first = sorted(glob.glob(f"{RAW}/HSR_2021*.nc"))[0]
with Dataset(_first) as d:
    LAT, LON = np.asarray(d.variables["lat"][:], "f4"), np.asarray(d.variables["lon"][:], "f4")


def read_hour(fp):
    with Dataset(fp) as d:
        a = np.ma.filled(d.variables["precip"][0].astype("f4"), np.nan)
    return a


def do_day(date):
    hr_out, lr_out = f"{OUTDIR}/{date}_hr.nc", f"{OUTDIR}/{date}_lr.nc"
    if os.path.exists(hr_out) and os.path.exists(lr_out):
        return date, "skip"
    ymd = date.replace("-", "")
    files = [f"{RAW}/HSR_{ymd}{h:02d}_hourly_precip.nc" for h in range(24)]
    if not all(os.path.exists(f) for f in files):
        return date, "incomplete"
    P = np.stack([read_hour(f) for f in files])                                  # (24,ny,nx) NaN=결측
    P[P < -1e8] = np.nan
    y = np.where(np.isfinite(P), P / PMAX, FILL).astype(np.float32)
    times = np.datetime64(date + "T00:00") + np.arange(24) * np.timedelta64(1, "h")
    full = xr.Dataset({"precip_norm": (("time", "lat", "lon"), y, {"_FillValue": FILL, "missing_value": FILL})},
                      coords={"time": times, "lat": LAT, "lon": LON})
    full.lat.attrs.update(units="degrees_north"); full.lon.attrs.update(units="degrees_east")
    f_full, f_2x = f"{TMP}/{date}_full.nc", f"{TMP}/{date}_2x.nc"
    full.to_netcdf(f_full)
    subprocess.run([CDO, "-s", f"remapbil,{GRID}", f_full, f_2x], check=True)
    hr = full.sel(lat=HR_LAT, lon=HR_LON, method="nearest")
    lr = xr.open_dataset(f_2x).sel(lat=LR_LAT, lon=LR_LON, method="nearest")
    assert np.abs(hr.lat.values - HR_LAT).max() < 1e-4 and np.abs(lr.lat.values - LR_LAT).max() < 1e-4, date
    assert np.abs(hr.lon.values - HR_LON).max() < 1e-4 and np.abs(lr.lon.values - LR_LON).max() < 1e-4, date
    enc = {"precip_norm": {"zlib": True, "complevel": 4}}
    hr.to_netcdf(hr_out + ".tmp", encoding=enc); lr.to_netcdf(lr_out + ".tmp", encoding=enc)
    os.replace(hr_out + ".tmp", hr_out); os.replace(lr_out + ".tmp", lr_out)
    os.remove(f_full); os.remove(f_2x)
    return date, "ok"


if __name__ == "__main__":
    dates = []
    for a in sys.argv[1:]:
        if len(a) == 7:                                                          # YYYY-MM
            ms = sorted({os.path.basename(f)[4:12] for f in glob.glob(f"{RAW}/HSR_{a.replace('-', '')}*.nc")})
            dates += [f"{m[:4]}-{m[4:6]}-{m[6:]}" for m in ms]
        else:
            dates.append(a)
    print(f"{len(dates)}일 처리 -> {OUTDIR}", flush=True)
    n = {"ok": 0, "skip": 0, "incomplete": 0}
    with Pool(int(os.environ.get("PROCS", "6"))) as p:
        for i, (d, st) in enumerate(p.imap_unordered(do_day, dates)):
            n[st] += 1
            if st != "skip":
                print(f"  {d} {st}  ({i + 1}/{len(dates)})", flush=True)
    print("done", n)
    if os.environ.get("MARK_DONE") == "1":
        open(f"{OUTDIR}/PREP_DONE", "w").close()                                     # 추론 워커가 큐 종료를 판단하는 표지
