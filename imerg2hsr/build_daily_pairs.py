"""
IMERG(입력, 0.1deg 일누적) -> HSR(정답, ~500m 일누적) LIIF 학습용 짝 데이터 생성.

- HR (HSR): /home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/hsr_nc_raw_all_utc_cap63
  의 UTC 시간별 nc(HSR_<YYYYMMDDHH>_hourly_precip.nc, 이미 UTC 변환 + DBZ cap 적용됨,
  build_pairs_daily_utc_jja_2019_2021.py 와 동일 소스)에서
  24시간이 모두 있는 UTC 날짜만 사용해 일누적. 결측/음수(no-echo)는 0(무강수)으로 처리.
  실제로는 다운로드가 장마철 위주라 2019/2020/2021 6~8월(JJA) 273일만 존재.
  이후 IMERG 도메인(lon/lat 범위)으로 crop해서 LR/HR이 같은 지리 영역을 나타내도록 맞춤
  (LIIF의 sr-implicit-paired wrapper는 LR crop 위치를 비율로 환산해 HR을 자르므로,
   두 그리드가 같은 영역을 커버해야 함).
- LR (IMERG): imerg_precip_korea_daily_01deg_1998_202504_noleaf.nc (네이티브 UTC 일,
  기존 IMERG self-SR 파이프라인과 동일 파일)에서 HR과 같은 날짜만 subset.
- split: train = 2019, 2020 / val = 2021 (radar_HSR/hsr_hourly_precip_nc/merge/codes/split.sh 와
  동일 관례, data leakage 방지 위해 train만으로 정규화 max 계산).
- 정규화: min=0 고정, max = train 구간 HR/LR 중 큰 값 하나를 공유(physical_norm).
  기존 IMERG self-SR(같은 물리량, 다른 해상도)과 동일하게 LR/HR에 같은 max를 적용해
  두 그리드가 같은 정규화 공간(mm 단위 보존)에 있도록 함.

출력 (imerg2hsr/data/):
  HSR_daily_kor_crop_raw.nc, IMERG_daily_kor_matched_raw.nc   : mm/day 원본 (QC/archival)
  LR_train_norm.nc / LR_val_norm.nc                            : IMERG, var=precipitation, 0~1
  HR_train_norm.nc / HR_val_norm.nc                            : HSR,   var=precipitation, 0~1
  norm_meta.json                                               : gmax, 날짜 목록, 그리드 정보

env: /home/inhye_yoo/anaconda3/envs/liif/bin/python (or base, netCDF4/xarray/numpy/pandas 필요)
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

HSR_UTC_DIR = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/hsr_nc_raw_all_utc_cap63"
IMERG_NC = "/home/inhye_yoo/data/IMERG/Imerg_raw/kor_data/imerg_precip_korea_daily_01deg_1998_202504_noleaf.nc"
OUT_DIR = "/home/inhye_yoo/deeplearning/inr/liif/imerg2hsr/data"

TRAIN_YEARS = {2019, 2020}
VAL_YEARS = {2021}

FILE_PAT = re.compile(r"HSR_(\d{8})(\d{2})_hourly_precip\.nc$")


def find_full_utc_days(limit=None):
    files = sorted(glob.glob(os.path.join(HSR_UTC_DIR, "HSR_*_hourly_precip.nc")))
    by_day = {}
    for f in files:
        m = FILE_PAT.search(os.path.basename(f))
        if not m:
            continue
        ymd, hh = m.group(1), int(m.group(2))
        by_day.setdefault(ymd, {})[hh] = f
    full_days = sorted(ymd for ymd, hrs in by_day.items() if len(hrs) == 24)
    if limit is not None:
        full_days = full_days[:limit]
    return full_days, by_day


def load_hr_grid():
    any_file = sorted(glob.glob(os.path.join(HSR_UTC_DIR, "HSR_*_hourly_precip.nc")))[0]
    nc = Dataset(any_file)
    hlat = np.asarray(nc["lat"][:], dtype=np.float64)
    hlon = np.asarray(nc["lon"][:], dtype=np.float64)
    nc.close()
    return hlat, hlon


def daily_hr_field(files_for_day, iy0, iy1, ix0, ix1):
    """24시간 파일을 합산해 crop된 (Hc, Wc) 일누적 필드 반환 (mm/day)."""
    acc = None
    for hh in range(24):
        fp = files_for_day[hh]
        nc = Dataset(fp)
        arr = np.asarray(nc["precip"][0], dtype=np.float32)
        nc.close()
        arr = arr[iy0:iy1, ix0:ix1]
        arr = np.where(np.isfinite(arr) & (arr >= 0), arr, 0.0)
        acc = arr.copy() if acc is None else acc + arr
    return acc


def main(limit_days=None):
    os.makedirs(OUT_DIR, exist_ok=True)

    full_days, by_day = find_full_utc_days(limit=limit_days)
    print(f"[HSR] 24h 모두 있는 UTC 날짜 {len(full_days)}개 "
          f"({full_days[0]} ~ {full_days[-1]})")

    hlat, hlon = load_hr_grid()

    im = xr.open_dataset(IMERG_NC)
    ilat = im["lat"].values
    ilon = im["lon"].values
    lat_lo, lat_hi = float(ilat.min()), float(ilat.max())
    lon_lo, lon_hi = float(ilon.min()), float(ilon.max())
    print(f"[IMERG] LR 도메인 lat [{lat_lo},{lat_hi}] lon [{lon_lo},{lon_hi}] "
          f"grid {ilat.size}x{ilon.size}")

    lat_mask = (hlat >= lat_lo) & (hlat <= lat_hi)
    lon_mask = (hlon >= lon_lo) & (hlon <= lon_hi)
    iy0, iy1 = np.where(lat_mask)[0][[0, -1]]
    iy1 += 1
    ix0, ix1 = np.where(lon_mask)[0][[0, -1]]
    ix1 += 1
    hlat_crop = hlat[iy0:iy1]
    hlon_crop = hlon[ix0:ix1]
    print(f"[HSR] crop grid {hlat_crop.size}x{hlon_crop.size} "
          f"(lat[{hlat_crop[0]:.3f},{hlat_crop[-1]:.3f}] "
          f"lon[{hlon_crop[0]:.3f},{hlon_crop[-1]:.3f}])")

    dates = pd.to_datetime(full_days, format="%Y%m%d")
    itime = pd.DatetimeIndex(im["time"].values).normalize()
    idx_by_date = {t: k for k, t in enumerate(itime)}

    missing = [d for d in dates if d not in idx_by_date]
    if missing:
        print(f"[경고] IMERG에 없는 날짜 {len(missing)}개 -> 제외: {missing[:5]}...")
    keep_mask = np.array([d in idx_by_date for d in dates])
    dates = dates[keep_mask]
    full_days = [d for d, k in zip(full_days, keep_mask) if k]
    print(f"[매칭] 최종 사용 날짜 {len(dates)}개")

    n = len(dates)
    hr = np.empty((n, hlat_crop.size, hlon_crop.size), dtype=np.float32)
    lr = np.empty((n, ilat.size, ilon.size), dtype=np.float32)

    precip_var = im["precipitation"]
    for k, (ymd, d) in enumerate(zip(full_days, dates)):
        files_for_day = by_day[ymd]
        hr[k] = daily_hr_field(files_for_day, iy0, iy1, ix0, ix1)
        fld = precip_var.isel(time=idx_by_date[d]).values.astype(np.float32)
        fld = np.where(np.isfinite(fld) & (fld >= 0), fld, 0.0)
        lr[k] = fld
        if (k + 1) % 20 == 0 or k == n - 1:
            print(f"  {k + 1}/{n}  {ymd}  HR max so far {hr[:k+1].max():.1f} mm")
    im.close()

    years = dates.year.values
    is_train = np.isin(years, list(TRAIN_YEARS))
    is_val = np.isin(years, list(VAL_YEARS))
    print(f"[split] train {is_train.sum()}일, val {is_val.sum()}일 "
          f"(그 외 {n - is_train.sum() - is_val.sum()}일은 미사용)")

    gmax = float(max(hr[is_train].max(), lr[is_train].max()))
    print(f"[norm] train HR max={hr[is_train].max():.2f}  "
          f"train LR max={lr[is_train].max():.2f}  -> gmax={gmax:.4f} (min=0 고정)")

    def save_nc(path, arr, dts, lat_arr, lon_arr, dtype_name):
        ds = xr.Dataset(
            {"precipitation": (("time", "lat", "lon"), arr.astype(np.float32))},
            coords={"time": dts, "lat": lat_arr, "lon": lon_arr},
        )
        enc = {"precipitation": {"zlib": True, "complevel": 4}}
        ds.to_netcdf(path, encoding=enc)
        print("saved", path, arr.shape, f"[{dtype_name}]")

    # raw (QC/archival)
    save_nc(os.path.join(OUT_DIR, "HSR_daily_kor_crop_raw.nc"), hr, dates, hlat_crop, hlon_crop, "mm/day raw")
    save_nc(os.path.join(OUT_DIR, "IMERG_daily_kor_matched_raw.nc"), lr, dates, ilat, ilon, "mm/day raw")

    # normalized splits
    hr_norm = np.clip(hr / gmax, 0.0, 1.0)
    lr_norm = np.clip(lr / gmax, 0.0, 1.0)

    save_nc(os.path.join(OUT_DIR, "HR_train_norm.nc"), hr_norm[is_train], dates[is_train], hlat_crop, hlon_crop, "0~1")
    save_nc(os.path.join(OUT_DIR, "HR_val_norm.nc"), hr_norm[is_val], dates[is_val], hlat_crop, hlon_crop, "0~1")
    save_nc(os.path.join(OUT_DIR, "LR_train_norm.nc"), lr_norm[is_train], dates[is_train], ilat, ilon, "0~1")
    save_nc(os.path.join(OUT_DIR, "LR_val_norm.nc"), lr_norm[is_val], dates[is_val], ilat, ilon, "0~1")

    meta = {
        "gmax": gmax,
        "gmin": 0.0,
        "train_years": sorted(TRAIN_YEARS),
        "val_years": sorted(VAL_YEARS),
        "n_train": int(is_train.sum()),
        "n_val": int(is_val.sum()),
        "hr_grid": [int(hlat_crop.size), int(hlon_crop.size)],
        "lr_grid": [int(ilat.size), int(ilon.size)],
        "dates_train": dates[is_train].strftime("%Y-%m-%d").tolist(),
        "dates_val": dates[is_val].strftime("%Y-%m-%d").tolist(),
        "source_hsr_dir": HSR_UTC_DIR,
        "source_imerg_nc": IMERG_NC,
    }
    with open(os.path.join(OUT_DIR, "norm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("saved", os.path.join(OUT_DIR, "norm_meta.json"))


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit_days=limit)
