"""
_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3[_lag1h] (2x LIIF + MLP adapter ASOS finetune) 로 2021 을 KST 일 단위로 추론해 저장.
  - 입력 = 1km LR (학습 입력과 동일 cdo remapbil), 출력 = 500m (obscrop 격자 1326x1234).
  - SRC=val    : JJA. HSR_val_norm_2x_obscrop.nc / HSR_val_500m_norm_obscrop.nc (기본)
    SRC=nonjja : JJA 밖 2021. prep_hsr_2021_nonjja.py 가 만든 날짜별 nc (hsr2021_nonjja/<날짜>_{hr,lr}.nc; 준비되는 대로 처리)
  - 모델은 시간별(mm/h)로 돌리고 BIN 시간씩 합산한다 (일합을 입력하면 학습 분포/PMAX 와 안 맞음).
  - 어댑터는 query_rgb 안에서 이미 적용됨 -> correction_layer 를 다시 부르지 말 것.
  - 비교용으로 같은 nc 에 raw HSR 합(precip_gt)도 저장. 레이더 영역 밖(BIN 내내 무효)은 둘 다 NaN.
  - 주의: run 이름에 lag1h 가 없으면(예: ..._ft_mlp_1e-3) lag0 obs csv 로 학습돼 ASOS 와 1h 어긋난 짝을 배운 모델임. nc 의 note 속성에 자동 표기.

사용:  python infer_hsr_2x_ft_daily_2021.py [날짜 ...]     # 기본 SRC=val 전체 (이미 있는 날은 건너뜀 -> 끊겨도 이어서)
       BIN=1 ...      # 시간별 저장 (하루 1파일, precip[24,lat,lon] mm/h). BIN=3 이면 3시간 누적 8구간, BIN=24(기본) 일합(2D)
       SHARD=0/3 ...  # (정적 분할) 날짜를 3등분해 i번째 몫만
       SRC=nonjja CLAIM=1 ...  # (동적 큐) 워커 여러 개를 띄우면 서로 안 겹치게 날짜를 하나씩 가져감(.claim_<날짜> 에 PID 기록, 죽은 PID 의 claim 은 회수).
                               #   전처리(PREP_DONE 없음)가 끝날 때까지 새 날짜를 기다리고, 다 끝나면 QUEUE_EMPTY 를 남기고 종료
       CKPT=epoch-last ...   # 체크포인트 (기본 epoch-best-loss)     RUN=<save 경로> ...   # 다른 ft run
       BSIZE=200000 ...      # 질의 청크 (GPU 메모리 부족 시 줄임; 병렬 실행 권장 200000)
출력:  <run>/preds_nc/{daily|3h|hourly}_2021_<CKPT>/hsr2x_ft_mlp_{daily|3h|hourly}_<날짜>.nc   (precip, precip_gt, lat asc; BIN=3/1 은 time = 구간 시작 KST)
env: liif, 물리 GPU 2 (CUDA_VISIBLE_DEVICES=2 -> cuda:0)
"""
import os, sys, copy, time, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import pandas as pd
import torch
import xarray as xr
import models
from utils import make_coord

dev = torch.device("cuda:0")
M = "/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid"
LR_NC = f"{M}/HSR_val_norm_2x_obscrop.nc"
HR_NC = f"{M}/HSR_val_500m_norm_obscrop.nc"
NJ_DIR = os.environ.get("NJ_DIR", f"{M}/hsr2021_nonjja")
SRC = os.environ.get("SRC", "val")
CLAIM = os.environ.get("CLAIM") == "1"
RUN = os.environ.get("RUN", "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3")   # 다른 ft run 은 RUN=<save 경로> 로
CKPT_TAG = os.environ.get("CKPT", "epoch-best-loss")
CKPT = f"{RUN}/{CKPT_TAG}.pth"
BIN = int(os.environ.get("BIN", "24"))                      # 누적 시간 수: 24=일합, 3=3시간, 1=시간별
assert BIN in (1, 3, 24), BIN
NAME = {24: "daily", 3: "3h", 1: "hourly"}[BIN]
OUT = f"{RUN}/preds_nc/{NAME}_2021_{CKPT_TAG}"
os.makedirs(OUT, exist_ok=True)
LAG_NOTE = "lag1h-aligned obs" if "lag1h" in RUN else "lag0 obs (ASOS 1h misaligned)"
SHARD = os.environ.get("SHARD")                             # "i/N"
PMAX = 74.8784
BSIZE = int(os.environ.get("BSIZE", "500000"))              # 질의 청크 크기. GPU 메모리 부족(OOM) 시 줄일 것 (병렬 실행 시 200000 권장)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
spec = copy.deepcopy(ck["model"])
model = models.make({"name": spec["name"], "args": spec["args"]}).to(dev)
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
print(f"{CKPT_TAG} (epoch {ck.get('epoch')}), SRC={SRC}, BIN={BIN}, CLAIM={CLAIM}, BSIZE={BSIZE}", flush=True)

hr_ds, lr_ds = xr.open_dataset(HR_NC), xr.open_dataset(LR_NC)          # 격자/좌표 기준 (SRC=nonjja 도 같은 obscrop 격자)
times = pd.to_datetime(hr_ds.time.values)
times_day = np.asarray(times.strftime("%Y-%m-%d"))
H, W = hr_ds.sizes["lat"], hr_ds.sizes["lon"]
coord = make_coord((H, W)).unsqueeze(0).to(dev)
cell = torch.ones_like(coord); cell[:, :, 0] *= 2.0 / H; cell[:, :, 1] *= 2.0 / W


def list_days():
    if SRC == "val":
        return sorted(set(times_day))
    have = {os.path.basename(f)[:10] for f in glob.glob(f"{NJ_DIR}/*_hr.nc")} & {os.path.basename(f)[:10] for f in glob.glob(f"{NJ_DIR}/*_lr.nc")}
    return sorted(have)


def day_getters(day):
    """(hr(k), lr(k)) : k=0..23 시간의 정규화 배열."""
    if SRC == "val":
        tidx = np.where(times_day == day)[0]
        assert len(tidx) == 24, (day, len(tidx))
        return (lambda k: hr_ds.precip_norm.isel(time=int(tidx[k])).values), (lambda k: lr_ds.precip_norm.isel(time=int(tidx[k])).values)
    h, l = xr.open_dataset(f"{NJ_DIR}/{day}_hr.nc"), xr.open_dataset(f"{NJ_DIR}/{day}_lr.nc")
    assert h.sizes["time"] == 24 and l.sizes["time"] == 24, day
    return (lambda k: h.precip_norm.isel(time=k).values), (lambda k: l.precip_norm.isel(time=k).values)


@torch.no_grad()
def predict(x):
    model.gen_feat(x)
    out = [model.query_rgb(coord[:, s:s + BSIZE], cell[:, s:s + BSIZE]) for s in range(0, coord.shape[1], BSIZE)]
    return (torch.cat(out, 1).view(H, W).clamp(0, 1) * PMAX).clamp(min=0).cpu().numpy()


def out_path(day):
    return f"{OUT}/hsr2x_ft_mlp_{NAME}_{day}.nc"


def try_claim(day):
    """날짜 하나를 원자적으로 선점. 죽은 PID 의 claim 은 회수."""
    cp = f"{OUT}/.claim_{day}"
    for _ in range(2):
        try:
            fd = os.open(cp, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
        except FileExistsError:
            try:
                pid = int(open(cp).read().strip() or 0)
                os.kill(pid, 0); return False                    # 살아 있음
            except (ValueError, ProcessLookupError):
                try: os.remove(cp)
                except FileNotFoundError: pass
            except PermissionError:
                return False
    return False


def process(day):
    get_hr, get_lr = day_getters(day)
    nb = 24 // BIN
    pred = np.zeros((nb, H, W), np.float32); gt = np.zeros((nb, H, W), np.float32); valid = np.zeros((H, W), bool)
    for k in range(24):
        hr = get_hr(k); lr = get_lr(k)
        valid |= np.isfinite(hr)
        gt[k // BIN] += np.nan_to_num(hr) * PMAX
        pred[k // BIN] += predict(torch.from_numpy(np.nan_to_num(lr).astype(np.float32))[None, None].to(dev))
        torch.cuda.empty_cache()
    pred[:, ~valid] = np.nan; gt[:, ~valid] = np.nan
    unit = {24: "mm/day", 3: "mm/3h", 1: "mm/h"}[BIN]
    attrs = {"date_kst": day, "ckpt": CKPT, "ckpt_epoch": int(ck.get("epoch") or -1), "input": LR_NC if SRC == "val" else NJ_DIR, "bin_hours": BIN,
             "note": f"hourly (mm/h) predictions summed over {BIN}h bins (KST, time = bin start); {LAG_NOTE}"}
    lat, lon = hr_ds.lat.values, hr_ds.lon.values
    if BIN == 24:
        ds = xr.Dataset({"precip": (("lat", "lon"), pred[0], {"units": unit, "long_name": "LIIF 2x + MLP adapter ft, 24h sum of hourly 500m predictions"}),
                         "precip_gt": (("lat", "lon"), gt[0], {"units": unit, "long_name": "raw HSR 500m, 24h sum"})},
                        coords={"lat": lat, "lon": lon}, attrs=attrs)
    else:
        tb = pd.Timestamp(day) + pd.to_timedelta(np.arange(nb) * BIN, unit="h")
        ds = xr.Dataset({"precip": (("time", "lat", "lon"), pred, {"units": unit, "long_name": f"LIIF 2x + MLP adapter ft, {BIN}h sum"}),
                         "precip_gt": (("time", "lat", "lon"), gt, {"units": unit, "long_name": f"raw HSR 500m, {BIN}h sum"})},
                        coords={"time": tb.values, "lat": lat, "lon": lon}, attrs=attrs)
    enc = {v: {"zlib": True, "complevel": 4} for v in ("precip", "precip_gt")}
    p = out_path(day)
    ds.to_netcdf(p + ".tmp", encoding=enc); os.replace(p + ".tmp", p)          # 끊겨도 반쪽 파일이 SKIP 되지 않게 원자적으로


t0 = time.time(); n_done = 0
if not CLAIM:                                                # 정적 모드 (기존 동작)
    days = list_days()
    todo = sys.argv[1:] or days
    if SHARD:
        si, sn = (int(v) for v in SHARD.split("/"))
        todo = todo[si::sn]
    print(f"{len(todo)}일 처리 예정 (전체 {len(days)}일), 출력 {OUT}", flush=True)
    for day in todo:
        if os.path.exists(out_path(day)):
            continue
        process(day); n_done += 1
        el = time.time() - t0
        print(f"  {day} 저장 ({n_done}/{len(todo)}, {el / 60:.1f}분 경과, 하루 평균 {el / n_done:.0f}초)", flush=True)
else:                                                        # 동적 큐 (여러 워커)
    print(f"동적 큐 워커 시작, 출력 {OUT}", flush=True)
    while True:
        got = None
        for day in list_days():
            if os.path.exists(out_path(day)):
                continue
            if try_claim(day):
                if os.path.exists(out_path(day)):            # claim 직전에 다른 워커가 끝낸 경우
                    os.remove(f"{OUT}/.claim_{day}"); continue
                got = day; break
        if got is None:
            if SRC == "nonjja" and not os.path.exists(f"{NJ_DIR}/PREP_DONE"):
                time.sleep(30); continue                     # 전처리가 더 만들 때까지 대기
            open(f"{NJ_DIR}/QUEUE_EMPTY" if SRC == "nonjja" else f"{OUT}/QUEUE_EMPTY", "w").close()
            break
        try:
            process(got); n_done += 1
        finally:
            try: os.remove(f"{OUT}/.claim_{got}")
            except FileNotFoundError: pass
        el = time.time() - t0
        print(f"  {got} 저장 (내 {n_done}일, {el / 60:.1f}분 경과, 하루 평균 {el / n_done:.0f}초)", flush=True)
print("done", flush=True)
