#!/bin/bash
# 2021 JJA 밖(nonjja) 시간별 예측 워커를 GPU 2 에서 항상 MAXP 개(기존 JJA 정적 워커 포함)까지 유지.
# 워커는 동적 큐(CLAIM=1)라 서로 날짜가 안 겹치고, 전처리(prep_hsr_2021_nonjja.py)가 만드는 날부터 바로 처리한다.
# 큐가 비면(QUEUE_EMPTY) 종료. 죽은 워커(OOM 등)는 자동 재기동, 그 워커의 claim 은 PID 확인 후 회수된다.
# 사용: RUN=<save 경로> MAXP=5 bash run_hourly_2021_workers.sh
PY=/home/inhye_yoo/anaconda3/envs/liif/bin/python
NJ=/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge/regrid/hsr2021_nonjja
RUN=${RUN:-/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h}
MAXP=${MAXP:-5}
LOGDIR=${LOGDIR:-/tmp}
cd "$(dirname "$0")"
i=0
while [ ! -f "$NJ/QUEUE_EMPTY" ]; do
  n=$(pgrep -fc "^$PY infer_hsr_2x_ft_daily_2021.py")
  if [ "$n" -lt "$MAXP" ]; then
    i=$((i+1))
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RUN=$RUN BIN=1 BSIZE=200000 SRC=nonjja CLAIM=1 \
      nohup $PY infer_hsr_2x_ft_daily_2021.py > "$LOGDIR/nonjja_worker_$i.log" 2>&1 &
    sleep 60          # 모델 로드/메모리 안정화 후 다음 워커
  else
    sleep 60
  fi
done
wait
echo "queue empty, supervisor done"
