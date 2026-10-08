# 실험 노트: 약한 비 과대 / 강한 비 과소를 동시에 줄이기 (문맥 어댑터 + 구간별 편향 손실)

작성: 2026-10-07 · 위치: `deeplearning/inr/liif/exp_ctx_binloss/` · **기존 코드는 수정하지 않았고, 모두 새 파일**이다.

## 1. 문제

2021 JJA 지점(ASOS) 평가에서 현재 가장 좋은 모델(8x LTE + Weighted-MAE, 2019+2020 학습)도 **약한 비는 과대, 강한 비는 과소**다.

기준선 수치 (2021 JJA, 48x48 크롭 방식, `eval_station_2021.py` 로 재현 검증함):

| 항목 | HSR 원본 | 8x LTE wmae **ep564** (`best`, r 기준) | 8x LTE wmae **ep542** (`best-loss`) |
|---|---|---|---|
| ASOS≥0.1 r / bias / RMSE / ratio | 0.760 / −1.24 / 3.84 / 0.554 | 0.810 / −0.15 / 3.02 / 0.945 | 0.806 / −0.24 / 3.05 / 0.915 |
| 필터 없음 bias / RMSE / ratio | −0.77 / 3.16 / 0.590 | +0.05 / 2.54 / 1.029 | −0.06 / 2.56 / 0.965 |
| 큰 비 ratio (ASOS≥10 / ≥20) | 0.40 / 0.35 | 0.70 / 0.60 | 0.72 / 0.62 |
| ASOS 0~0.5 구간 예측/ASOS | 1.54 | **2.81** | **1.97** |
| ASOS 0.5~1 / 1~2 | 0.93 / 0.78 | 1.55 / 1.26 | 1.34 / 1.18 |
| ASOS=0 시각: 예측≥0.1 비율 / 예측 평균 | 47.5% / 0.21 | **93.5% / 0.48** | **30.2% / 0.28** |

- **발견(주의)**: 같은 학습에서 고른 두 체크포인트(ep542 vs ep564)가 약한 비 과대에서 크게 다르다(0~0.5 구간 1.97 vs 2.81, ASOS=0 시각 예측≥0.1 비율 30% vs 93%). r·RMSE·큰 비 ratio 는 거의 같다(0.806 vs 0.810 등). 즉 **약한 비 과대는 체크포인트 선택에 민감**하고, 이전 대화의 "LTE wmae 약한 비 과대" 수치는 ep564(`best`) 기준이었다. 새 실험의 기준선은 **둘 다** 적어 둔다.
- 다른 모델 비교: vzero 라벨을 쓴 2x LIIF wmae `plain final` 은 약한 비가 거의 맞지만(0.5~1: 0.95) 큰 비가 크게 모자랐다(ASOS≥10 ratio 0.48).

## 2. 원인 가설 (확인된 것 / 추정)

1. **(확인) 손실 가중치**: Weighted-MAE 는 `w = clip(정답, 1mm, 30mm)` 를 가중평균으로 정규화한다. 정답 1 mm/h 미만(약한 비)은 30 mm/h 인 점의 1/30 비중이다 → 약한 비 오차는 손실에서 거의 안 보인다.
2. **(추정) 어댑터가 보는 정보가 부족**: 기존 어댑터는 방향별 imnet 출력 p_k **숫자 하나**만 입력으로 받는다. 같은 p_k 에 ASOS=0 인 시각과 비 오는 시각이 섞여 있으면 구분할 수 없어 평균으로 타협한다(약한 쪽은 위로, 강한 쪽은 아래로). 어댑터 g(p) 는 낮은 p_k 를 약 1.3배 키우는 형태였다(2x LIIF 기준 확인).
3. **(확인) 라벨**: 기존(non-vzero) 학습 라벨에도 0 은 있지만(2019 기준 약 32%) ASOS 파일에 0.0 으로 기록된 시각뿐이고 맑은 시각은 거의 없다. vzero 는 확인된 무강수 시각을 추가해 0 비율이 80%가 되지만, 랜덤 초기화 어댑터는 1 epoch 만에 "전부 0"으로 무너진 전례가 있어 항등 초기화가 필요하다.

## 3. 실험 설계 (이 폴더의 새 파일)

| 파일 | 내용 |
|---|---|
| `models_ctx.py` | 모델 `lte_ctx`: 기존 `LTE` 를 상속해 어댑터 입력에 **문맥**을 추가 |
| `train_ctx_binloss.py` | `train_liif_finetune_vzero_wmae.py` 복사본 + 구간별 편향 손실(binloss) + `lte_ctx` 지원 + 검증 진단 |
| `eval_station_2021.py` | 체크포인트 → 2021 JJA 지점 평가표(기준선과 나란히). 기존 `station_preds_dump_*.py` 규약 |
| `configs/ctxA_lte_ctx_wmae.yaml` | A: 문맥 어댑터 + 기존 손실 (1번만) |
| `configs/ctxB_lte_plain_wmae_binloss.yaml` | B: 기존 plain 어댑터 + 구간별 손실 (2번만) |
| `configs/ctxC_lte_ctx_wmae_binloss.yaml` | C: 문맥 어댑터 + 구간별 손실 (1번+2번) |
| `results/` | `eval_station_2021.py` 가 저장하는 예측 npz |

### 3.1 (1번) 문맥 어댑터 `lte_ctx`

방향 k 의 어댑터 입력(정규화 공간, `in_scale=10` 배): `p_k`, 4방향 `mean/max/min`, 질의 위치의 저해상도(LR) bilinear 값, LR 3×3 최대값.
출력 `g_k = p_k + f(입력)`, f 는 MLP(64-32), **마지막 Linear 0 초기화 → 시작 시점은 어댑터 없는 사전학습 출력과 동일**(항등 초기화, 확인: 초기 `|g−p|` 최대 0). 최종은 `Σ_k w_k g_k + LR bilinear`(원본 `adapter_per_direction=True` 와 같은 결합). add_layer 파라미터 2,561개(기존 plain 1,169개). 모듈 이름을 `add_layer.asos_adapter.*` 로 맞춰 기존 freeze/체크포인트 규약과 호환.
의도: 방향 간 일관성(모두 크면 확실한 비)과 주변 비 유무로 "ASOS=0 인 약한 값"과 "실제 약한 비"를 가르게 한다.

### 3.2 (2번) 구간별 편향 손실 `binloss`

`L = L_wmae + λ_abs · L_balMAE + λ_bias · L_bias` (모두 정규화 공간, 배치 안에서 계산)
- 구간: ASOS 정답 기준 `[=0, (0,0.5], (0.5,1], (1,2], (2,5], (5,10], (10,20], >20]` mm/h (`edges_mm`), 구간 안 표본이 `min_count`(=2) 미만이면 그 구간은 제외.
- `L_balMAE` = 구간별 평균 절대오차의 평균 (구간마다 같은 비중 → 약한 비 비중을 1/30 에서 끌어올림)
- `L_bias` = 구간별 평균 부호오차 절댓값의 평균 (약한 비 과대는 아래로, 큰 비 과소는 위로 직접 밀어줌)
- 기본 λ_abs = λ_bias = 1.0 (튜닝 안 함)
- 검증 손실도 같은 항을 전체 검증셋 기준으로 더해서 best-loss 선택이 이 목표를 반영한다. 검증 로그에 `pred/ASOS by bin`, `binMAE`, `binBias` 출력.

### 3.3 공통 조건 (기준선과 동일하게 맞춘 것)

기준 설정 = `climate_edsr_radar_hsr_8x_minmax_lte_ft_mlp_1e-3_lag1h_2019_2020_wmae.yaml`. 학습 2019+2020 JJA(8x LR 입력, non-vzero lag1h 라벨), 검증 val csv, resume = 8x LTE pretrain(epoch 500), optimizer adam lr 1e-3, epoch 501~800, Weighted-MAE(1~30 mm/h), seed 42. **LTE 헤드(coef/freq/phase)도 기준선처럼 같이 학습된다**(`adapter_only` 아님; 시험 학습에서 trainable 1,182,977개 확인).
바꾼 것은 A: 모델(어댑터), B: 손실, C: 둘 다 — 이 세 개 + 기준선으로 요인을 분리한다.

## 4. 실행

```
cd /home/inhye_yoo/deeplearning/inr/liif
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python exp_ctx_binloss/train_ctx_binloss.py \
    --config exp_ctx_binloss/configs/ctxC_lte_ctx_wmae_binloss.yaml --gpu 2          # A/B 도 같은 방식
# 저장 폴더: save/_ctxC_lte_ctx_wmae_binloss  (config 파일 이름 기준)
# 평가 (체크포인트마다):
RUN=save/_ctxC_lte_ctx_wmae_binloss CKPT=epoch-best-loss python exp_ctx_binloss/eval_station_2021.py
RUN=save/_ctxC_lte_ctx_wmae_binloss CKPT=epoch-best      python exp_ctx_binloss/eval_station_2021.py
```
- GPU 2번 사용(작업 규약). 시험 학습 기준 epoch 약 1.6분(GPU 를 다른 작업과 나눠 쓸 때), 300 epoch 약 8시간. 평가 1회 약 6분.
- 학습은 사용자가 직접 실행(자동 실행하지 않음).

## 5. 평가 방법과 성공 기준

평가는 `eval_station_2021.py` 의 표 한 벌(ASOS≥0.1 / 필터 없음, 큰 비 ratio, ASOS 구간별 예측/ASOS, ASOS=0 시각)로 하고, **best-loss 와 best 체크포인트를 둘 다** 본다(§1 발견).
목표(기준선 ep542·ep564 대비 상대적으로; 숫자는 임의로 정한 가이드):
- 약한 비: ASOS 0~0.5 구간 비율 ≤ 1.5, ASOS=0 시각 예측 평균 ≤ 0.25 mm/h
- 큰 비: ASOS≥10 ratio ≥ 0.75 (기준선 0.70~0.72), ASOS≥20 ratio ≥ 0.65
- 전체: r ≥ 0.80, RMSE(≥0.1) ≤ 3.05 — 약한/큰 비를 맞추면서 전체 지표를 해치지 않을 것
가설: H1 A 만으로도 약한 비 과대가 줄고 큰 비는 유지. H2 B 만으로 양쪽이 같이 개선되나 전체 r 은 소폭 하락 가능. H3 C 가 가장 좋다. 틀리면 그 자체가 결과다.

## 6. 검증한 것 / 하지 않은 것

검증함:
- `lte_ctx` 가 만들어지고 forward 가 된다. 초기 잔차 0(항등).
- 학습 스크립트: ctxC 로 1 epoch 시험 학습 성공(1.6분, 구간별 손실·문맥 어댑터 작동, 검증 로그에 `pred/ASOS by bin` 출력). 시험 폴더(`save/_smoke_ctxC_tmp`)는 삭제함.
- 평가 스크립트: 기존 8x LTE wmae `epoch-best-loss`(ep542)로 돌려 저장된 ep542 예측 npz 와 **완전 일치**(최대 차 0).

하지 않음 / 한계:
- A·B·C 의 본 학습과 결과 — **아직 없음**. 위 가설은 검증 전.
- λ(1.0) 와 구간 경계, 문맥 특징 조합은 튜닝하지 않았다. 구간별 항은 배치(스텝당 관측소 약 50개)에서 계산해 큰 비 구간이 얇고 잡음이 크다(min_count=2 로 완화). 효과가 약하면 λ 를 키우거나 누적 통계를 쓰는 변형을 고려.
- 평가는 48×48 크롭 방식이다. 전체 영상(nc) 방식과 점별로 약간 다르다(2021 2x LIIF 로 확인: 평균 같고 RMSE 0.35, 큰 값에서 크롭이 약간 낮음). 모델 간 비교는 같은 방식이면 유효하다.
- 체크포인트를 2021 검증으로 골랐고 평가도 같은 2021 JJA 라서 약간 유리할 수 있다(검증 세트와 평가 세트의 정확한 겹침은 확인 안 함).
- 학습 라벨은 non-vzero(기준선과 동일). vzero 라벨(맑은 시각 0 추가)은 이번 요인에서 제외 — 다음 후보.

## 7. 다음 후보 (이번에 안 한 것)
- vzero 라벨 + 항등 초기화(이미 `lte_ctx` 가 항등 초기화) 와 결합
- 어댑터 입력 확장: 지형(topo), 더 큰 주변 창(5×5), 4방향 분산
- Bernoulli-Gamma(비 올 확률 × 강도) 구조
- 학습 후 체크포인트 선택 기준을 구간별 지표로 바꾸기 (best r / best-loss 외)
