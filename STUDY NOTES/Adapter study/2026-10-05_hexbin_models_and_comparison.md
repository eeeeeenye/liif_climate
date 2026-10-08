# HSR 시간별 강수 다운스케일 모델 정리 및 성능 비교

작성일: 2026-10-05

## 0. 요약

현재 최고 모델은 **T8-WMAE-1920 (ep564)** 로, 2021 JJA 에서 r 0.82, RMSE 2.54 mm/h, ratio 1.03 이다.

핵심 결론:

1. **학습 기간 효과가 가장 크다.** 2019만 → 2019+2020 으로 늘리면 ratio 가 8x 0.64 → 0.90, 2x 0.68 → 1.06 으로 오른다.
2. **손실은 WMAE 가 MSE 보다 낫다.** 8x LIIF 에서 bias −0.19 → +0.06, ratio(ASOS≥10) 0.56 → 0.66.
3. **LTE 는 coef/freq/phase 를 어댑터와 같이 미세조정해야 한다.** 어댑터만 학습하면 r 0.811 → 0.732 로 떨어진다 (4절).
4. **강한 비는 전 모델이 과소예측한다.** ASOS≥20 에서 ratio 최고 0.62.
5. **큰 값에서 pred 가 HSR 보다 작아지는 것은 어댑터가 아니라 imnet 단계 탓이다.** 2x 와 8x 에서 비율이 거의 같다 (5절).

모든 수치는 2021 JJA 로 체크포인트도 골랐으므로 낙관적일 수 있다. 독립 테스트가 아니다.

다음 할 일: L2-MSE-1920 학습이 끝나면 L8-MSE-1920 과 비교해 해상도 효과만 분리한다. L8-L1-1920, T2-WMAE-1920 은 설정만 만들어 두었다 (6절).

## 1. 공통 설정

모든 모델은 사전학습된 LIIF/LTE 위에 ASOS 어댑터를 파인튜닝한 것이다. 아래는 모델 간 다르지 않은 항목이다.

| 항목 | 내용 |
| --- | --- |
| 입력 | HSR 500 m 를 remapbil 로 거칠게 만든 LR, 지점 중심 48x48 crop. 2x = 1 km, 8x = 약 4 km |
| 출력 | 지점 1점 예측. clamp(0,1) 후 x 74.8784 → mm/h |
| 정답 | ASOS 시간별 강수, 96 지점. lag1h 정렬: ASOS[T] ↔ nc[T−1h] |
| 학습 | encoder/imnet 동결, lr 1e-3. LIIF 는 어댑터만, LTE 는 어댑터 + coef/freq/phase 학습 (4절) |
| 어댑터 | 1차원 입출력 MLP, ReLU. 32-16-32 = 1-32-16-32-1, 128-64-128 = 1-128-64-128-1 |
| 어댑터 위치 | LIIF: 방향별(4방향) imnet 출력 p_k 에 적용 후 area 가중 평균. LTE: 기본은 blend 후 1회, 여기서는 `adapter_per_direction: true` |
| 학습 데이터 | 2019 JJA = 2208 h, obs 28,468 / 2019+2020 JJA = 4416 h, obs 74,390 |
| 검증 | 2021 JJA, 2208 h. `epoch-best-loss`, `epoch-best` 선택에도 사용 |
| 사전학습 resume | 8x 모델 epoch 500, 2x 모델 epoch 300 |

수식으로 쓰면 LIIF 의 최종 예측은 다음과 같다 (w_k 는 합 1 인 area 가중치).

```
pred = clamp( sum_k w_k * adapter(p_k), 0, 1 ) * 74.8784
```

`loss:` 키가 없는 config 는 기본값 MSE 로 학습된다 (`train_liif_finetune.py:261`).

## 2. 모델 목록

이 문서는 모델을 아래 ID 로만 부른다. ID 규칙은 `백본해상도-손실-학습기간` 이고, L = LIIF, T = LTE, 19 = 2019 JJA, 1920 = 2019+2020 JJA 이다.

| ID | 백본 | 손실 | 학습 기간 | 학습 obs | 어댑터 | hexbin 그림 이름 | 상태 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L8-MSE-19 | 8x LIIF | MSE | 2019 | lag1h | 32-16-32 | `8x_2019only` | 완료 |
| L8-MSE-1920 | 8x LIIF | MSE | 2019+20 | lag1h | 32-16-32 | `8x_2019_2020` | 완료 |
| L8-WMAE-1920 | 8x LIIF | WMAE | 2019+20 | lag1h | 32-16-32 | `8x_2019_2020_wmae` | 완료 |
| T8-WMAE-1920 | 8x LTE | WMAE | 2019+20 | lag1h | 32-16-32 | `8x_lte_..._wmae` (snap530, ep542, ep564) | 653/800 에서 중단 |
| L2-WMAE-19 | 2x LIIF | WMAE | 2019 | lag1h + vzero | 32-16-32 | `wmae_plainadapter_final` | 완료 |
| L2-WMAE-1920-L | 2x LIIF | WMAE | 2019+20 | lag1h + vzero | 128-64-128 | `wmae_plainadapter_128_64_128` | 완료 |
| T8-WMAE-1920-AO | 8x LTE | WMAE | 2019+20 | lag1h | 32-16-32, 어댑터만 학습 | (없음) | 완료, 4절 비교용 |
| L2-MSE-1920 | 2x LIIF | MSE | 2019+20 | lag1h | 32-16-32 | (아직 없음) | 학습 중 |
| L8-L1-1920 | 8x LIIF | L1 | 2019+20 | lag1h | 32-16-32 | (아직 없음) | 설정만 |
| T2-WMAE-1920 | 2x LTE | WMAE | 2019+20 | lag1h | 32-16-32 | (아직 없음) | 설정만 |

- T8-WMAE-1920 의 snap530, ep542, ep564 는 같은 run 의 체크포인트다. 정확한 에폭은 파일 이름 이상으로 확인하지 못했다.
- `hsr_cap53_vs_cap65` 그림은 모델이 아니라 HSR 원본끼리의 비교다.
- 사전학습 모델: 8x LIIF `_climate_edsr_radar_hsr_8x_minmax_liif`, 8x LTE `..._8x_minmax_lte`, 2x LIIF `..._2x_minmax_liif`, 2x LTE `..._2x_minmax_lte`.

## 3. hexbin 성능 비교

T8-WMAE-1920 이 r, RMSE 모두 가장 좋고, 2019 단독 학습 모델은 HSR 원본과 비슷하게 과소예측한다.

> **데이터** 2021 JJA hourly, 필터 없음(THR=0), n = 31,533 · **출처** `station_eval/preds_npz/*.npz` · **단위** mm/h. 모든 npz 의 ASOS/HSR 값 동일 확인.

| 모델 | r | bias | RMSE | ratio | OLS 기울기 | ratio (ASOS≥10) | ratio (ASOS≥20) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HSR 원본 500 m | 0.77 | −0.77 | 3.16 | 0.59 | 1.58 | 0.40 | 0.35 |
| L8-MSE-19 | 0.79 | −0.67 | 3.00 | 0.64 | 1.48 | 0.44 | 0.39 |
| L8-MSE-1920 | 0.79 | −0.19 | 2.77 | 0.90 | 1.21 | 0.56 | 0.49 |
| L8-WMAE-1920 | 0.79 | +0.06 | 2.70 | 1.03 | 1.02 | 0.66 | 0.56 |
| T8-WMAE-1920 snap530 | 0.81 | −0.07 | 2.60 | 0.96 | 0.99 | 0.71 | 0.61 |
| T8-WMAE-1920 ep542 | 0.81 | −0.06 | 2.56 | 0.97 | 0.99 | 0.72 | 0.62 |
| **T8-WMAE-1920 ep564 (best)** | **0.82** | +0.05 | **2.54** | 1.03 | 1.05 | 0.70 | 0.60 |
| L2-WMAE-19 | 0.80 | −0.59 | 2.90 | 0.68 | 1.43 | 0.48 | 0.41 |
| L2-WMAE-1920-L | 0.80 | +0.12 | 2.68 | 1.06 | 0.95 | 0.70 | 0.58 |

지표 정의: bias = mean(pred − ASOS), ratio = mean(pred) / mean(ASOS), OLS 기울기 = ASOS 를 pred 에 회귀한 기울기 (1 이면 이상적). ratio(ASOS≥10) 은 n = 1,322, ratio(ASOS≥20) 은 n = 376 표본만으로 계산.

### 관찰

1. **학습 기간.** 2019 만 쓰면 HSR-ASOS 기울기가 약 0.98 인 관계를 배운다. 2020, 2021 은 기울기가 1.3~1.6 이라 2019 단독 모델은 과소예측한다.
2. **손실.** L8-MSE-1920 → L8-WMAE-1920 에서 bias −0.19 → +0.06, ratio(ASOS≥10) 0.56 → 0.66.
3. **백본.** LTE 가 LIIF 보다 r 0.02~0.03 높고 RMSE 0.1~0.16 낮다.
4. **강한 비.** ASOS≥20 ratio 는 최고 0.62 (T8 ep542).
5. **해상도 비교 불가.** L2-WMAE-1920-L 과 L8-WMAE-1920 은 어댑터 크기와 학습 obs(vzero)가 다르다. 같은 조건 비교는 L2-MSE-1920 완료 후 L8-MSE-1920 과 한다.

## 4. LTE 미세조정 범위: 같이 학습 vs 어댑터만

LTE 는 coef/freq/phase 를 어댑터와 같이 미세조정하는 편이 모든 지표에서 낫다. 공통 epoch 501~653 (153개) 전부에서 T8-WMAE-1920 의 val loss 가 더 낮았다.

| | T8-WMAE-1920 (같이) | T8-WMAE-1920-AO (어댑터만) |
| --- | --- | --- |
| 학습되는 것 | coef, freq, phase, add_layer | add_layer 만 |
| 학습 파라미터 수 | 1,181,585 | 1,169 |
| config 차이 | (기준) | yaml 끝에 `adapter_only: true` 한 줄 |
| 진행 | 653/800 에서 중단된 것으로 보임 | 800/800 완료 (약 1.7 h) |

두 run 의 나머지 설정은 같다: 8x LTE 사전학습 epoch 500 resume, 어댑터 32-16-32 `adapter_per_direction: true`, lr 1e-3, WMAE (w 1~30 mm/h), batch 64, 스크립트 `train_liif_finetune_vzero_wmae.py` (`adapter_only` 처리 197~202행). coef/freq/phase 는 encoder 나 imnet 이름이 아니어서 `adapter_only` 가 없으면 기본으로 같이 학습된다.

### best val loss epoch 기준

> **데이터** 2021 JJA 관측소 · **출처** 각 run 의 `log.txt` val 지표 · **단위** mm/h. 3절 hexbin 통계와 계산 방식이 달라 숫자를 섞어 쓰면 안 된다.

| 지표 | T8-WMAE-1920 (같이) | T8-WMAE-1920-AO (어댑터만) |
| --- | --- | --- |
| best val loss (epoch) | **0.0445** (542) | 0.0509 (688) |
| r | **0.811** | 0.732 |
| MAE | **1.069** | 1.300 |
| bias | **+0.013** | +0.263 |
| RMSE | **2.434** | 2.822 |
| dry (gt=0) 평균 pred | **0.332** | 0.484 |
| heavy (≥10) bias | **−4.75** | −6.22 |
| heavy (≥10) ratio | **0.73** | 0.64 |
| 마지막 20 epoch 평균 r | **0.813** | 0.738 |
| 마지막 20 epoch 평균 RMSE | **2.619** | 2.960 |
| 마지막 20 epoch 평균 heavy ratio | **0.70** | 0.62 |

### 같은 epoch 끼리

어댑터만 학습하면 epoch 이 진행될수록 r 이 0.80 → 0.72 로 내려간다. 같이 학습하면 0.81 안팎을 유지한다.

| epoch | r (같이 / AO) | val loss (같이 / AO) | MAE (같이 / AO) | RMSE (같이 / AO) | heavy ratio (같이 / AO) |
| --- | --- | --- | --- | --- | --- |
| 501 | 0.805 / 0.802 | 0.067 / 0.069 | 1.849 / 1.865 | 3.086 / 3.136 | 0.49 / 0.47 |
| 510 | 0.768 / 0.751 | 0.048 / 0.064 | 1.208 / 1.688 | 2.732 / 3.099 | 0.68 / 0.44 |
| 542 | 0.811 / 0.754 | 0.045 / 0.055 | 1.069 / 1.272 | 2.434 / 2.736 | 0.73 / 0.58 |
| 580 | 0.803 / 0.744 | 0.051 / 0.058 | 1.261 / 1.544 | 2.694 / 3.039 | 0.68 / 0.62 |
| 620 | 0.808 / 0.742 | 0.050 / 0.058 | 1.114 / 1.221 | 2.628 / 2.986 | 0.68 / 0.60 |
| 650 | 0.807 / 0.726 | 0.050 / 0.056 | 1.071 / 1.303 | 2.632 / 3.067 | 0.68 / 0.64 |

### 한계

- 각 설정 1회씩, 시드 반복 없음.
- 같이 학습하는 쪽은 파라미터 118만 개에 학습 obs 74,390 개라 과적합 위험이 있다. 단 2021 검증에서는 징후가 없다.
- T8-WMAE-1920-AO 는 hexbin 통계(npz)를 만들지 않았다.

## 5. 분석: 어댑터를 거쳤는데 값이 줄어 보이는 이유

줄어드는 단계는 어댑터가 아니라 그 앞의 imnet 이다. imnet 출력이 큰 값에서 HSR 보다 낮고, 어댑터가 올려도 HSR 까지 못 미친다.

### 질문

L8-MSE-19 의 어댑터 곡선은 p_k 0~20 에서 1:1 선 위에 있다. 그런데 2019 JJA (학습 데이터) HSR-Pred hexbin 에서는 x 0~10 인 점들이 줄어든 것처럼 보인다.

### 왜 모순이 아닌가

두 그림의 x 축이 다른 변수다.

| 그림 | x 축 | y 축 |
| --- | --- | --- |
| 어댑터 곡선 | p_k (imnet 출력) | adapter(p_k) |
| hexbin | HSR 500 m 원본 (모델 입력 아님) | pred |

따라서 "곡선이 1:1 위" 는 adapter(p_k) > p_k 라는 뜻이지 pred > HSR 이 아니다. hexbin 에서 줄어드는 양은 (HSR → p_k 에서 깎인 양) − (어댑터가 올린 양) 이다.

### 8x 단계별 분해

> **데이터** 2019 JJA 학습 샘플, n = 28,468 · **모델** L8-MSE-19 · **방법** scratchpad `stages.py` 에서 `del model.add_layer` 후 재추론 · **단위** mm/h

| HSR 구간 | n | HSR 평균 | blend(p_k), 어댑터 끔 | pred, 어댑터 켬 | p_k / HSR | HSR 보다 작은 점 (끔 → 켬) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1~1 | 11,202 | 0.46 | 0.52 | 0.49 | 1.13 | 43% → 58% |
| 1~3 | 6,610 | 1.76 | 1.75 | 2.02 | 0.99 | 54% → 41% |
| 3~6 | 2,614 | 4.21 | 3.85 | 4.82 | 0.91 | 63% → 34% |
| 6~10 | 944 | 7.56 | 6.49 | 8.25 | 0.86 | 72% → 36% |
| 10~15 | 386 | 12.17 | 9.22 | 11.57 | 0.76 | 80% → 55% |
| 15~25 | 157 | 18.23 | 12.80 | 16.32 | 0.70 | 88% → 65% |

- 전체 평균: HSR 1.55, 어댑터 끔 1.42, 어댑터 켬 1.69, ASOS 1.79. 평균은 어댑터가 올린다.
- 어댑터가 값을 오히려 낮춘 샘플이 36% 있다. 0.1~1 구간은 평균도 0.52 → 0.49 로 낮아진다.

### 2x 도 같은가

거의 같다. 즉 "큰 값에서 imnet 이 HSR 보다 낮게 낸다" 는 해상도 의존이 약하고, 8x 4 km 입력의 평활화만으로는 설명되지 않는다.

> **데이터** 2021 JJA 검증 · **모델** L2-WMAE-1920-L · **출처** npz 의 `pb` (어댑터 전), `pf` (최종) · **단위** mm/h

| HSR 구간 | n | HSR 평균 | pb, 어댑터 전 | pf, 최종 | pb / HSR |
| --- | --- | --- | --- | --- | --- |
| 0.1~3 | 20,266 | 0.80 | 0.78 | 1.54 | 0.98 |
| 3~10 | 2,614 | 5.09 | 4.55 | 9.30 | 0.89 |
| 10~20 | 301 | 13.04 | 10.13 | 18.75 | 0.78 |
| 20~80 | 49 | 25.51 | 18.18 | 29.06 | 0.71 |

- 전체 평균: HSR 1.10, pb 1.01, pf 1.99, ASOS 1.87. corr(pb, HSR) = 0.916.
- 8x 의 p_k / HSR (10~15: 0.76, 15~25: 0.70) 와 2x 의 pb / HSR (10~20: 0.78, 20~80: 0.71) 이 거의 같다.
- 단 모델과 기간이 다르다 (8x 는 2019 학습 샘플, 2x 는 2021 검증). 참고용 비교다.

### 미확인

- imnet 이 큰 값에서 낮게 내는 원인. 후보: 사전학습 MSE 의 평균 회귀, HSR 500 m 극값의 샘플링/잡음, 시간 평균 효과. 확인 계획은 7절.
- 코드 주의: `models/liif_baseline.py` 의 `le_preds` 주석은 "adapter 전" 이지만 실제로는 어댑터 적용 후 값이다 (어댑터 호출은 219행).

## 6. 실험 현황

2026-10-05 22:20 KST 기준 GPU 2 에서 두 학습이 동시에 돌고 있고, 두 설정은 아직 실행 전이다. 진행률은 `ps` 와 각 run 의 `log.txt` 스냅샷이다.

| run | 상태 | 진행 | 목적 | 학습 스크립트 |
| --- | --- | --- | --- | --- |
| L2-MSE-1920 | 학습 중 (PID 3649680) | epoch 333/600, epoch 당 약 37 초, 총 약 3.1 h | L8-MSE-1920 과 같은 레시피로 해상도 효과만 비교 | `train_liif_finetune.py` |
| T2-topo 사전학습 | 학습 중 (PID 3032636) | epoch 223/300, epoch 당 약 3.3 분, 남은 시간 약 2.4 h | 파인튜닝 전 단계. 지형(GMTED2010) 1채널을 imnet 입력에 추가 | `train_liif.py` |
| L8-L1-1920 | 설정만 | - | MSE / L1 / WMAE 세 손실 비교 | `train_liif_finetune.py` |
| T2-WMAE-1920 | 설정만 | - | T8-WMAE-1920 과 해상도 비교 | `train_liif_finetune_vzero_wmae.py` |
| T8-WMAE-1920 | 중단된 것으로 보임 | 653/800 에서 로그 끝, 해당 프로세스 없음 | (완료된 체크포인트는 3절에서 사용) | `train_liif_finetune_vzero_wmae.py` |

실행 명령은 모두 `$LIIF` 에서 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python <스크립트> --config <yaml> --gpu 2` 형식이다. config 와 저장 위치는 8절 부록에 있다.

### run 별 메모

**L2-MSE-1920**

- 설정: 2x LIIF epoch 300 resume → epoch_max 600, 어댑터 32-16-32, lr 1e-3, MSE, batch 64, inp_size 48.
- 중간 지표 (정규화 단위): best val loss 0.0012, 최근 0.0013~0.0017. val psnr 은 33~35 에서 흔들리고 가끔 inf. 수렴 판단은 아직 이르다.
- 끝나면: dump 스크립트를 2x 용으로 변환 → hexbin → 3절 표에 행 추가.

**T2-topo 사전학습**

- config: `climate_edsr_radar_hsr_2x_minmax_lte_topo_cap63clip.yaml`. 학습 2019~2022 JJA, 검증 2023 JJA.
- HSR cap63 병합본을 74.8784 로 정규화 후 1 로 clip, obscrop. lte, `extra_in_dim: 1`, lr 1e-4, batch 16, augment true.
- 최근 train loss 0.0004, val psnr 약 63. 학습 데이터 61 GB 라 `load_into_memory: false`.

**L8-L1-1920**

- L8-MSE-1920 yaml 과 실제 설정 차이는 `loss: l1` 한 줄뿐이다 (diff 확인). 검증 손실도 같은 종류다 (`test.py:70`).
- 주의: `epoch-best-loss` 는 run 마다 자기 손실로 고르므로 기준이 다르다. `epoch-best` (PSNR) 도 같이 볼 것.
- 주의: WMAE run 은 `train_liif_finetune_vzero_wmae.py` 로 돌렸다. 이 스크립트와 `train_liif_finetune.py` 가 손실 외에 다른지는 비교하지 않았다.

**T2-WMAE-1920**

- T8-WMAE-1920 yaml 에서 학습 LR, 검증 LR, resume, epoch_max 만 바꿈. `adapter_only` 없음 (4절 결과에 따라 coef/freq/phase 같이 학습).
- resume: 2x LTE epoch 300 (모델 구조 EDSR + MLP 256x4, 8x LTE 와 동일 확인) → epoch_max 600.
- 시작 후 확인: `log.txt` 의 `Trainable parameters:` 에 coef/freq/phase/add_layer 가 `[TRAIN]` 으로 찍히는지, 첫 1~2 epoch heavy ratio 가 0 근처로 붕괴하지 않는지.

### 주의

- epoch_max 는 resume epoch 보다 커야 한다. 같거나 작으면 학습 루프가 에러 없이 건너뛴다.
- GPU 2 를 두 학습이 공유해 속도가 느려질 수 있다. GPU 1 은 다른 프로세스가 69 GB 사용 중이었다 (내용 미확인).

## 7. TODO

우선순위 순서다.

**1. 배율(2x vs 8x) 효과** — 조건이 같은 짝 두 개로 본다.

- [ ] L2-MSE-1920 완료 후 dump → hexbin → 3절 표에 추가, L8-MSE-1920 과 비교 (LIIF, MSE)
- [ ] T2-WMAE-1920 실행 후 T8-WMAE-1920 과 비교 (LTE, WMAE). 시작 직후 학습 파라미터 목록과 heavy ratio 확인
- [ ] 두 짝에서 r, bias, ratio(ASOS≥10, ≥20) 차이가 같은 방향인지 확인

**2. imnet 과소모의 원인**

- [ ] 모든 모델의 어댑터 전 출력 / HSR 비율을 같은 2021 JJA 데이터, 같은 HSR 구간으로 표 하나에 정리 (현재 8x 는 2019 학습 샘플, 2x 는 2021 검증이라 직접 비교 불가)
- [ ] 사전학습 모델 자체가 HSR 500 m 를 큰 값에서 과소 재현하는지 확인 (사전학습 검증 데이터, 구간별 pred / HR)
- [ ] 원인 후보 점검: 사전학습 MSE 의 평균 회귀, HSR 500 m 극값의 샘플링/잡음, 시간 평균 효과, remapbil 로 LR 을 만들 때 깎이는 피크
- [ ] 어댑터가 값을 낮추는 36% 샘플의 p_k 구간 확인

**3. 손실 비교**

- [ ] L8-L1-1920 실행 후 MSE / L1 / WMAE 비교 (`epoch-best` 기준도 함께)
- [ ] 강한 비 (ASOS≥20, ratio ≤ 0.62) 개선을 위한 손실/가중치 설계

**4. 기타**

- [ ] T8-WMAE-1920 중단 여부 확인, 필요하면 800 까지 재개
- [ ] 체크포인트 선택과 분리된 독립 테스트 기간 마련
- [ ] 입력 파일 이름 정리 (8절 표)

## 8. 부록

### 8.1 경로 약어

| 약어 | 경로 |
| --- | --- |
| `$LIIF` | `/home/inhye_yoo/deeplearning/inr/liif` |
| `$HSR` | `/home/inhye_yoo/data/radar_HSR/hsr_hourly_precip_nc/merge` |
| `$OBS` | `/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs` |
| `$EVAL` | `$LIIF/inference_tools/station_eval` |

### 8.2 run 별 config 와 저장 위치

config 는 `$LIIF/configs/train_climate/`, 저장은 `$LIIF/save/` 아래다. 공통 접두어 `climate_edsr_radar_hsr_` 는 `…` 로 줄였다.

| run | config | 저장 디렉터리 |
| --- | --- | --- |
| L8-MSE-1920 | `…8x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020.yaml` | 같은 이름 앞에 `_` |
| L8-WMAE-1920 | `…8x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020_wmae.yaml` | 같은 이름 앞에 `_` |
| T8-WMAE-1920 | `…8x_minmax_lte_ft_mlp_1e-3_lag1h_2019_2020_wmae.yaml` | `_…8x_minmax_lte_ft_mlp_1e-3_lag1h_2019_2020_wmae/` |
| T8-WMAE-1920-AO | `…_wmae_adapteronly.yaml` | `…_wmae_adapteronly/` |
| L2-MSE-1920 | `…2x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020.yaml` | `_…2x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020/` |
| L8-L1-1920 | `…8x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020_l1.yaml` | `_…8x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020_l1/` (예상) |
| T2-WMAE-1920 | `…2x_minmax_lte_ft_mlp_1e-3_lag1h_2019_2020_wmae.yaml` | `_…2x_minmax_lte_ft_mlp_1e-3_lag1h_2019_2020_wmae/` (예상) |
| T2-topo 사전학습 | `…2x_minmax_lte_topo_cap63clip.yaml` | - |

resume 원본: `$LIIF/save/_climate_edsr_radar_hsr_{8x,2x}_minmax_{liif,lte}/epoch-last.pth`.

### 8.3 데이터 파일

| 용도 | 파일 |
| --- | --- |
| 2x 학습 LR | `$HSR/regrid/HSR_train_norm_2x_obscrop_JJA_2019_2021_1440x1152.nc` |
| 학습 HR | `$HSR/HSR_2019_2020JJA_500m_norm_obscrop.nc` |
| 학습 obs | `$OBS/obs_hourly_HSR_500m_2019_2020JJA_obscrop_lag1h_train.csv` + `meta_…_train.json` |
| 검증 2x LR | `$HSR/regrid/HSR_val_norm_2x_obscrop.nc` |
| 검증 HR | `$HSR/regrid/HSR_val_500m_norm_obscrop.nc` |
| 검증 obs | `$OBS/obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv` |

### 8.4 이름과 내용이 다른 파일

| 파일 | 이름이 말하는 것 | 실제 내용 | 제안 |
| --- | --- | --- | --- |
| `HSR_train_norm_2x_obscrop_JJA_2019_2021_1440x1152.nc` | 2019~2021, 1440x1152 | 2019+2020 JJA, 4416 h, 마지막 2020-08-31 23:00, 617x663 | `HSR_2019_2020JJA_norm_2x_obscrop.nc` |
| `obs_hourly_HSR_500m_2019JJA_obscrop_lag1h_val.csv` | 2019 JJA | 2021-06-01 ~ 2021-08-31, 211,968 행 | 미정 |

원래 2x 학습 LR 이던 `HSR_train_norm_2x.nc` 는 `regrid/HSR_norm_2x_JJA_2019_1441x1153_train.nc` 로 개명된 것으로 보인다.

### 8.5 그림과 스크립트

| 항목 | 경로 |
| --- | --- |
| hexbin 그림 | `$EVAL/figures/hexbin_hsr_preds_vs_asos/` |
| 예측 npz | `$EVAL/preds_npz/` |
| hexbin 스크립트 | `$LIIF/inference_tools/codes/hexbin_hsr_preds_vs_asos_2021jja_*.py` |
| 어댑터 곡선 스크립트 | `…/codes/adapter_fold_hsr_8x_2019_2020.py`, `adapter_fold_hsr_8x_lag1h.py` |
| 단계 분해 / 케이스 | scratchpad `stages.py`, `case.py` |

L2 모델 dump 스크립트가 쓰는 비교용 ASOS csv 는 열어 보지 않았다. npz 의 obs 값과 n 은 8x 와 같다.

### 8.6 작업 로그

| 날짜 | 항목 | 내용 |
| --- | --- | --- |
| 2026-10-05 | T2-WMAE-1920 config | yaml 작성, 미실행 |
| 2026-10-05 | L8-L1-1920 config | yaml 작성, 미실행 |
| 2026-10-05 | LTE 미세조정 범위 비교 | `log.txt` 기반 비교 (4절) |
| 2026-10-05 | L8-MSE-19 hexbin | `hexbin_hsr_preds_vs_asos_2021jja_8x_2019only.py` 작성 → `…_8x_2019only_epoch-best-loss_nofilter.png`. liif env 파이썬이 중첩 f-string 을 못 읽어 그 줄만 수정 |
| 2026-10-05 | L2-MSE-1920 | yaml 작성, 사용자가 직접 실행 시작 |
| 2026-10-05 | 어댑터 on/off 추론 | `stages.py` 작성, 28,468 샘플 (5절). `case.py` 는 세션 종료로 중단 |