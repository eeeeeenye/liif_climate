""" [새 파일] MLP adapter ASOS finetune — 검증된 무강수(verified zero) 라벨 + Weighted-MAE.
    train_liif_finetune.py 를 복사해 손실/검증만 바꾼 것 (기존 파일은 수정하지 않음).

    변경점
      - config loss: 'wmae' 지원.  L = sum(w*|pred-gt|)/sum(w),  w = clip(y_true, MIN, MAX)  (mm/h 기준 MIN/MAX 를 config wmae 로 지정)
          Wang, F., Tian, D., & Carroll, M. (2023). Customized deep learning for precipitation bias correction and downscaling.
          Geosci. Model Dev., 16, 535-556. https://doi.org/10.5194/gmd-16-535-2023   (Sec. 3.1.3, Eq. 1)
      - 라벨 csv = make_hsr_hourly_obs_vzero.py 산출(일합과 시간합이 일치하는 지점-일의 미기록 시각 = 0). config 의 obs_path_3 만 바꾸면 됨.
      - 검증: 학습과 같은 손실 + mm/h 진단(전체 MAE/bias/RMSE, 무강수 라벨 평균 예측=오프셋, 강한 비 bias/ratio). best.pth = val r 최대, best-loss.pth = val 손실 최소.
    사용: python train_liif_finetune_vzero_wmae.py --config configs/train_climate/climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h_vzero_wmae.yaml --gpu 2
"""
""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

""" [새 파일] 문맥 어댑터(lte_ctx) 지원 + 구간별 편향 손실(binloss) 을 더한 ASOS finetune.
    train_liif_finetune_vzero_wmae.py 를 복사해 아래만 바꾼 것 (원본 파일은 수정하지 않음). 실험 노트: exp_ctx_binloss/EXPERIMENT_NOTES.md

    변경점
      1) models_ctx 를 import -> config model.name 에 'lte_ctx' (문맥 어댑터) 사용 가능. 기존 'lte'/'liif' 도 그대로 사용 가능.
      2) config binloss 를 주면 손실에 항 2개 추가 (ASOS 강도 구간별로 배치 안에서 계산, 정규화 공간):
           L = L_wmae + lambda_abs * L_balMAE + lambda_bias * L_bias
           L_balMAE = mean_b [ mean_{i in b} |pred_i - gt_i| ]      구간마다 같은 비중 (wmae 는 약한 비를 1/30 비중으로 취급)
           L_bias   = mean_b | mean_{i in b} (pred_i - gt_i) |      구간별 평균 편향 -> 약한 비 과대와 강한 비 과소를 동시에 겨냥
           binloss: {edges_mm: [0.001,0.5,1,2,5,10,20], lambda_abs: 1.0, lambda_bias: 1.0, min_count: 2}   (구간 b: gt<=edges[0] 이면 '=0', 이후 (e[k-1], e[k]], 마지막은 20 초과)
      3) 검증: val 손실 = (wmae 검증 손실) + lambda * (전체 검증셋 기준 같은 항들).  val 진단에 ASOS 구간별 '예측평균/ASOS평균' 출력.
    사용: (liif 루트에서)  python exp_ctx_binloss/train_ctx_binloss.py --config exp_ctx_binloss/configs/<cfg>.yaml --gpu 2
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                                  # datasets/models/utils/test 는 liif 루트에서 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # models_ctx

import argparse                                      # 터미널에서 입력하는 실행 옵션(--config, --gpu 등)을 파싱하기 위한 라이브러리
import os                                            # 파일 경로 생성, 환경변수(CUDA_VISIBLE_DEVICES) 설정 등에 사용하는 라이브러리
import random                                        # 파이썬 기본 난수 seed를 고정하기 위해 사용하는 라이브러리
import numpy as np                                   # numpy 난수 seed 고정 및 배열 연산에 사용하는 라이브러리

import yaml                                          # yaml 형식의 config 파일을 읽고 저장하기 위한 라이브러리
import torch                                         # PyTorch의 tensor 연산, GPU 연산, 모델 저장/로드에 사용하는 라이브러리
import torch.nn as nn                                # loss function, DataParallel 등 신경망 관련 모듈을 사용하기 위한 별칭
from tqdm import tqdm                                # 학습 loop 진행률을 progress bar 형태로 보여주는 라이브러리
from torch.utils.data import DataLoader              # Dataset을 batch 단위로 묶어주는 PyTorch DataLoader
from torch.optim.lr_scheduler import MultiStepLR     # 특정 epoch milestone마다 learning rate를 줄이는 scheduler

import datasets                                      # 프로젝트 내부의 datasets 모듈: config에 맞는 dataset 객체를 생성할 때 사용
import models                                        # 프로젝트 내부의 models 모듈: config에 맞는 모델 객체를 생성할 때 사용
import models_ctx  # noqa: F401  (registry 에 lte_ctx 등록)
import utils                                         # 프로젝트 내부의 utils 모듈: optimizer, logger, timer, averager 등을 만들 때 사용
from test import eval_psnr                           # validation 단계에서 PSNR과 validation loss를 계산하는 평가 함수
import torch.nn.functional as F                      # padding 등 함수형 PyTorch 연산을 사용하기 위한 모듈

# 가장 큰 샘플 길이를 따라감
"""
ASOS 자료는 날짜마다 관측 스테이션의 개수가 달라 각 샘플의 길이가 서로 다르다.
배치 단위로 학습하기 위해 배치 내 최대 스테이션 개수(max_len)에 맞춰
coord, gt, cell을 padding(일정한 길이로 만들고 부족한 부분을 0으로 채우기)하고,
실제 데이터 위치를 구분하기 위한 mask(padding한 부분을 알려주는 mask 배열)를 생성한다.
"""

def pad_collate(batch):
    lengths = [b['coord'].shape[0] for b in batch]          # 각 샘플의 coord 개수, 즉 해당 샘플이 가진 station/query point 개수를 리스트로 저장
    max_len = max(lengths)                                  # 배치 안에서 가장 긴 station/query point 개수를 구해 padding 기준 길이로 사용

    # 만약 가장 긴 길이가 0이라면 None을 반환
    if max_len == 0:                                        # 배치 내 모든 샘플에 유효한 coord가 하나도 없는 경우 학습에 사용할 수 없으므로
        return None                                         # train loop에서 이 batch를 건너뛸 수 있도록 None을 반환

    # 각 데이터 배열 생성 (패딩처리할 데이터)
    padded_inp = []                                         # inp는 모든 샘플이 같은 격자 형태라고 가정하므로 padding 없이 stack하기 위해 저장
    padded_gt = []                                          # gt는 station/query point 길이가 다를 수 있으므로 padding 후 저장
    padded_coord=[]                                         # coord는 station/query point 길이가 다를 수 있으므로 padding 후 저장
    padded_cell = []                                        # cell은 coord와 같은 point 단위 정보이므로 padding 후 저장
    padded_level=[]                                         # level은 샘플별 고도/레벨 정보로, 기존 shape를 유지해 stack하기 위해 저장
    masks = []                                              # 실제 데이터와 padding 데이터를 구분하는 mask를 저장

    # 길이만큼 반복 처리
    for b, q in zip(batch, lengths):                         # batch의 각 샘플 b와 해당 샘플의 실제 길이 q를 함께 순회
        # 각 데이터 가져오기
        # 모델 입력 이미지/격자 데이터(피처맵) 
        inp = b['inp']          # dim : [1, 50, 50]
        # 정답 관측값 또는 target 값                                                           
        gt = b['gt']            # dim : [station, 2]  
        # 예측해야 할 위치 좌표                                                    
        coord = b['coord']      # dim : [station, 1]  
        # 각 좌표가 대표하는 cell 크기 또는 해상도 정보                                       
        cell = b['cell']        # dim : [station,2]
        # 고도/level 등 추가 입력 정보 (topo_nc_path 없는 config면 None)
        level = b['level']      # dim : [1, 100, 100] 또는 None
        # 패드 길이
        pad_len = max_len-q                                 # 현재 샘플 길이 q를 배치 최대 길이 max_len에 맞추기 위해 추가해야 하는 길이

        # coord, gt, cell 패딩 채우기
        # coord의 첫 번째 차원(point 개수)을 max_len까지 0으로 padding / dim : [max_len,2]
        coord_padded = F.pad(coord, (0,0,0,pad_len))
        # gt의 첫 번째 차원(point 개수)을 max_len까지 0으로 padding / dim : [max_len,1]     
        gt_padded = F.pad(gt, (0, 0, 0, pad_len))
        # cell의 첫 번째 차원(point 개수)을 max_len까지 0으로 padding / dim : [max_len,2]              
        cell_padded = F.pad(cell, (0, 0, 0, pad_len))        

        # 배열 길이만큼의 mask 배열 생성
        mask = torch.zeros(max_len, dtype=torch.float32)     # padding을 포함한 max_len 길이의 mask를 0으로 초기화
        # 실제 데이터가 있는 위치를 1.0으로 할당
        # 1: 실제 스테이션 데이터, 0: 패딩으로 추가한 가짜 데이터
        mask[:q] = 1.0                                      # dim : [max_len]  

        # 이 코드에서 append는 샘플 하나를 리스트에 추가하는 것!
        # 예시 입니다! [inp1(shape: 1,50,50), inp2(shape:1, 50,50), ...] -> 그냥 리스트이므로 axis가 없습니다!
        padded_inp.append(inp)                              # 현재 샘플의 inp를 리스트에 추가
        padded_gt.append(gt_padded)                         # padding된 gt를 리스트에 추가
        padded_coord.append(coord_padded)                   # padding된 coord를 리스트에 추가
        padded_cell.append(cell_padded)                     # padding된 cell을 리스트에 추가
        padded_level.append(level)                          # level 정보를 리스트에 추가
        masks.append(mask)                                  # 현재 샘플의 mask를 리스트에 추가

    # 마스크와 패딩을 각각 만들어 저장
    batch_out = {                                           # DataLoader가 최종적으로 반환할 batch dictionary 생성
            'inp' : torch.stack(padded_inp, dim=0),          # 샘플별 inp를 batch 차원(dim=0)으로 쌓아 [B, ...] 형태로 만듦 dim : [16(batch), 1(ch), 50(H), 50(W)]
            'gt' : torch.stack(padded_gt, dim=0),            # padding된 gt를 [B, max_len, 1(channel)] 형태로 쌓음
            'coord': torch.stack(padded_coord, dim=0),       # padding된 coord를 [B, max_len, 2(x, y)] 형태로 쌓음
            'cell': torch.stack(padded_cell, dim=0),         # padding된 cell을 [B, max_len, 2(x, y)] 형태로 쌓음
            # topo_nc_path 없는 config(model B radar_hsr_2x 등)면 모든 샘플의 level이 None
            # -> stack하지 않고 None 그대로 둠 (model.forward(inp, topo=None, ...)와 동일 규약)
            'level': torch.stack(padded_level, dim=0) if padded_level[0] is not None else None,
            'mask': torch.stack(masks, dim=0),               # mask를 [B, max_len] 형태로 쌓음
        }

    return batch_out                                        # padding과 mask 처리가 완료된 batch를 반환

# 데이터 로더 함수
def make_data_loader(spec, tag=''):
    """
        spec : train, val 데이터 정보와 batch_size 등 DataLoader 생성에 필요한 설정
        tag : train인지 val인지 구분하는 문자열
    """

    if spec is None:                                        # train_dataset 또는 val_dataset 설정이 없는 경우
        return None                                         # 해당 DataLoader를 만들지 않고 None 반환

    # make함수는 config에서 태그를 확인하고 객체를 생성
    # dataset 하나에 spec에 해당하는 객체가 생성됨
    dataset = datasets.make(spec['dataset'])                # config의 dataset 설정을 이용해 기본 dataset 객체 생성
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})  # 기본 dataset을 wrapper로 감싸 학습에 필요한 출력 형태로 변환

    # 데이터셋 사이즈를 프롬프트상에 보여줌
    log('{} dataset: size={}'.format(tag, len(dataset)))     # train/val dataset의 전체 샘플 개수를 log에 출력
    for k, v in dataset[0].items():                          # 첫 번째 샘플의 key와 tensor 값을 확인하면서
        # topo_nc_path 없는 config면 level이 None이라 .shape가 없음
        if torch.is_tensor(v):
            log('  {}: shape={}'.format(k, tuple(v.shape)))  # 각 항목(inp, gt, coord 등)의 shape를 log에 출력
        else:
            log('  {}: {}'.format(k, v))

    # PyTorch DataLoader 객체 생성
    loader = DataLoader(                                     
        dataset,
        # batch로 묶을 dataset 지정                                             
        batch_size=spec['batch_size'],                      # 데이터를 병렬로 불러올 worker 프로세스 수
        shuffle=(tag == 'train'),                            # train일 때만 데이터를 섞고, val일 때는 순서를 유지
        num_workers=8,                                       # 데이터를 병렬로 불러올 worker 프로세스 수
        pin_memory=True,                                     # GPU 전송 속도를 높이기 위해 pinned memory 사용
        collate_fn=pad_collate)                              # 길이가 다른 샘플을 padding하기 위해 직접 만든 collate 함수 사용
    return loader                                            # 생성된 DataLoader 반환


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')  # config의 train_dataset 설정으로 train loader 생성
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')        # config의 val_dataset 설정으로 validation loader 생성
    return train_loader, val_loader                              # train loader와 validation loader를 함께 반환

# train 준비 함수
def prepare_training():
    # resume이라는 태그가 config 상에 있다면, 그 path의 모델을 불러옴
    # 중간에 학습이 예기치 않게 멈추거나 학습을 재시작할 때 사용 (resume)
    if config.get('resume') is not None:                         # config에 resume 경로가 있으면 저장된 checkpoint에서 이어서 시작
        sv_file = torch.load(config['resume'])                    # resume 경로의 .pth 파일을 로드하여 저장된 학습 상태를 가져옴
        # model을 불러오는데, 모델 내의 저장된 w, b 사용
        # sd : 학습된 파라미터를 저장한 딕셔너리
        pretrained_sd = sv_file['model']['sd']                    # checkpoint 안에서 모델의 state_dict(weight, bias 등)를 꺼냄

        # 모델 객체 생성
        model = models.make(config['model']).cuda()               # config에 정의된 모델 구조를 생성하고 GPU로 이동

        # pretrained weight를 현재 모델에 로드
        # strict=False: 현재 모델에 새로 추가된 layer가 있어도 일치하는 parameter만 로드
        missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)  # 이름과 shape가 맞는 parameter만 로드하고 불일치 항목은 허용

        # encoder와 imnet은 freeze하고, add_layer가 있으면 add_layer만 학습 가능하게 설정하는 코드
        # requires_grad=False이면 backward 때 gradient가 계산되지 않아 해당 파라미터가 업데이트되지 않음
        for p in model.encoder.parameters():                      # 모델의 encoder 부분 파라미터를 하나씩 순회
            p.requires_grad = False                               # encoder는 pretrained 값을 유지하도록 학습에서 제외

        for p in model.imnet.parameters():                        # 모델의 implicit decoder/imnet 부분 파라미터를 하나씩 순회
            p.requires_grad = False                               # imnet도 pretrained 값을 유지하도록 학습에서 제외

        if model.add_layer is not None:                           # 보정층/additional layer가 모델에 존재하는 경우
            for p in model.add_layer.parameters():                 # add_layer의 파라미터를 하나씩 순회
                p.requires_grad = True                            # add_layer만 학습되도록 gradient 계산을 활성화

        # config 'adapter_only: true' 면 encoder/imnet 외 나머지(LTE 의 coef/freq/phase 등)도 얼리고 add_layer 만 학습.
        # 기본(false)은 기존 동작 그대로라 이전 런과 재현성이 유지됨.
        if config.get('adapter_only', False):
            assert model.add_layer is not None, 'adapter_only 인데 add_layer(layer_spec)가 없음'
            for name, p in model.named_parameters():
                p.requires_grad = name.startswith('add_layer')

        # optimizer 생성
        optimizer = utils.make_optimizer(                         # 프로젝트 utils 함수로 optimizer 객체 생성
            model.parameters(),                                   # 모델 전체 파라미터를 넘기지만, requires_grad=True인 것만 실제 업데이트됨
            config['optimizer']                                   # optimizer 종류, learning rate 등 config 설정 사용
        )

        # 학습시작 전 파라미터 직접 확인 (freeze/adapter가 의도대로 됐는지 log에 남김)
        log("=" * 50)                                              # 출력 구분선
        log("Trainable parameters:")                               # trainable/frozen parameter 목록 출력 시작 표시
        for name, p in model.named_parameters():                   # 모델의 모든 파라미터 이름과 tensor를 순회하며 학습 여부 확인
            if p.requires_grad:                                   # gradient 계산이 켜진 파라미터라면
                log(f" [TRAIN] {name}")                            # 학습되는 파라미터로 출력
            else:                                                 # gradient 계산이 꺼진 파라미터라면
                log(f" [FROZEN] {name}")                           # freeze된 파라미터로 출력
        log("=" * 50)                                              # 출력 구분선

        # total parameter를 확인하고 학습할 수 있는 파라미터만 확인
        # 최종적으로 total에서 학습가능하지 않은 파라미터 빼고 확인
        total = sum(p.numel() for p in model.parameters())         # 모델 전체 파라미터 개수 계산
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 실제 학습되는 파라미터 개수 계산
        frozen = total - trainable                                 # 전체 파라미터에서 학습되는 파라미터를 빼 freeze된 개수 계산
        log(f"frozen param : {frozen}")
        log(f"trainable param : {trainable}")

        # epoch을 저장된 파일에서 마지막으로 저장된 epoch 다음부터 시작
        epoch_start = sv_file['epoch'] + 1                         # checkpoint가 저장된 epoch 다음 epoch부터 학습 재개

        # multi step learning rate 스케줄러 설정이 config에 있다면 실행
        if config.get('multi_step_lr') is None:                    # learning rate scheduler 설정이 없으면
            lr_scheduler = None                                    # scheduler를 사용하지 않음
        else:                                                      # scheduler 설정이 있으면
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])  # milestone과 gamma 설정으로 MultiStepLR 생성
            for _ in range(epoch_start - 1):                       # resume 이전 epoch 수만큼 scheduler를 진행시켜
                lr_scheduler.step()                                # learning rate 상태를 이어받은 epoch에 맞춤

    else:
        # 만약 resume 설정이 없다면 그대로 새로운 모델 객체 생성
        model = models.make(config['model']).cuda()                # 새 모델을 config대로 생성하고 GPU로 이동
        # 바로 옵티마이저 생성 후 epoch 1에서부터 시작
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])  # 새 모델 파라미터를 업데이트할 optimizer 생성
        epoch_start = 1                                            # 새 학습이므로 1 epoch부터 시작

        if config.get('multi_step_lr') is None:                    # scheduler 설정이 없으면
            lr_scheduler = None                                    # scheduler를 사용하지 않음
        else:                                                      # scheduler 설정이 있으면
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])  # MultiStepLR scheduler 생성

    return model, optimizer, epoch_start, lr_scheduler             # 학습에 필요한 모델, optimizer, 시작 epoch, scheduler 반환



def _wm():
    """wmae 설정 -> (w_min, w_max, normalize_by_weight). config: wmae: {w_min_mm, w_max_mm, normalize_by_weight}, 정규화 공간으로 환산."""
    c = config.get('wmae', {}) or {}
    pm = config['physical_norm']['max'] - config['physical_norm'].get('min', 0.0)
    return c.get('w_min_mm', 1.0) / pm, c.get('w_max_mm', 30.0) / pm, c.get('normalize_by_weight', True)


def point_loss(pred, gt):
    """점별 손실 + 가중치.  wmae: w = clip(y_true, MIN, MAX) (Wang, Tian & Carroll 2023, GMD 16, 535, Eq.1), 손실 = w*|pred-gt|."""
    kind = config.get('loss', 'mse')
    if kind == 'wmae':
        wmin, wmax, _ = _wm()
        w = torch.clamp(gt.detach(), min=wmin, max=wmax)
        return w * torch.abs(pred - gt), w
    if kind == 'l1':
        return torch.abs(pred - gt), torch.ones_like(gt)
    return (pred - gt) ** 2, torch.ones_like(gt)


@torch.no_grad()
def evaluate(loader, model):
    """검증: 학습과 같은 손실 + mm/h 진단.  무강수(gt=0) 라벨에서 예측 평균(=상수 오프셋)과 강한 비(>=10mm/h) bias 를 같이 본다."""
    model.eval()
    kind = config.get('loss', 'mse'); _, _, wnorm = _wm()
    pm = config['physical_norm']['max'] - config['physical_norm'].get('min', 0.0)
    num = den = 0.0; P = []; G = []
    for batch in tqdm(loader, leave=False, desc='val'):
        if batch is None:
            continue
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)
        pred = model(batch['inp'], batch.get('level'), batch['coord'], batch['cell'])
        gt = batch['gt']; m = batch['mask'].unsqueeze(-1)
        l, w = point_loss(pred, gt)
        num += (l * m).sum().item(); den += ((m * w).sum() if (kind == 'wmae' and wnorm) else m.sum()).item()
        mb = m.bool().expand_as(gt)
        P.append(pred.clamp(0, 1)[mb].cpu()); G.append(gt[mb].cpu())
    p = torch.cat(P).numpy() * pm; g = torch.cat(G).numpy() * pm            # mm/h
    val_loss = num / max(den, 1e-12)
    e = p - g; dry = g == 0; hv = g >= 10
    r = float(np.corrcoef(p, g)[0, 1]) if p.std() > 0 and g.std() > 0 else 0.0
    extra = (f"mm/h MAE={np.abs(e).mean():.3f} bias={e.mean():+.3f} RMSE={np.sqrt((e ** 2).mean()):.3f} | dry(gt=0,n={int(dry.sum())}) mean_pred={p[dry].mean():.3f} "
             f"| heavy(>=10,n={int(hv.sum())}) bias={e[hv].mean() if hv.any() else float('nan'):+.2f} ratio={p[hv].sum() / max(g[hv].sum(), 1e-9) if hv.any() else float('nan'):.2f}")
    BL = config.get('binloss') or {}
    la, lb, btxt = bin_eval_np(p, g)
    if BL:
        val_loss = val_loss + BL.get('lambda_abs', 0.0) * la / pm + BL.get('lambda_bias', 0.0) * lb / pm
    extra += f" | binMAE={la:.3f} binBias={lb:.3f} | pred/ASOS by bin [{btxt}]"
    return r, val_loss, extra


def bin_terms(pred, gt, mask):
    """학습용(미분 가능) 구간별 항.  반환 (L_balMAE, L_bias), 정규화 공간. binloss 설정이 없으면 0."""
    c = config.get('binloss')
    if not c:
        z = pred.new_zeros(()); return z, z
    pm = config['physical_norm']['max'] - config['physical_norm'].get('min', 0.0)
    edges = torch.tensor([x / pm for x in c['edges_mm']], device=pred.device, dtype=pred.dtype)
    m = mask.unsqueeze(-1).bool().expand_as(gt)
    p, g = pred[m], gt[m]
    idx = torch.bucketize(g.detach(), edges)               # boundaries[i-1] < g <= boundaries[i]
    a, b = [], []
    for k in range(len(edges) + 1):
        s = idx == k
        if int(s.sum()) >= c.get('min_count', 2):
            e = p[s] - g[s]; a.append(e.abs().mean()); b.append(e.mean().abs())
    if not a:
        z = pred.new_zeros(()); return z, z
    return torch.stack(a).mean(), torch.stack(b).mean()


def bin_eval_np(p, g):
    """검증용(numpy, mm/h).  반환 (L_balMAE, L_bias [mm/h], 구간별 진단 문자열)."""
    c = config.get('binloss') or {'edges_mm': [0.001, 0.5, 1, 2, 5, 10, 20], 'min_count': 2}
    edges = np.array(c['edges_mm']); idx = np.digitize(g, edges, right=True)
    names = ['=0'] + ['%g-%g' % (edges[k - 1], edges[k]) for k in range(1, len(edges))] + ['>%g' % edges[-1]]
    a, b, txt = [], [], []
    for k in range(len(edges) + 1):
        s = idx == k
        if s.sum() >= 1:
            e = p[s] - g[s]
            if s.sum() >= c.get('min_count', 2): a.append(np.abs(e).mean()); b.append(abs(e.mean()))
            txt.append('%s:%s' % (names[k], ('%.2fmm' % p[s].mean()) if k == 0 else '%.2f' % (p[s].mean() / max(g[s].mean(), 1e-9))))
    return (float(np.mean(a)) if a else 0.0), (float(np.mean(b)) if b else 0.0), ' '.join(txt)


def train(train_loader, model, optimizer):
    model.train()                                                   # 모델을 train mode로 전환하여 dropout/batchnorm 등이 학습 방식으로 동작하게 함
    # config 의 loss: 'mse'(기본, 기존 동작) | 'l1'.  각 위치별 loss 를 따로 얻기 위해 reduction 없이 선언 (padding mask 적용용)
    kind = config.get('loss', 'mse')                                # 'mse' | 'l1' | 'wmae' (Wang et al. 2023 Eq.1, 정답 강수량으로 가중)
    wmin, wmax, wnorm = _wm()
    train_loss = utils.Averager()                                   # 학습과정에서 각 batch의 loss를 누적하여 평균을 계산하는 객체

    # train 데이터 배치별로 가져오기
    for batch in tqdm(train_loader, leave=False, desc='train'):      # train_loader에서 batch를 하나씩 가져오며 진행률 표시
        # batch가 비어있으면 다음으로 넘기기 (한 시점에 한스테이션도 데이터가 존재하지 않을 경우)
        if batch is None:                                           # pad_collate에서 유효한 point가 없다고 판단한 batch라면
            continue                                                # 해당 batch는 학습하지 않고 다음 batch로 넘어감

        # key, value 가져오기
        for k, v in batch.items():                                  # batch dictionary의 모든 tensor 항목을 순회
            # topo_nc_path 없는 config면 level이 None이라 .cuda()가 없음 (test.py와 동일 규약)
            if torch.is_tensor(v):
                batch[k] = v.cuda()                                 # 각 tensor를 GPU로 이동

        inp = batch['inp']

        # 모델에 train 데이터, 고도, 좌표, cell을 input
        pred = model(inp, batch['level'], batch['coord'], batch['cell'])  # 입력 격자, level, 좌표, cell 정보를 이용해 각 point의 예측값 계산
        gt = batch['gt']

        # loss 적용
        loss, w = point_loss(pred, gt)                              # 점별 손실과 가중치 (mse/l1 은 w=1)

        # mask 함수 가져오기
        # 이전에 구한 pred는 padding이 적용되었기 때문에 mask를 씌워주어야 함
        mask = batch['mask']                                        # 실제 데이터 위치는 1, padding 위치는 0인 mask 가져오기
        mask_expanded = mask.unsqueeze(-1)                          # 마지막 차원을 하나 추가하여 loss shape [B,max_len,D]와 broadcast 가능하게 맞춤
        masked_loss = loss * mask_expanded                          # mask가 1인 실제 위치의 loss만 남기고, 0인 padding 위치의 loss는 0으로 만듦

        loss = masked_loss.sum() / ((mask_expanded * w).sum() if (kind == 'wmae' and wnorm) else mask_expanded.sum())   # wmae: 가중평균
        BL = config.get('binloss') or {}
        if BL:
            l_abs, l_bias = bin_terms(pred, gt, mask)
            loss = loss + BL.get('lambda_abs', 0.0) * l_abs + BL.get('lambda_bias', 0.0) * l_bias   # 구간별 항 (정규화 공간)
        train_loss.add(loss.item())                                 # 현재 batch loss 값을 Averager에 추가하여 epoch 평균 loss 계산에 사용

        optimizer.zero_grad()                                       # 이전 batch에서 계산되어 남아있는 gradient를 0으로 초기화
        loss.backward()                                             # 현재 loss 기준으로 역전파를 수행하여 trainable parameter의 gradient 계산
        optimizer.step()                                            # optimizer가 gradient를 이용해 weight, bias 등 parameter 업데이트

        pred = None; loss = None                                    # 큰 tensor 참조를 끊어 GPU 메모리 사용을 줄이려는 처리

    return train_loss.item()                                        # 한 epoch 동안 누적한 train loss 평균을 반환


# 최종적으로 모델 train하는 함수들을 모두 사용하는 main 함수
def main(config_, save_path):
    global config, log, writer                                      # 여러 함수에서 config/log/writer를 공유하기 위해 전역 변수로 선언
    config = config_                                                # main 함수로 받은 config를 전역 config에 저장
    log, writer = utils.set_save_path(save_path)                    # 저장 경로를 만들고 log 함수와 TensorBoard writer를 생성
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:     # 학습에 사용한 config를 save_path 아래 config.yaml로 저장하기 위해 파일 열기
        yaml.dump(config, f, sort_keys=False)                       # config 내용을 key 순서를 유지한 채 yaml 파일로 저장

    # train_loader와 val_loader 객체 생성
    train_loader, val_loader = make_data_loaders()                  # config를 기반으로 train/validation DataLoader 생성

    # prepare_training에서 모델과 옵티마이저, 에폭 시작 시점, 스케줄러 생성
    model, optimizer, epoch_start, lr_scheduler = prepare_training() # 학습에 필요한 모델/optimizer/scheduler와 시작 epoch 준비

    # gpu 몇개 사용할 것인지 설정
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))      # CUDA_VISIBLE_DEVICES에 적힌 GPU 번호 개수를 세어 사용할 GPU 수 계산
    if n_gpus > 1:                                                  # 사용할 GPU가 2개 이상이면
        model = nn.parallel.DataParallel(model)                     # DataParallel로 모델을 감싸 여러 GPU에 batch를 나누어 학습

    # 학습과 검증을 관리하기 위한 설정 및 변수 초기화
    epoch_max = config['epoch_max']                                 # 전체 학습을 몇 epoch까지 수행할지 설정
    epoch_val = config.get('epoch_val')                             # validation을 몇 epoch마다 수행할지 설정 # 데이터를 병렬로 불러올 worker 프로세스 수 없으면 validation 생략 가능
    epoch_save = config.get('epoch_save')                           # checkpoint를 몇 epoch마다 저장할지 설정 없으면 주기 저장 생략
    max_val_v = -1e18                                               # 지금까지 가장 높은 validation PSNR을 저장하기 위한 초기값
    min_val_loss = 1e18                                             # 지금까지 가장 낮은 validation loss를 저장하기 위한 초기값

    # 시간 측정을 위한 객체 생성
    timer = utils.Timer()                                           # 데이터를 병렬로 불러올 worker 프로세스 수

    # epoch start부터 설정해둔 에폭까지 반복
    for epoch in range(epoch_start, epoch_max + 1):                 # 시작 epoch부터 epoch_max까지 학습 반복
        t_epoch_start = timer.t()                                   # 현재 epoch 시작 시점의 누적 시간을 기록
        # 프롬프트에 보여줄 epoch 현황
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]         # 현재 epoch 번호와 전체 epoch 수를 log 문자열 리스트에 추가
        # TensorBoard에 learning rate를 기록
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)  # 현재 learning rate를 TensorBoard에 기록

        # train 함수에서 loss 구함
        train_loss = train(train_loader, model, optimizer)          # 한 epoch 동안 train을 수행하고 평균 train loss를 반환받음
        if lr_scheduler is not None:                                # learning rate scheduler가 설정되어 있으면
            lr_scheduler.step()                                     # epoch이 끝난 뒤 scheduler를 한 단계 진행

        # 한번 train 돌때마다 초기화하지 않고 뒤에서 이어서 붙임 (프롬프트에)
        log_info.append('train: loss={:.5f}'.format(train_loss))    # train loss를 log 출력용 문자열에 추가
        writer.add_scalars('loss', {'train': train_loss}, epoch)    # train loss를 TensorBoard에 기록

        if n_gpus > 1:                                              # DataParallel을 사용하는 경우
            model_ = model.module                                   # 실제 원본 모델은 model.module 안에 있으므로 이를 사용
        else:                                                       # 단일 GPU를 사용하는 경우
            model_ = model                                          # 현재 model 객체를 그대로 사용
        # 모델 스펙, 파라미터, 옵티마이저를 불러옴 -> svfile 생성
        model_spec = config['model']                                # 모델 구조 설정 dictionary를 가져옴
        model_spec['sd'] = model_.state_dict()                      # 현재 모델 파라미터 state_dict를 모델 설정에 함께 저장
        optimizer_spec = config['optimizer']                        # optimizer 설정 dictionary를 가져옴
        optimizer_spec['sd'] = optimizer.state_dict()               # 현재 optimizer 상태(momentum 등 포함)를 optimizer 설정에 함께 저장
        sv_file = {                                                 # checkpoint로 저장할 dictionary 생성
            'model': model_spec,                                    # 모델 설정과 모델 파라미터 저장
            'optimizer': optimizer_spec,                            # optimizer 설정과 optimizer 상태 저장
            'epoch': epoch                                          # 현재 저장 시점의 epoch 번호 저장
        }

        # 현재 학습 상태를 파일로 저장하는 코드
        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))  # 매 epoch마다 최신 checkpoint를 epoch-last.pth로 덮어쓰기 저장

        # epoch 저장을 몇에폭마다 저장할 것인지에 대한 정보가 있다면
        # 그 숫자에 따라 에폭을 저장함
        if (epoch_save is not None) and (epoch % epoch_save == 0):  # epoch_save가 설정되어 있고 현재 epoch이 저장 주기에 해당하면
            torch.save(sv_file,                                    # 현재 checkpoint를
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))  # epoch 번호가 들어간 별도 파일명으로 저장

        # Validation을 수행할지 결정하고, Validation 성능을 평가
        # epoch_val마다 validation을 수행
        if (epoch_val is not None) and (epoch % epoch_val == 0):    # epoch_val 설정이 있고 현재 epoch이 validation 주기에 해당하면 평가 수행
            if n_gpus > 1 and (config.get('eval_bsize') is not None):  # multi-GPU이고 eval_bsize가 별도로 설정된 경우
                model_ = model.module                               # 평가 함수에 DataParallel wrapper가 아닌 실제 모델을 넘김
            else:                                                   # 그 외 경우에는
                model_ = model                                      # 현재 model 객체를 평가에 사용
            val_res, val_loss, extra = evaluate(val_loader, model_)   # val_res = mm/h 상관 r (best.pth 기준), val_loss = 학습과 같은 손실

            # 학습 정보 log로 보여줌 (프롬프트에)
            log_info.append('val: r={:.4f}'.format(val_res))
            log_info.append('val loss: {:.5f} | {}'.format(val_loss, extra))
            writer.add_scalars('psnr', {'val': val_res}, epoch)     # validation PSNR을 TensorBoard에 기록
            # 만약 현재 나오는 val PSNR이 이전 best보다 크다면 best PSNR 업데이트
            if val_res > max_val_v:                                # 현재 validation PSNR이 지금까지의 최고 PSNR보다 높으면
                max_val_v = val_res                                # 최고 PSNR 값을 현재 값으로 갱신
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))  # best PSNR 모델로 저장
            # 만약 val loss가 min 보다 작다면 min_val 업데이트 그리고 epoch 저장
            if val_loss < min_val_loss:                            # 현재 validation loss가 지금까지의 최소 loss보다 낮으면
                min_val_loss = val_loss                            # 최소 validation loss 값을 현재 값으로 갱신
                torch.save(sv_file, os.path.join(save_path, 'epoch-best-loss.pth'))  # best loss 모델로 저장
            log_info.append('best val loss={:.5f}'.format(min_val_loss))  # 현재까지의 best validation loss를 log에 추가

        # 현재 시각 출력 및 기입 (한 에폭 학습이 끝난 시점)
        t = timer.t()                                               # 학습 시작 후 현재까지의 누적 시간 계산
        # 전체 학습 중 진행률
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)  # 현재까지 완료한 epoch 비율 계산
        t_epoch = utils.time_text(t - t_epoch_start)                # 이번 에폭이 걸린 시간을 사람이 읽기 좋은 문자열로 변환
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)  # t_elapsed: 지금까지 총 학습 시간, t_all: 현재 속도로 추정한 전체 소요 시간
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))  # epoch 시간, 경과 시간, 예상 전체 시간을 log에 추가

        log(', '.join(log_info))                                    # log_info에 모은 학습/검증/시간 정보를 한 줄로 출력 및 저장
        # tensor 보드에 기록된 내용을 저장
        writer.flush()                                              # TensorBoard writer에 남아 있는 기록을 디스크에 반영

# 같은 코드라도 난수가 사용되는 부분이 있어서 seed를 고정하고 실행마다 결과가 달라지는 것을 방지
def set_seed(seed=42):
    random.seed(seed)                                               # 파이썬 random 모듈의 난수 seed 고정
    np.random.seed(seed)                                            # numpy 난수 seed 고정
    torch.manual_seed(seed)                                         # PyTorch CPU 연산의 난수 seed 고정
    torch.cuda.manual_seed_all(seed)                                # 모든 GPU의 PyTorch 난수 seed 고정
    torch.backends.cudnn.deterministic = True                       # cudnn 연산을 가능한 한 deterministic하게 수행하도록 설정
    torch.backends.cudnn.benchmark = False                          # 입력 크기에 따른 cudnn 자동 최적화 선택을 끄고 재현성을 우선시

if __name__ == '__main__':
    set_seed(42)                                                    # 실행마다 결과가 최대한 같아지도록 난수 seed 설정
    parser = argparse.ArgumentParser()                              # 터미널 명령어 인자를 읽기 위한 parser 생성
    parser.add_argument('--config')                                 # 사용할 config yaml 파일 경로를 입력받는 인자
    parser.add_argument('--name', default=None)                     # 저장 폴더 이름을 직접 지정할 때 사용하는 인자
    parser.add_argument('--tag', default=None)                      # 저장 폴더 이름 뒤에 추가 태그를 붙일 때 사용하는 인자
    parser.add_argument('--gpu', default='2')                       # 사용할 GPU 번호를 지정하는 인자; 기본값은 2번 GPU
    args = parser.parse_args()                                      # 실제 터미널 입력을 파싱하여 args 객체에 저장

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu                   # PyTorch가 볼 수 있는 GPU를 args.gpu로 제한

    # 입력받은 config 파일을 읽어서 로드
    with open(args.config, 'r') as f:                               # --config로 받은 yaml 파일을 읽기 모드로 열기
        config = yaml.load(f, Loader=yaml.FullLoader)               # yaml 내용을 Python dictionary로 변환
        print('config loaded.')                                     # config 로드 완료 메시지 출력

    # 저장폴더를 지정했다면 저장이름을 가져옴 아니면 config 파일 이름을 저장 파일로 사용함
    save_name = args.name                                           # --name 인자로 받은 저장 이름을 우선 사용
    if save_name is None:                                           # --name을 따로 주지 않았다면
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]  # config 파일명에서 .yaml을 뺀 값을 저장 이름으로 사용
        # tag가 입력된 경우 저장 이름 뒤에 추가
    if args.tag is not None:                                        # --tag 인자가 입력되었다면
        save_name += '_' + args.tag                                 # 저장 이름 뒤에 tag를 이어 붙임
    save_path = os.path.join(ROOT, 'save', save_name)                   # 최종 저장 경로를 ./save/save_name 형태로 생성

    # main 실행
    main(config, save_path)                                        