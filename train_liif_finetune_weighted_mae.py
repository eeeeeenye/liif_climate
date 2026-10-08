""" LIIF/LTE finetune with the Weighted-MAE loss of Wang, Tian & Carroll (2023).

    참고문헌
    --------
    Wang, F., Tian, D., & Carroll, M. (2023).
    Customized deep learning for precipitation bias correction and downscaling.
    Geoscientific Model Development, 16, 535-556.
    https://doi.org/10.5194/gmd-16-535-2023
    -> Section 3.1.3 (Customized loss functions), Eq. (1) : Weighted MAE

    이 스크립트는 train_liif_finetune.py 를 베이스로 하되,
      1) correction layer(adapter / add_layer) 를 전혀 쓰지 않고
      2) 손실을 nn.MSELoss / nn.L1Loss 대신 losses_wang2023.WeightedMAELoss 로 바꾸어
    사전학습된 LTE 체크포인트의 "마지막 에폭"에서 이어서 파인튜닝한다.

    wrapper.obs_state 는 true/false 모두 지원한다.
      - obs_state: false -> HR 그리드에서 sample_q 개 샘플링한 값이 GT
      - obs_state: true  -> ASOS 관측소 값이 GT (날짜마다 지점 수가 달라 pad_collate 로
                            0-padding + mask 생성; Weighted MAE 는 mask=1 인 실제 지점에만 적용)
        이때 dataset.args 에 obs_path_3(관측 CSV), meta_path(좌표/값 범위 JSON) 가 필요하다.
        ObsInMemory 는 gt 를 meta 의 min_val/max_val 로 minmax 정규화하는데, 현재 meta
        (train_imerg80x100_meta.json)는 0 ~ 506.02 로 physical_norm 과 동일하므로
        loss_space: physical 의 denorm(×506.02)을 그대로 써도 된다.

    사용 예:
      python train_liif_finetune_weighted_mae.py \
        --config configs/train_climate/climate_edsr_topo_80x100_topo_minmax_lte_ft_weighted_mae.yaml \
        --gpu 2

    config 필수/선택 키:
        finetune_from : *.pth        (필수) 이어서 학습할 체크포인트. 보통 epoch-last.pth
        physical_norm : {type,min,max} 또는 {type,mean,std}  (필수)
                        정규화([0,1] 등) 공간의 pred/gt 를 mm/day 로 되돌리는 데 사용.
        loss:
          name: weighted_mae
          args: {w_min, w_max, normalize_by_weight}
        loss_space : 'physical' | 'normalized'   (기본 'physical')
                     physical  -> pred/gt 를 mm/day 로 denorm 후 손실 계산 (논문과 동일 단위).
                                  이때 loss.args.w_min/w_max 도 mm/day 로 준다.
                     normalized -> [0,1] 정규화 공간 그대로 손실 계산.
                                  이때 w_min/w_max 도 [0,1] 스케일로 준다.
        freeze_encoder : bool  (기본 False)  True 면 encoder 를 얼리고 나머지만 학습.
                         (ASOS 지점만으로 encoder 까지 풀면 과적합 위험 -> true 권장 검토)
        reset_optimizer : bool (기본 True)   False 면 체크포인트의 optimizer 상태를 복원.
        reset_epoch : bool     (기본 True)   False 면 체크포인트 epoch+1 부터 번호 이어감.
        optimizer / multi_step_lr / epoch_max / epoch_val / epoch_save : 기존과 동일.
"""

import argparse
import os
import random

import numpy as np
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import torch.nn.functional as F

import datasets
import models
import utils
from test import eval_psnr
from losses_wang2023 import make_loss


# ---------------------------------------------------------------------------
# pad_collate : obs_state=true(ASOS) 대응
#   ASOS 는 날짜마다 유효 관측소 개수가 달라 coord/gt/cell 길이가 샘플마다 다르다.
#   배치 내 최대 길이에 맞춰 0-padding 하고, 실제 지점을 1 로 표시한 mask 를 만든다.
#   (obs_state=false 로 길이가 고정이어도 그대로 동작: pad_len=0, mask 전부 1.)
#   train_liif_finetune.py 의 pad_collate 와 동일 로직.
# ---------------------------------------------------------------------------
def pad_collate(batch):
    lengths = [b['coord'].shape[0] for b in batch]
    max_len = max(lengths)
    if max_len == 0:
        return None

    padded_inp, padded_gt, padded_coord = [], [], []
    padded_cell, padded_level, masks = [], [], []

    for b, q in zip(batch, lengths):
        pad_len = max_len - q
        padded_inp.append(b['inp'])
        padded_gt.append(F.pad(b['gt'], (0, 0, 0, pad_len)))
        padded_coord.append(F.pad(b['coord'], (0, 0, 0, pad_len)))
        padded_cell.append(F.pad(b['cell'], (0, 0, 0, pad_len)))
        padded_level.append(b['level'])
        m = torch.zeros(max_len, dtype=torch.float32)
        m[:q] = 1.0
        masks.append(m)

    return {
        'inp': torch.stack(padded_inp, dim=0),
        'gt': torch.stack(padded_gt, dim=0),
        'coord': torch.stack(padded_coord, dim=0),
        'cell': torch.stack(padded_cell, dim=0),
        'level': torch.stack(padded_level, dim=0),
        'mask': torch.stack(masks, dim=0),
    }


# ---------------------------------------------------------------------------
# 데이터 로더
#   obs_state=true  -> 길이가 샘플마다 다르므로 pad_collate 필요 (mask 생성)
#   obs_state=false -> 길이 고정이지만 pad_collate 를 써도 무해 (mask 전부 1)
#   -> 항상 pad_collate 사용.
# ---------------------------------------------------------------------------
def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=(tag == 'train'),
        num_workers=8,
        pin_memory=True,
        collate_fn=pad_collate,
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 정규화 공간 <-> mm/day 변환 계수
# ---------------------------------------------------------------------------
def get_denorm(physical_norm):
    """ x_mm = x_norm * scale + shift  형태의 (scale, shift) 를 반환. """
    t = physical_norm['type']
    if t == 'minmax':
        pmin = float(physical_norm['min'])
        pmax = float(physical_norm['max'])
        return (pmax - pmin), pmin
    elif t == 'zscore':
        return float(physical_norm['std']), float(physical_norm['mean'])
    raise ValueError(f'Unknown physical_norm type: {t}')


# ---------------------------------------------------------------------------
# 학습 준비
#   - finetune_from 체크포인트에서 "그대로" 모델을 만들고 가중치를 로드
#   - correction layer 는 절대 부착하지 않음 (있으면 즉시 에러)
# ---------------------------------------------------------------------------
def prepare_training():
    global base_model_spec
    ckpt_path = config['finetune_from']
    log(f'finetune_from: {ckpt_path}')
    sv_file = torch.load(ckpt_path, map_location='cpu')

    # 체크포인트에 저장된 model spec 을 그대로 사용해 구조/가중치 불일치를 원천 차단
    model_spec = sv_file['model']
    base_model_spec = {k: v for k, v in model_spec.items() if k != 'sd'}

    # ---- correction layer 금지 가드 --------------------------------------
    if 'layer_spec' in model_spec.get('args', {}):
        raise RuntimeError(
            '이 스크립트는 correction layer 를 사용하지 않습니다. '
            'finetune_from 체크포인트의 model.args 에 layer_spec 이 있습니다.')

    model = models.make(model_spec, load_sd=True).cuda()

    if hasattr(model, 'add_layer'):
        raise RuntimeError(
            'model 에 add_layer(correction layer) 가 부착되어 있습니다. '
            'correction layer 없는 체크포인트를 사용하세요.')
    log('[check] correction layer 없음 확인 완료 (add_layer 미부착)')

    # ---- encoder freeze (선택) -----------------------------------------
    freeze_encoder = bool(config.get('freeze_encoder', False))
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        log('[freeze] encoder 파라미터 동결')

    # ---- optimizer ---------------------------------------------------------
    reset_optimizer = bool(config.get('reset_optimizer', True))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if reset_optimizer:
        optimizer = utils.make_optimizer(trainable, config['optimizer'])
        log('[optimizer] 새로 생성 (config.optimizer 사용)')
    else:
        opt_spec = sv_file['optimizer']
        optimizer = utils.make_optimizer(trainable, opt_spec, load_sd=True)
        log('[optimizer] 체크포인트 상태 복원')

    # ---- epoch 번호 ------------------------------------------------------
    reset_epoch = bool(config.get('reset_epoch', True))
    epoch_start = 1 if reset_epoch else (sv_file['epoch'] + 1)

    # ---- lr scheduler --------------------------------------------------
    if config.get('multi_step_lr') is None:
        lr_scheduler = None
    else:
        lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()

    # ---- trainable 파라미터 로그 -------------------------------------------
    log('=' * 50)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f'trainable param : {n_train} / total : {n_total} '
        f'(frozen : {n_total - n_train})')
    log(f'epoch_start : {epoch_start}')
    log('=' * 50)

    return model, optimizer, epoch_start, lr_scheduler


# ---------------------------------------------------------------------------
# 한 epoch 학습 : Weighted MAE (Wang et al. 2023, Eq. 1)
# ---------------------------------------------------------------------------
def train(train_loader, model, optimizer, loss_fn, denorm, loss_space):
    model.train()
    train_loss = utils.Averager()
    scale, shift = denorm

    for batch in tqdm(train_loader, leave=False, desc='train'):
        if batch is None:            # 배치 내 유효 관측소가 하나도 없는 경우
            continue
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)

        inp = batch['inp']
        pred = model(inp, batch.get('level'), batch['coord'], batch['cell'])
        gt = batch['gt']

        # padding 된 가짜 지점 제외 (obs_state=true 대응; false 면 전부 유효).
        # boolean 인덱싱으로 유효 원소만 뽑으면 WeightedMAE(elementwise -> mean) 가
        # 정확히 (1/n_valid) * sum(w * |e|) 이 된다.
        mask = batch.get('mask')
        if mask is not None:
            valid = mask.bool()
            pred = pred[valid]
            gt = gt[valid]

        if loss_space == 'physical':
            # [0,1] (또는 zscore) -> mm/day 로 되돌린 뒤 논문과 동일 단위로 손실 계산
            pred_l = pred * scale + shift
            gt_l = gt * scale + shift
        else:  # 'normalized'
            pred_l = pred
            gt_l = gt

        if pred_l.numel() == 0:
            continue

        loss = loss_fn(pred_l, gt_l)
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None
        loss = None

    return train_loss.item()


# ---------------------------------------------------------------------------
def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    # ---- 손실 함수 (Wang et al. 2023, Eq. 1) -----------------------------
    loss_space = config.get('loss_space', 'physical')
    assert loss_space in ('physical', 'normalized')
    denorm = get_denorm(config['physical_norm'])
    loss_fn = make_loss(config['loss'])
    log(f'[loss] {loss_fn}  (loss_space={loss_space})')
    if loss_space == 'physical':
        log(f'[loss] denorm  x_mm = x*{denorm[0]:.4f} + {denorm[1]:.4f}  '
            f'-> w_min/w_max 는 mm/day 단위로 해석됨')

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18
    min_val_loss = 1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer,
                           loss_fn, denorm, loss_space)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: wmae={:.6f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        model_ = model.module if n_gpus > 1 else model
        # 저장 시엔 원본 체크포인트의 model spec(구조) 을 그대로 유지하고 sd 만 갱신
        model_spec = dict(base_model_spec)
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch,
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            model_v = model.module if (n_gpus > 1 and config.get('eval_bsize')
                                       is not None) else model
            val_res, val_loss = eval_psnr(
                val_loader, model_v,
                physical_norm=config['physical_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'),
                loss_type='l1')

            log_info.append('val: psnr={:.4f}'.format(val_res))
            log_info.append('val loss(L1, norm)={:.6f}'.format(val_loss))
            writer.add_scalars('psnr', {'val': val_res}, epoch)

            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file,
                           os.path.join(save_path, 'epoch-best-psnr.pth'))
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(sv_file,
                           os.path.join(save_path, 'epoch-best-loss.pth'))
            log_info.append('best val loss={:.6f}'.format(min_val_loss))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='2')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
