""" LIIF 본체 직접 파인튜닝 (ASOS 관측만, correction layer / adapter 없음).

    사전학습된 LIIF(encoder+imnet) 체크포인트에서 시작해 ASOS 관측소 값만을 GT 로 삼아
    모델 자체의 가중치를 갱신한다. adapter 를 붙여 encoder/imnet 을 얼리던
    train_liif_finetune.py 와 반대 방향.

    - correction layer 금지 (체크포인트 spec 에 layer_spec 이 있으면 에러)
    - topo 없는 모델(HSR 등)도 동작: pad_collate 가 level=None 을 그대로 통과
    - loss: mse | l1  ([0,1] 정규화 공간, mask=1 인 실제 관측소만)

    사용 예:
      python train_liif_finetune_asos.py \
        --config configs/train_climate/climate_edsr_radar_hsr_2x_minmax_liif_ft_asos_lag1h.yaml \
        --gpu 2

    config 키:
        finetune_from : *.pth   (필수) 출발 체크포인트. 모델 구조는 체크포인트 spec 을 그대로 사용.
        trainable     : 'imnet' | 'all'   (기본 'imnet')
                        imnet -> encoder(EDSR) 동결, imnet MLP 만 학습 (ASOS 소수 지점 과적합 방지)
                        all   -> encoder 까지 전부 학습 (lr 을 더 작게 잡을 것)
        loss          : 'mse' | 'l1'   (기본 mse)
        physical_norm, optimizer, epoch_max, epoch_val, epoch_save, multi_step_lr : 기존과 동일.
        reset_epoch   : true(기본) 면 epoch 1 부터 번호 시작.
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

import datasets
import models
import utils
from test import eval_psnr
from train_liif_finetune import pad_collate   # obs_state 용 padding+mask (level=None 대응 포함)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        if torch.is_tensor(v):
            log('  {}: shape={}'.format(k, tuple(v.shape)))
        else:
            log('  {}: {}'.format(k, v))
    return DataLoader(dataset, batch_size=spec['batch_size'],
                      shuffle=(tag == 'train'), num_workers=8,
                      pin_memory=True, collate_fn=pad_collate)


def prepare_training():
    global base_model_spec
    ckpt_path = config['finetune_from']
    log(f'finetune_from: {ckpt_path}')
    sv_file = torch.load(ckpt_path, map_location='cpu')

    model_spec = sv_file['model']
    base_model_spec = {k: v for k, v in model_spec.items() if k != 'sd'}
    if model_spec.get('args', {}).get('layer_spec') is not None:
        raise RuntimeError('체크포인트에 layer_spec(correction layer) 이 있습니다. '
                           'adapter 없는 사전학습 체크포인트를 쓰세요.')

    model = models.make(model_spec, load_sd=True).cuda()
    if hasattr(model, 'add_layer'):
        raise RuntimeError('model 에 add_layer 가 부착되어 있습니다.')

    trainable_mode = config.get('trainable', 'imnet')
    assert trainable_mode in ('imnet', 'all'), trainable_mode
    if trainable_mode == 'imnet':
        for p in model.encoder.parameters():
            p.requires_grad = False
    log(f'[trainable] {trainable_mode}')

    for name, p in model.named_parameters():
        log(f' [{"TRAIN" if p.requires_grad else "FROZEN"}] {name}')
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f'trainable param : {n_train} / total : {n_total}')

    optimizer = utils.make_optimizer(
        [p for p in model.parameters() if p.requires_grad], config['optimizer'])

    epoch_start = 1 if config.get('reset_epoch', True) else sv_file['epoch'] + 1
    lr_scheduler = None
    if config.get('multi_step_lr') is not None:
        lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    log(f'epoch_start : {epoch_start}')
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer, loss_type):
    model.train()
    if config.get('trainable', 'imnet') == 'imnet':
        model.encoder.eval()    # EDSR 에 norm/dropout 은 없지만 동결 의도를 명시
    loss_fn = nn.L1Loss(reduction='none') if loss_type == 'l1' else nn.MSELoss(reduction='none')
    train_loss = utils.Averager()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        if batch is None:
            continue
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)

        pred = model(batch['inp'], batch['level'], batch['coord'], batch['cell'])
        mask = batch['mask'].unsqueeze(-1)
        loss = (loss_fn(pred, batch['gt']) * mask).sum() / mask.sum()
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = None
        loss = None
    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    loss_type = config.get('loss', 'mse')
    assert loss_type in ('mse', 'l1'), loss_type
    log(f'[loss] {loss_type} (normalized space, mask=1 관측소만)')

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18
    min_val_loss = 1e18
    timer = utils.Timer()

    # 파인튜닝 전 baseline 을 epoch 0 으로 기록 -> 이후 epoch 의 개선 여부를 바로 비교
    if epoch_val is not None:
        v_psnr, v_loss = eval_psnr(val_loader, model, physical_norm=config['physical_norm'],
                                   eval_type=config.get('eval_type'),
                                   eval_bsize=config.get('eval_bsize'), loss_type=loss_type)
        log('epoch 0 (pretrained, 파인튜닝 전): val psnr={:.4f}, val loss={:.6e}'.format(v_psnr, v_loss))

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, loss_type)
        if lr_scheduler is not None:
            lr_scheduler.step()
        # 정규화 공간 손실은 1e-4 수준이라 {:.4f} 로는 0.0000 으로 보임 -> 지수 표기
        log_info.append('train: loss={:.4e}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        model_ = model.module if n_gpus > 1 else model
        model_spec = dict(base_model_spec)
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = dict(config['optimizer'])
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {'model': model_spec, 'optimizer': optimizer_spec, 'epoch': epoch}

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            model_v = model.module if (n_gpus > 1 and config.get('eval_bsize') is not None) else model
            val_res, val_loss = eval_psnr(
                val_loader, model_v, physical_norm=config['physical_norm'],
                eval_type=config.get('eval_type'), eval_bsize=config.get('eval_bsize'),
                loss_type=loss_type)
            log_info.append('val: psnr={:.4f}'.format(val_res))
            log_info.append('val loss={:.4e}'.format(val_loss))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            writer.add_scalars('loss', {'val': val_loss}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best-psnr.pth'))
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(sv_file, os.path.join(save_path, 'epoch-best-loss.pth'))
            log_info.append('best val loss={:.4e}'.format(min_val_loss))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        log_info.append('{} {}/{}'.format(utils.time_text(t - t_epoch_start),
                                          utils.time_text(t), utils.time_text(t / prog)))
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
    main(config, os.path.join('./save', save_name))
