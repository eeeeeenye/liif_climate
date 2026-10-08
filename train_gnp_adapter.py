""" 얼린 LTE 위에 확률적 adapter 를 학습한다 (cnp-adapter / anp-adapter 공용).

    train_cnp_adapter.py 와 동일하지만 anp-adapter(models/anp_adapter.py) 도 레지스트리에
    올려두어, config.model.name 만 바꾸면 attention 버전을 학습할 수 있다.


    개요
    ----
      Y0(x)      = LTE(IMERG, x)                         (동결, no_grad)
      e_i        = y_i^ASOS - Y0(s_i)                    (ASOS 잔차, [0,1] minmax 공간)
      context C  = { [coord_i, Y0(s_i), e_i] }           (ASOS 일부)
      p(dy_t|C)  = N( mu_phi(x_t, C), sigma_phi(x_t, C)^2 )   (models/cnp_adapter.py)
      loss       = Gaussian NLL  (losses_cnp.py)

    매 iteration 마다 그 날짜의 유효 ASOS 지점을 context / target 으로 랜덤 분할한다
    (CNP 방식). context 만 adapter 에 주고, target 은 좌표만 주어 잔차분포를 예측하게 한 뒤
    실제 target 잔차와 NLL 로 비교한다. 다음 iteration 은 다시 다르게 분할.

    - 기존 파일은 하나도 수정하지 않는다. LTE 는 config.lte_from 체크포인트에서
      "있는 그대로" 만들어 즉시 freeze + eval.  correction_layer(add_layer) 는 쓰지 않는다.
    - 학습되는 파라미터는 cnp-adapter (수십만 개) 뿐.

    실행:
      python train_anp_adapter.py \
        --config configs/train_climate/gnp_adapter_lte_minmax.yaml --gpu 2
"""

import argparse
import os
import random

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import models.cnp_adapter          # noqa: F401
import models.anp_adapter          # noqa: F401
import models.gnp_adapter          # noqa: F401  (cnp- / anp- / gnp-adapter 레지스트리 등록)
import utils
from losses_cnp import make_loss


# ---------------------------------------------------------------------------
# pad_collate : ASOS 는 날짜마다 유효 지점 수가 달라 0-padding + mask 생성
#   (train_liif_finetune_weighted_mae.py 와 동일 로직)
# ---------------------------------------------------------------------------
def pad_collate(batch):
    lengths = [b['coord'].shape[0] for b in batch]
    max_len = max(lengths) if lengths else 0
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


def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        if torch.is_tensor(v):
            log('  {}: shape={}'.format(k, tuple(v.shape)))
    return DataLoader(
        dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8,
        pin_memory=True, collate_fn=pad_collate,
    )


def make_data_loaders():
    return (make_data_loader(config.get('train_dataset'), tag='train'),
            make_data_loader(config.get('val_dataset'), tag='val'))


# ---------------------------------------------------------------------------
def get_denorm(physical_norm):
    """ x_mm = x_norm * scale + shift 의 (scale, shift). NLL/RMSE 스케일 변환용. """
    t = physical_norm['type']
    if t == 'minmax':
        return (float(physical_norm['max']) - float(physical_norm['min'])), \
               float(physical_norm['min'])
    if t == 'zscore':
        return float(physical_norm['std']), float(physical_norm['mean'])
    raise ValueError(f'Unknown physical_norm type: {t}')


def load_frozen_lte(ckpt_path):
    log(f'[lte] load & freeze: {ckpt_path}')
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    spec = ck['model']
    if 'layer_spec' in spec.get('args', {}):
        raise RuntimeError('lte_from 체크포인트에 correction layer(layer_spec) 가 있습니다. '
                           'add_layer 없는 순수 LTE 체크포인트를 쓰세요.')
    lte = models.make({'name': spec['name'], 'args': spec['args']}, load_sd=False)
    lte.load_state_dict(spec['sd'], strict=True)
    lte = lte.cuda().eval()
    for p in lte.parameters():
        p.requires_grad = False
    log(f"[lte] epoch={ck.get('epoch', -1)}, params={sum(p.numel() for p in lte.parameters())}")
    return lte


@torch.no_grad()
def lte_query(lte, inp, level, coord, cell, chunk=20000):
    """ 얼린 LTE 로 coord 위치의 Y0 (minmax [0,1] 공간) 예측. -> (B, N, 1). """
    lte.gen_feat(inp)
    lte.topo = level
    n = coord.shape[1]
    outs = []
    for ql in range(0, n, chunk):
        qr = min(ql + chunk, n)
        outs.append(lte.query_rgb(coord[:, ql:qr], cell[:, ql:qr]))
    return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
def split_context_target(n, ctx_ratio, min_ctx, min_tgt, generator):
    """ n 개 유효 지점을 context/target index 로 분할 (disjoint). """
    perm = torch.randperm(n, generator=generator)
    n_ctx = int(round(ctx_ratio * n))
    n_ctx = max(min_ctx, min(n_ctx, n - min_tgt))
    return perm[:n_ctx], perm[n_ctx:]


def build_ct(coord_b, y0_b, e_b, ci, ti):
    """ context (1, Nc, 4) / target (1, Nt, 3) / target 잔차 (1, Nt) 구성. """
    ctx = torch.cat([coord_b[ci], y0_b[ci], e_b[ci]], dim=-1).unsqueeze(0)
    tgt = torch.cat([coord_b[ti], y0_b[ti]], dim=-1).unsqueeze(0)
    tgt_true = e_b[ti].squeeze(-1).unsqueeze(0)
    return ctx, tgt, tgt_true


# ---------------------------------------------------------------------------
def train_epoch(loader, lte, adapter, optimizer, loss_fn, ct_cfg, scale, loss_space):
    adapter.train()
    avg = utils.Averager()
    g = torch.Generator()   # iteration 마다 새 분할

    for batch in tqdm(loader, leave=False, desc='train'):
        if batch is None:
            continue
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)

        y0 = lte_query(lte, batch['inp'], batch['level'],
                       batch['coord'], batch['cell'])          # (B, N, 1) [0,1]
        e = batch['gt'] - y0                                    # (B, N, 1)
        mask = batch['mask'].bool()

        losses = []
        for b in range(y0.shape[0]):
            m = mask[b]
            n = int(m.sum().item())
            if n < ct_cfg['min_context'] + ct_cfg['min_target']:
                continue
            coord_b, y0_b, e_b = batch['coord'][b][m], y0[b][m], e[b][m]
            ci, ti = split_context_target(
                n, ct_cfg['ctx_ratio'], ct_cfg['min_context'],
                ct_cfg['min_target'], g)
            ctx, tgt, tgt_true = build_ct(coord_b, y0_b, e_b, ci, ti)
            mu, sigma = adapter(ctx, tgt)                       # (1, Nt)
            if loss_space == 'physical':
                mu, sigma, tgt_true = mu * scale, sigma * scale, tgt_true * scale
            losses.append(loss_fn(mu, sigma, tgt_true))

        if not losses:
            continue
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg.add(loss.item())

    return avg.item()


@torch.no_grad()
def validate(loader, lte, adapter, loss_fn, ct_cfg, scale, loss_space):
    """ 고정 seed 로 분할 -> epoch 간 비교 가능. NLL + 보정 RMSE(mm) + 커버리지. """
    adapter.eval()
    nll_avg = utils.Averager()
    sse_adapter = sse_base = n_pts = 0.0
    cover1 = utils.Averager()

    for batch in tqdm(loader, leave=False, desc='val'):
        if batch is None:
            continue
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)

        y0 = lte_query(lte, batch['inp'], batch['level'],
                       batch['coord'], batch['cell'])
        e = batch['gt'] - y0
        mask = batch['mask'].bool()

        for b in range(y0.shape[0]):
            m = mask[b]
            n = int(m.sum().item())
            if n < ct_cfg['min_context'] + ct_cfg['min_target']:
                continue
            coord_b, y0_b, e_b = batch['coord'][b][m], y0[b][m], e[b][m]
            g = torch.Generator().manual_seed(ct_cfg['val_split_seed'] + b)
            ci, ti = split_context_target(
                n, ct_cfg['ctx_ratio'], ct_cfg['min_context'],
                ct_cfg['min_target'], g)
            ctx, tgt, tgt_true = build_ct(coord_b, y0_b, e_b, ci, ti)
            mu, sigma = adapter(ctx, tgt)

            mu_l, sig_l, true_l = mu, sigma, tgt_true
            if loss_space == 'physical':
                mu_l, sig_l, true_l = mu * scale, sigma * scale, tgt_true * scale
            nll_avg.add(loss_fn(mu_l, sig_l, true_l).item())

            # 보정 효과: (mm 공간) adapter 잔차예측 vs 무보정(0)
            true_mm = (tgt_true * scale).flatten()
            pred_mm = (mu * scale).flatten()
            sig_mm = (sigma * scale).flatten()
            sse_adapter += torch.sum((true_mm - pred_mm) ** 2).item()
            sse_base += torch.sum(true_mm ** 2).item()
            n_pts += true_mm.numel()
            cover1.add(torch.mean(((true_mm - pred_mm).abs() <= sig_mm).float()).item())

    rmse_adapter = (sse_adapter / max(n_pts, 1)) ** 0.5
    rmse_base = (sse_base / max(n_pts, 1)) ** 0.5
    return {
        'nll': nll_avg.item(),
        'rmse_adapter_mm': rmse_adapter,
        'rmse_base_mm': rmse_base,
        'rmse_ratio': rmse_adapter / max(rmse_base, 1e-8),
        'coverage_1sigma': cover1.item(),
    }


# ---------------------------------------------------------------------------
def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()

    lte = load_frozen_lte(config['lte_from'])

    adapter_spec = config['model']
    adapter = models.make(adapter_spec).cuda()
    n_ad = sum(p.numel() for p in adapter.parameters())
    log(f'[adapter] {adapter_spec["name"]}  params={n_ad}')
    log(str(adapter))

    optimizer = utils.make_optimizer(adapter.parameters(), config['optimizer'])
    if config.get('multi_step_lr') is None:
        lr_scheduler = None
    else:
        lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    loss_space = config.get('loss_space', 'normalized')
    assert loss_space in ('normalized', 'physical')
    scale, _shift = get_denorm(config['physical_norm'])
    loss_fn = make_loss(config['loss'])
    log(f'[loss] {loss_fn}  (loss_space={loss_space}, mm_scale={scale:.3f})')

    ct_cfg = {
        'ctx_ratio': 0.6, 'min_context': 3, 'min_target': 2, 'val_split_seed': 1234,
    }
    ct_cfg.update(config.get('context_target', {}) or {})
    log(f'[context/target] {ct_cfg}')

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    min_val_nll = 1e18
    min_val_rmse = 1e18
    timer = utils.Timer()

    for epoch in range(1, epoch_max + 1):
        t0 = timer.t()
        log_info = [f'epoch {epoch}/{epoch_max}']
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train_epoch(train_loader, lte, adapter, optimizer,
                                 loss_fn, ct_cfg, scale, loss_space)
        if lr_scheduler is not None:
            lr_scheduler.step()
        log_info.append(f'train: nll={train_loss:.6f}')
        writer.add_scalars('nll', {'train': train_loss}, epoch)

        adapter_spec_out = {k: v for k, v in adapter_spec.items() if k != 'sd'}
        adapter_spec_out['sd'] = adapter.state_dict()
        optimizer_spec = dict(config['optimizer'])
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {'model': adapter_spec_out, 'optimizer': optimizer_spec,
                   'epoch': epoch, 'lte_from': config['lte_from']}
        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
        if epoch_save and epoch % epoch_save == 0:
            torch.save(sv_file, os.path.join(save_path, f'epoch-{epoch}.pth'))

        if epoch_val and epoch % epoch_val == 0:
            res = validate(val_loader, lte, adapter, loss_fn, ct_cfg, scale, loss_space)
            log_info.append(
                'val: nll={nll:.6f} rmse_adapter={rmse_adapter_mm:.3f}mm '
                'rmse_base={rmse_base_mm:.3f}mm ratio={rmse_ratio:.3f} '
                'cov1sig={coverage_1sigma:.3f}'.format(**res))
            writer.add_scalars('nll', {'val': res['nll']}, epoch)
            writer.add_scalars('rmse_mm', {'adapter': res['rmse_adapter_mm'],
                                           'base': res['rmse_base_mm']}, epoch)
            writer.add_scalar('coverage_1sigma', res['coverage_1sigma'], epoch)
            if res['nll'] < min_val_nll:
                min_val_nll = res['nll']
                torch.save(sv_file, os.path.join(save_path, 'epoch-best-nll.pth'))
            if res['rmse_adapter_mm'] < min_val_rmse:
                min_val_rmse = res['rmse_adapter_mm']
                torch.save(sv_file, os.path.join(save_path, 'epoch-best-rmse.pth'))
            log_info.append(f'best nll={min_val_nll:.6f}')

        t = timer.t()
        prog = epoch / epoch_max
        log_info.append('{} {}/{}'.format(utils.time_text(t - t0),
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
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
