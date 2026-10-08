""" LTE imnet LoRA 파인튜닝 (Truong et al., "Low-Rank Adaptation of Neural Fields") — ASOS station loss.

    - 사전학습 LTE(encoder/coef/freq/phase/imnet/skip) 전부 freeze
    - imnet 의 모든 nn.Linear 에 LoRA 삽입:  W h + (1/r) B A h + b   (A~N(0,1/d_in), B=0)
    - 학습 대상은 A, B 뿐. ASOS 관측소 좌표에서만 query -> station loss (mask=1 인 실제 관측소)
    - LTE forward(local ensemble + bilinear skip)는 그대로. imnet 만 LoRA 가 붙은 것으로 바뀜.
    - 학습 시작 시 B=0 이므로 LoRA-LTE == 원본 LTE (시작 시 자동 검증 + epoch 0 val 로 기록)

    사용 예:
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_lte_lora_asos.py \
        --config configs/train_climate/climate_edsr_radar_hsr_2x_minmax_lte_lora_r4_asos_lag1h.yaml --gpu 2

    config 키:
        finetune_from : 사전학습 LTE *.pth (필수)
        lora.rank     : r (기본 4)
        lora.layers   : 'all'(논문 방식, 기본) | 'no_first'(첫 층 제외) | 'last'(출력층만, ablation)
        loss          : 'rel_l2'(논문 §5.1, 기본) | 'l1'(LTE 사전학습 손실) | 'huber' | 'mse'
                        | {name: balanced_huber, args: {...}}
                        ([0,1] 정규화 공간, 실제 관측소만)
        rel_eps       : rel_l2 분모 eps (정규화 공간). 0.01 = Instant-NGP 값 (~0.75 mm/h 규모 이상에서 분모가 pred^2 로 지배)
        huber_delta   : huber 의 delta (정규화 공간; 0.01 ~= 0.75 mm/h)
        val_seed      : val 표본(랜덤 crop anchor)을 고정하는 seed. rank 간 val 비교를 공정하게 하기 위함
        lora_plus     : {enabled: true, b_lr_ratio: 4}; B의 LR을 A보다 크게 두는 LoRA+ 옵션.
        functional_distill : {weight: 0.05, sample_q: 128, loss: mse, cell_mode: both}; sparse ASOS 이외 격자에서
                        LoRA 출력이 base LTE의 공간장을 보존하도록 하는 function-space regularization.
        physical_norm, optimizer, epoch_max, epoch_val, epoch_save, multi_step_lr : 기존과 동일

    저장물 (.pth): {'lora': A,B state_dict, 'rank', 'layers', 'base_ckpt', 'epoch', 'optimizer'}  (수십 KB)
      -> 복원: models.lora.build_lora_lte(base_ckpt, rank, layers, lora_sd)
      -> 일반 LTE 체크포인트로 합치기: merge_lora_lte.py
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import utils
from models.lora import (build_lora_lte, lora_modules, lora_state_dict, set_lora_enabled)
from models.global_residual import GlobalResidualAdapter
from train_liif_finetune import pad_collate   # obs_state 용 padding+mask (level=None 대응)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    log('{} dataset: size={}'.format(tag, len(dataset)))
    return DataLoader(dataset, batch_size=spec['batch_size'],
                      shuffle=(tag == 'train'), num_workers=8,
                      pin_memory=True, collate_fn=pad_collate)


REL_EPS = 0.01   # main() 에서 config['rel_eps'] 로 덮어씀


def to_cuda(batch):
    return {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_global_residual(spec):
    """Build a bounded absolute-location residual adapter from config metadata."""
    spec = spec or {}
    if not spec.get('enabled', False):
        return None
    csv_path, meta_path = spec.get('station_csv'), spec.get('station_meta')
    if not csv_path or not meta_path:
        raise ValueError('global_residual requires station_csv and station_meta')
    with open(meta_path) as f:
        meta = json.load(f)
    lat_max, lat_min = meta['lat_range']
    lon_min, lon_max = meta['lon_range']
    seen = set()
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            seen.add((float(row['y']), float(row['x'])))  # (latitude, longitude)
    # Wrapper flips its y coordinate, so this is the same south->north global
    # [-1,1] convention returned as `global_coord`.
    station_coords = torch.tensor([
        [2.0 * (lat - lat_min) / (lat_max - lat_min) - 1.0,
         2.0 * (lon - lon_min) / (lon_max - lon_min) - 1.0]
        for lat, lon in sorted(seen)
    ], dtype=torch.float32)
    return GlobalResidualAdapter(
        fourier_bands=spec.get('fourier_bands', [1, 2, 4, 8]),
        hidden_dim=int(spec.get('hidden_dim', 32)),
        max_delta_norm=float(spec.get('max_delta_norm', 0.25)),
        station_coords=station_coords,
        gate_radius_norm=float(spec.get('gate_radius_norm', 0.15)),
    ).cuda()


def _loss_name_and_args(loss_spec):
    """Accept legacy string loss configs and the richer mapping form."""
    if isinstance(loss_spec, str):
        return loss_spec, {}
    if not isinstance(loss_spec, dict) or 'name' not in loss_spec:
        raise ValueError("loss must be a name string or {name: ..., args: {...}} mapping")
    return loss_spec['name'], loss_spec.get('args', {}) or {}


def _rain_intensity_weight(gt, loss_args, pmin, pmax):
    """Return target-conditioned precipitation weights in physical mm/hr.

    The default thresholds are Shi et al. (NeurIPS 2017).  We normalize by
    the effective weight sum in ``station_loss`` so a batch containing a rare
    heavy-rain gauge does not change the overall loss scale merely because its
    composition differs from another batch.
    """
    bins = loss_args.get('bins_mm_h', [2.0, 5.0, 10.0, 30.0])
    weights = loss_args.get('weights', [1.0, 2.0, 5.0, 10.0, 30.0])
    if len(weights) != len(bins) + 1:
        raise ValueError('balanced_huber weights must have len(bins_mm_h) + 1 entries')
    if any(b <= 0 for b in bins) or any(b1 >= b2 for b1, b2 in zip(bins[:-1], bins[1:])):
        raise ValueError('bins_mm_h must be strictly increasing positive values')
    if any(w <= 0 for w in weights):
        raise ValueError('balanced_huber weights must be positive')

    target_mm_h = gt.detach() * (pmax - pmin) + pmin
    out = torch.full_like(target_mm_h, float(weights[0]))
    for threshold, weight in zip(bins, weights[1:]):
        out = torch.where(target_mm_h >= float(threshold), float(weight), out)
    return out


def station_loss(pred, gt, mask, loss_spec, delta, pmin, pmax):
    """Masked ASOS loss, including a tail-aware robust option.

    ``balanced_huber`` is a sparse-gauge adaptation of precipitation balanced
    losses: target intensity controls the weight while Huber keeps a single
    noisy extreme gauge from producing an unbounded gradient.
    """
    loss_type, loss_args = _loss_name_and_args(loss_spec)
    if loss_type == 'rel_l2':
        # 논문 §5.1 relative L2 (Müller et al. 2022 정의): (pred - gt)^2 / (sg(pred)^2 + eps). sg = stop-gradient
        l = (pred - gt) ** 2 / (pred.detach() ** 2 + REL_EPS)
    elif loss_type == 'l1':
        l = F.l1_loss(pred, gt, reduction='none')
    elif loss_type == 'mse':
        l = F.mse_loss(pred, gt, reduction='none')
    elif loss_type == 'huber':
        l = F.huber_loss(pred, gt, reduction='none', delta=delta)
    elif loss_type == 'balanced_huber':
        l = F.huber_loss(pred, gt, reduction='none', delta=delta)
        weight = _rain_intensity_weight(gt, loss_args, pmin, pmax)
        mask = mask.unsqueeze(-1)
        return (l * weight * mask).sum() / (weight * mask).sum().clamp_min(1.0)
    else:
        raise ValueError(f'Unknown loss: {loss_type}')
    mask = mask.unsqueeze(-1)
    return (l * mask).sum() / mask.sum().clamp_min(1.0)


def _query_with_cached_features(model, coord, cell):
    """Query LTE after ``gen_feat`` has been called once for an input crop."""
    return model.correction_layer(model.query_rgb(coord, cell))


def _random_grid_queries(batch, sample_q, cell_mode, grid_scale=2.0):
    """Uniform random queries with ASOS, raster-grid, or mixed cell sizes."""
    b, _, h, w = batch['inp'].shape
    grid_scale = float(grid_scale)
    if grid_scale <= 0:
        raise ValueError('functional_distill.grid_scale must be positive')
    coord = torch.rand(b, sample_q, 2, device=batch['inp'].device) * 2.0 - 1.0
    asos_cell = torch.full_like(coord, 2.0 / 1000.0)
    grid_cell = torch.ones_like(coord)
    grid_cell[..., 0] *= 2.0 / (h * grid_scale)
    grid_cell[..., 1] *= 2.0 / (w * grid_scale)
    if cell_mode == 'asos':
        cell = asos_cell
    elif cell_mode == 'grid':
        cell = grid_cell
    elif cell_mode == 'both':
        # LTE's phase depends on cell.  Preserve both the point-observation
        # function used by ASOS and the raster function used for maps.
        cell = grid_cell
        cell[:, :sample_q // 2] = asos_cell[:, :sample_q // 2]
    else:
        raise ValueError("functional_distill.cell_mode must be 'asos', 'grid', or 'both'")
    return coord, cell


def _local_to_global(local_coord, global_bbox):
    """Map crop-local LIIF coordinates to the wrapper's whole-domain frame.

    ``global_bbox`` is [y_min, y_max, x_min, x_max] for each crop.  Station
    coordinates already arrive in that frame as ``batch['global_coord']``;
    this mapping is only needed for random distillation queries.
    """
    if global_bbox.ndim != 2 or global_bbox.shape[-1] != 4:
        raise ValueError('global_bbox must have shape [B, 4]')
    y0, y1, x0, x1 = global_bbox.unbind(dim=-1)
    y = y0[:, None] + 0.5 * (local_coord[..., 0] + 1.0) * (y1 - y0)[:, None]
    x = x0[:, None] + 0.5 * (local_coord[..., 1] + 1.0) * (x1 - x0)[:, None]
    return torch.stack([y, x], dim=-1)


def functional_distill_loss(model, batch, distill_spec, global_residual=None):
    """Preserve base-LTE outputs at unlabelled coordinates (LwF-style).

    Both outputs share the already-computed encoder/LTE features.  Only the
    LoRA-enabled query receives gradients, so this adds no trainable model and
    avoids a second encoder forward pass.
    """
    if not distill_spec or float(distill_spec.get('weight', 0.0)) <= 0:
        return None
    sample_q = int(distill_spec.get('sample_q', 128))
    if sample_q <= 0:
        return None
    coord, cell = _random_grid_queries(batch, sample_q, distill_spec.get('cell_mode', 'both'),
                                       distill_spec.get('grid_scale', 2.0))
    with torch.no_grad():
        set_lora_enabled(model, False)
        try:
            base_pred = _query_with_cached_features(model, coord, cell)
        finally:
            set_lora_enabled(model, True)
    lora_pred = _query_with_cached_features(model, coord, cell)
    if global_residual is not None:
        if 'global_bbox' not in batch:
            raise KeyError('global_residual requires global_bbox from the wrapper')
        lora_pred = global_residual(lora_pred, _local_to_global(coord, batch['global_bbox']))
    kind = distill_spec.get('loss', 'mse')
    if kind == 'mse':
        return F.mse_loss(lora_pred, base_pred)
    if kind == 'rel_l2':
        eps = float(distill_spec.get('rel_eps', REL_EPS))
        return ((lora_pred - base_pred) ** 2 / (base_pred.detach() ** 2 + eps)).mean()
    raise ValueError("functional_distill.loss must be 'mse' or 'rel_l2'")


def forward(model, batch, distill_spec=None, global_residual=None):
    """Run station queries and optionally retain base behavior elsewhere."""
    # Use one encoder feature computation for station and random-grid queries.
    model.gen_feat(batch['inp'])
    model.topo = batch['level']
    pred = _query_with_cached_features(model, batch['coord'], batch['cell'])
    if global_residual is not None:
        if 'global_coord' not in batch:
            raise KeyError('global_residual requires return_global_coord: true in both wrappers')
        pred = global_residual(pred, batch['global_coord'])
    distill = functional_distill_loss(model, batch, distill_spec, global_residual)
    return pred, distill


@torch.no_grad()
def _intensity_bin_metrics(pred, target):
    """Diagnostics for the precipitation bins used by Shi et al. (2017)."""
    bounds = [0.0, 2.0, 5.0, 10.0, 30.0, np.inf]
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        idx = (target >= lo) & (target < hi)
        if not idx.any():
            out.append((lo, hi, 0, np.nan, np.nan))
            continue
        err = pred[idx] - target[idx]
        out.append((lo, hi, int(idx.sum()), float(np.abs(err).mean()), float(err.mean())))
    return out


@torch.no_grad()
def evaluate(loader, model, loss_spec, delta, pmin, pmax, seed, global_residual=None):
    """val 지표. loss 는 학습 손실과 동일(정규화 공간), 나머지는 mm/h 로 환산한 관측소 지표.
    seed 로 val 의 랜덤 crop anchor 를 고정 (전역 RNG 상태는 복원해서 train shuffle 에 영향 없음)."""
    model.eval()
    if global_residual is not None:
        global_residual.eval()
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)          # DataLoader worker seed(python random 포함)가 여기서 유도됨
    preds, gts = [], []
    for batch in loader:
        if batch is None:
            continue
        batch = to_cuda(batch)
        pred, _ = forward(model, batch, global_residual=global_residual)
        m = batch['mask'].bool()
        preds.append(pred[m].reshape(-1, 1).cpu())
        gts.append(batch['gt'][m].reshape(-1, 1).cpu())
    torch.set_rng_state(rng_state)

    pred_norm, gt_norm = torch.cat(preds), torch.cat(gts)
    n = pred_norm.shape[0]
    # Compute a single global weighted loss.  Averaging batch-level losses is
    # incorrect for balanced_huber because each batch has a different sum of
    # rainfall weights.
    loss = station_loss(pred_norm, gt_norm, torch.ones(n), loss_spec, delta, pmin, pmax).item()
    p = (pred_norm * (pmax - pmin) + pmin).clamp(min=0).numpy().reshape(-1)
    g = (gt_norm * (pmax - pmin) + pmin).clamp(min=0).numpy().reshape(-1)
    wet = g > 0.1
    corr = float(np.corrcoef(p, g)[0, 1]) if p.std() > 0 and g.std() > 0 else float('nan')
    return {
        'loss': loss, 'n': n,
        'mae': float(np.abs(p - g).mean()), 'bias': float((p - g).mean()),
        'rmse': float(np.sqrt(((p - g) ** 2).mean())), 'corr': corr,
        'wet_mae': float(np.abs(p - g)[wet].mean()), 'wet_bias': float((p - g)[wet].mean()),
        'wet_n': int(wet.sum()),
        'intensity_bins': _intensity_bin_metrics(p, g),
    }


def fmt_metrics(m):
    return ('loss={loss:.4e} | mm/h MAE={mae:.4f} bias={bias:+.4f} RMSE={rmse:.4f} r={corr:.3f} '
            '| wet(>0.1) MAE={wet_mae:.3f} bias={wet_bias:+.3f} (n={n}, wet {wet_n})').format(**m)


def fmt_intensity_bins(m):
    cells = []
    for lo, hi, n, mae, bias in m['intensity_bins']:
        name = f'{lo:g}-{hi:g}' if np.isfinite(hi) else f'>={lo:g}'
        cells.append(f'{name}: n={n}, MAE={mae:.3f}, b={bias:+.3f}' if n else f'{name}: n=0')
    return ' | '.join(cells)


def rel_delta_w(model):
    """층별 ||dW||_F / ||W||_F — LoRA 가 각 층을 얼마나 움직였는지."""
    out = []
    for m in lora_modules(model):
        with torch.no_grad():
            out.append((m.delta_weight().norm() / m.base.weight.norm()).item())
    return out


def train_one_epoch(loader, model, optimizer, loss_spec, delta, pmin, pmax, distill_spec,
                    global_residual=None):
    model.eval()     # dropout/BN 없음. 전 모듈 동결이므로 eval 고정 (학습되는 건 A, B 뿐)
    if global_residual is not None:
        global_residual.train()
    total_avg, station_avg, distill_avg = utils.Averager(), utils.Averager(), utils.Averager()
    for batch in tqdm(loader, leave=False, desc='train'):
        if batch is None:
            continue
        batch = to_cuda(batch)
        pred, distill = forward(model, batch, distill_spec, global_residual)
        station = station_loss(pred, batch['gt'], batch['mask'], loss_spec, delta, pmin, pmax)
        loss = station
        if distill is not None:
            loss = loss + float(distill_spec['weight']) * distill
        total_avg.add(loss.item())
        station_avg.add(station.item())
        if distill is not None:
            distill_avg.add(distill.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {'total': total_avg.item(), 'station': station_avg.item(), 'distill': distill_avg.item()}


def make_lora_optimizer(model, optimizer_spec, lora_plus_spec, global_residual=None,
                        global_residual_lr=None):
    """Build ordinary LoRA or LoRA+ parameter groups.

    LoRA+ assigns a larger learning rate to B than A.  With this repository's
    A-random/B-zero initialization, B must first move before A obtains a useful
    gradient, which is the setting addressed by LoRA+.
    """
    enabled = bool((lora_plus_spec or {}).get('enabled', False))
    residual_params = [] if global_residual is None else list(global_residual.parameters())
    if not enabled:
        params = [p for p in model.parameters() if p.requires_grad] + residual_params
        return utils.make_optimizer(params, optimizer_spec)

    ratio = float(lora_plus_spec.get('b_lr_ratio', 4.0))
    if ratio <= 1.0:
        raise ValueError('lora_plus.b_lr_ratio must be > 1')
    a_params, b_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith('.A'):
            a_params.append(parameter)
        elif name.endswith('.B'):
            b_params.append(parameter)
        else:
            raise AssertionError(f'Unexpected trainable parameter: {name}')
    if not a_params or not b_params:
        raise RuntimeError('LoRA+ requires trainable A and B parameters')
    b_lr = float(optimizer_spec['args']['lr'])
    groups = [
        {'params': a_params, 'lr': b_lr / ratio, 'name': 'lora_A'},
        {'params': b_params, 'lr': b_lr, 'name': 'lora_B'},
    ]
    if residual_params:
        groups.append({
            'params': residual_params,
            'lr': b_lr if global_residual_lr is None else float(global_residual_lr),
            'name': 'global_residual',
        })
    return utils.make_optimizer(groups, optimizer_spec)


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    lcfg = config.get('lora', {})
    rank, layers = lcfg.get('rank', 4), lcfg.get('layers', 'all')
    global REL_EPS
    loss_spec = config.get('loss', 'rel_l2')
    loss_type, loss_args = _loss_name_and_args(loss_spec)
    assert loss_type in ('rel_l2', 'l1', 'huber', 'mse', 'balanced_huber'), loss_type
    REL_EPS = config.get('rel_eps', 0.01)
    delta = config.get('huber_delta', 0.01)
    pmin, pmax = float(config['physical_norm']['min']), float(config['physical_norm']['max'])
    assert config['physical_norm']['type'] == 'minmax'
    val_seed = config.get('val_seed', 0)
    distill_spec = config.get('functional_distill', {}) or {}
    global_residual_spec = config.get('global_residual', {}) or {}

    train_loader = make_data_loader(config['train_dataset'], 'train')
    val_loader = make_data_loader(config['val_dataset'], 'val')

    base_ckpt = config['finetune_from']
    model, _ = build_lora_lte(base_ckpt, rank, layers)
    global_residual = make_global_residual(global_residual_spec)
    log(f'base ckpt: {base_ckpt}')
    extra = {'huber': f'(delta={delta})', 'balanced_huber': f'(delta={delta}, args={loss_args})',
             'rel_l2': f'(eps={REL_EPS})'}.get(loss_type, '')
    log(f'LoRA rank={rank}, layers={layers}, loss={loss_type}{extra}')
    if float(distill_spec.get('weight', 0.0)) > 0:
        log('functional_distill: '
            f"weight={float(distill_spec['weight']):g}, q={int(distill_spec.get('sample_q', 128))}, "
            f"loss={distill_spec.get('loss', 'mse')}")
    if global_residual is not None:
        log('global_residual: '
            f"bands={global_residual_spec.get('fourier_bands', [1, 2, 4, 8])}, "
            f"hidden={global_residual_spec.get('hidden_dim', 32)}, "
            f"max_delta_norm={global_residual_spec.get('max_delta_norm', 0.25):g}, "
            f"gate_radius_norm={global_residual_spec.get('gate_radius_norm', 0.15):g}")

    # 학습 대상은 A, B 만이어야 함
    for name, p in model.named_parameters():
        train = p.requires_grad
        assert train == (name.endswith('.A') or name.endswith('.B')), name
        if train:
            log(f' [TRAIN] {name} {tuple(p.shape)}')
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f'trainable(LoRA) param : {n_train} / total : {n_total} ({100 * n_train / n_total:.3f}%)')
    if global_residual is not None:
        n_residual = sum(p.numel() for p in global_residual.parameters())
        log(f'trainable(global residual) param : {n_residual}')

    optimizer = make_lora_optimizer(
        model, config['optimizer'], config.get('lora_plus'), global_residual,
        global_residual_spec.get('lr'))
    if config.get('lora_plus', {}).get('enabled', False):
        log('LoRA+: ' + ', '.join(f"{g.get('name', i)} lr={g['lr']:.3g}"
                                   for i, g in enumerate(optimizer.param_groups)))
    lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr']) if config.get('multi_step_lr') else None

    # 초기 동치 검증: B=0 이므로 LoRA on/off 출력이 정확히 같아야 함
    for b in val_loader:
        if b is not None:
            b = to_cuda(b)
            with torch.no_grad():
                model.eval()
                if global_residual is not None:
                    global_residual.eval()
                y_on, _ = forward(model, b, global_residual=global_residual)
                set_lora_enabled(model, False)
                y_off, _ = forward(model, b, global_residual=global_residual)
                set_lora_enabled(model, True)
            d = (y_on - y_off).abs().max().item()
            log(f'[init check] max|adapted LTE - LTE| = {d:.3e} (LoRA B/residual 최종층=0 이므로 0 이어야 함)')
            assert d == 0.0, d
            break

    epoch_max, epoch_val, epoch_save = config['epoch_max'], config.get('epoch_val'), config.get('epoch_save')
    early_cfg = config.get('early_stopping', {}) or {}
    patience = int(early_cfg.get('patience', 0))
    min_delta = float(early_cfg.get('min_delta', 0.0))
    stale_epochs = 0
    timer = utils.Timer()

    def save(name, epoch):
        torch.save({'lora': lora_state_dict(model), 'rank': rank, 'layers': layers,
                    'base_ckpt': base_ckpt, 'epoch': epoch,
                    'global_residual': (global_residual.state_dict()
                                        if global_residual is not None else None),
                    'global_residual_spec': global_residual_spec,
                    'optimizer': optimizer.state_dict()}, os.path.join(save_path, name))

    m0 = evaluate(val_loader, model, loss_spec, delta, pmin, pmax, val_seed, global_residual)
    # Treat the unadapted LTE as a valid candidate. Sparse fine-tuning can
    # regress a metric on the first epoch, so a true base fallback is safer
    # than forcing epoch 1 to become the initial "best" checkpoint.
    min_val_loss, min_val_mae = m0['loss'], m0['mae']
    min_val_abs_bias, max_val_corr = abs(m0['bias']), m0['corr']
    save('epoch-base.pth', 0)
    save('epoch-best-loss.pth', 0)
    save('epoch-best-mae.pth', 0)
    save('epoch-best-abs-bias.pth', 0)
    save('epoch-best-corr.pth', 0)
    log('epoch 0 (사전학습 LTE, LoRA B=0): val ' + fmt_metrics(m0))
    log('  intensity bins: ' + fmt_intensity_bins(m0))
    for k in ('loss', 'mae', 'bias', 'corr'):
        writer.add_scalar(f'val_{k}', m0[k], 0)

    for epoch in range(1, epoch_max + 1):
        t0 = timer.t()
        for i, group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"lr/{group.get('name', i)}", group['lr'], epoch)
        tr = train_one_epoch(train_loader, model, optimizer, loss_spec, delta, pmin, pmax,
                             distill_spec, global_residual)
        if lr_scheduler is not None:
            lr_scheduler.step()
        writer.add_scalars('loss', {'train_total': tr['total'], 'train_station': tr['station'],
                                    'train_distill': tr['distill']}, epoch)
        info = [f'epoch {epoch}/{epoch_max}',
                f"train: total={tr['total']:.4e} station={tr['station']:.4e} distill={tr['distill']:.4e}"]

        save('epoch-last.pth', epoch)
        if epoch_save and epoch % epoch_save == 0:
            save(f'epoch-{epoch}.pth', epoch)

        if epoch_val and epoch % epoch_val == 0:
            m = evaluate(val_loader, model, loss_spec, delta, pmin, pmax, val_seed,
                         global_residual)
            info.append('val: ' + fmt_metrics(m))
            writer.add_scalars('loss', {'val': m['loss']}, epoch)
            for k in ('mae', 'bias', 'corr'):
                writer.add_scalar(f'val_{k}', m[k], epoch)
            log('  intensity bins: ' + fmt_intensity_bins(m))
            if m['loss'] < min_val_loss - min_delta:
                min_val_loss = m['loss']
                save('epoch-best-loss.pth', epoch)
                stale_epochs = 0
            else:
                stale_epochs += 1
            if m['mae'] < min_val_mae:
                min_val_mae = m['mae']
                save('epoch-best-mae.pth', epoch)
            if abs(m['bias']) < min_val_abs_bias:
                min_val_abs_bias = abs(m['bias'])
                save('epoch-best-abs-bias.pth', epoch)
            if m['corr'] > max_val_corr:
                max_val_corr = m['corr']
                save('epoch-best-corr.pth', epoch)
            info.append(f'best val loss={min_val_loss:.4e}, |bias|={min_val_abs_bias:.4f}, r={max_val_corr:.3f}')
        info.append('dW/W=[' + ' '.join(f'{x:.3f}' for x in rel_delta_w(model)) + ']')

        t = timer.t()
        info.append('{} {}'.format(utils.time_text(t - t0), utils.time_text(t)))
        log(', '.join(info))
        writer.flush()
        if patience and stale_epochs >= patience:
            log(f'early stopping: no val-loss improvement >= {min_delta:g} for {stale_epochs} validations')
            break


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
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    save_name = args.name or ('_' + os.path.basename(args.config)[:-len('.yaml')])
    if args.tag is not None:
        save_name += '_' + args.tag
    main(cfg, os.path.join('./save', save_name))
