""" Weighted-MAE 파인튜닝 효과 평가 (mm/day 공간).

    baseline(_climate_edsr_topo_80x100_topo_minmax_lte 의 마지막 에폭) 과
    weighted-MAE 파인튜닝 체크포인트들을 동일한 val 데이터(2020-2022 ASOS 관측소)에
    대해 추론하고, mm/day 공간에서
      - 전체 RMSE / MAE / bias / corr
      - Weighted-MAE (학습 손실 그 자체, w_min/w_max = config 값)
      - GT 강수 구간별(무강수 ~ 호우) MAE / bias / count
      - 분포 퍼센타일(90/95/99/99.9th) : GT vs 각 모델
      - 호우 임계값(>=50, >=80 mm/day) 에서의 bias / 포착률
    을 비교한다.

    실행:
      cd /home/inhye_yoo/deeplearning/inr/liif
      ~/anaconda3/envs/liif/bin/python inference_tools/eval_weighted_mae_ft_effect.py --gpu 2
"""

import argparse
import os
import sys

import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
import models


CONFIG = ('configs/train_climate/'
          'climate_edsr_topo_80x100_topo_minmax_lte_ft_weighted_mae.yaml')

CKPTS = {
    'baseline (pretrain, L1)':
        'save/_climate_edsr_topo_80x100_topo_minmax_lte/epoch-last.pth',
    'ft wMAE epoch-best-loss':
        'save/_climate_edsr_topo_80x100_topo_minmax_lte_ft_weighted_mae/epoch-best-loss.pth',
    'ft wMAE epoch-best-psnr':
        'save/_climate_edsr_topo_80x100_topo_minmax_lte_ft_weighted_mae/epoch-best-psnr.pth',
    'ft wMAE epoch-last (30)':
        'save/_climate_edsr_topo_80x100_topo_minmax_lte_ft_weighted_mae/epoch-last.pth',
}


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


@torch.no_grad()
def collect(model, loader, scale, shift):
    model.eval()
    P, G = [], []
    for batch in loader:
        if batch is None:
            continue
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)
        pred = model(batch['inp'], batch.get('level'),
                     batch['coord'], batch['cell'])
        gt = batch['gt']
        valid = batch['mask'].bool()
        pred = pred[valid].flatten()
        gt = gt[valid].flatten()
        P.append(pred.cpu())
        G.append(gt.cpu())
    pred = torch.cat(P).numpy() * scale + shift
    gt = torch.cat(G).numpy() * scale + shift
    pred = np.clip(pred, 0.0, None)
    gt = np.clip(gt, 0.0, None)
    return pred, gt


def wmae(pred, gt, w_min, w_max):
    w = np.clip(gt, w_min, w_max)
    return float(np.mean(w * np.abs(pred - gt)))


def metrics(pred, gt, w_min, w_max):
    err = pred - gt
    return {
        'n': len(gt),
        'RMSE': float(np.sqrt(np.mean(err ** 2))),
        'MAE': float(np.mean(np.abs(err))),
        'bias': float(np.mean(err)),
        'corr': float(np.corrcoef(pred, gt)[0, 1]),
        'wMAE': wmae(pred, gt, w_min, w_max),
    }


BINS = [(0, 1), (1, 10), (10, 20), (20, 50), (50, 80), (80, 1e9)]
PCTS = [50, 90, 95, 99, 99.9]
THRESHOLDS = [20, 50, 80]


def fmt_row(name, d, keys, widths):
    cells = [f'{name:<26}']
    for k, w in zip(keys, widths):
        v = d[k]
        cells.append(f'{v:>{w}.3f}' if isinstance(v, float) else f'{v:>{w}}')
    return ''.join(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='2')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(CONFIG) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    pn = config['physical_norm']
    scale, shift = float(pn['max']) - float(pn['min']), float(pn['min'])
    w_min = float(config['loss']['args']['w_min'])
    w_max = float(config['loss']['args']['w_max'])

    spec = config['val_dataset']
    ds = datasets.make(spec['dataset'])
    ds = datasets.make(spec['wrapper'], args={'dataset': ds})
    loader = DataLoader(ds, batch_size=spec['batch_size'], shuffle=False,
                        num_workers=8, pin_memory=True, collate_fn=pad_collate)
    print(f'val dataset size = {len(ds)}   '
          f'(denorm x_mm = x*{scale:.2f}+{shift:.2f};  '
          f'wMAE w_min={w_min}, w_max={w_max})\n')

    results = {}
    gt_ref = None
    for name, path in CKPTS.items():
        if not os.path.exists(path):
            print(f'[skip] {name}: not found {path}')
            continue
        sd = torch.load(path, map_location='cpu')
        model = models.make(sd['model'], load_sd=True).cuda()
        pred, gt = collect(model, loader, scale, shift)
        results[name] = pred
        gt_ref = gt
        del model
        torch.cuda.empty_cache()

    gt = gt_ref
    names = list(results.keys())

    # ---- 전체 지표 --------------------------------------------------------
    print('=' * 96)
    print('전체 (val 2020-2022, ASOS 관측소 전 지점/전 날짜, mm/day 공간)')
    print('-' * 96)
    keys = ['n', 'RMSE', 'MAE', 'bias', 'corr', 'wMAE']
    widths = [9, 10, 10, 10, 9, 12]
    print(f'{"model":<26}' + ''.join(f'{k:>{w}}' for k, w in zip(keys, widths)))
    base = None
    for nm in names:
        m = metrics(results[nm], gt, w_min, w_max)
        if base is None:
            base = m
        print(fmt_row(nm, m, keys, widths))
    print()

    # ---- GT 강수 구간별 --------------------------------------------------
    print('=' * 96)
    print('GT 강수 구간별  MAE / bias  (괄호=해당 구간 표본수)')
    print('-' * 96)
    hdr = f'{"bin (mm/day)":<16}{"count":>9}'
    for nm in names:
        hdr += f'{nm[:20]:>22}'
    print(hdr)
    for lo, hi in BINS:
        sel = (gt >= lo) & (gt < hi)
        cnt = int(sel.sum())
        label = f'[{lo:g}, {hi:g})' if hi < 1e8 else f'[{lo:g}, inf)'
        line = f'{label:<16}{cnt:>9}'
        for nm in names:
            if cnt == 0:
                line += f'{"-":>22}'
            else:
                e = results[nm][sel] - gt[sel]
                line += f'{np.mean(np.abs(e)):>10.2f}/{np.mean(e):>+9.2f} '
        print(line)
    print('  (MAE / bias, 단위 mm/day)\n')

    # ---- 분포 퍼센타일 -------------------------------------------------
    print('=' * 96)
    print('강수 분포 퍼센타일 값 (mm/day) : GT 대비 각 모델')
    print('-' * 96)
    line = f'{"pct":<8}{"GT":>10}'
    for nm in names:
        line += f'{nm[:20]:>22}'
    print(line)
    for p in PCTS:
        gv = np.percentile(gt, p)
        line = f'{p:<8}{gv:>10.2f}'
        for nm in names:
            pv = np.percentile(results[nm], p)
            line += f'{pv:>15.2f}({pv - gv:>+5.1f})'
        print(line)
    print('  (괄호 = GT 퍼센타일 대비 편차)\n')

    # ---- 호우 임계값 ---------------------------------------------------
    print('=' * 96)
    print('호우 임계값별  (GT >= thr)  : 표본수 / bias / RMSE / 평균 pred')
    print('-' * 96)
    for thr in THRESHOLDS:
        sel = gt >= thr
        cnt = int(sel.sum())
        print(f'  GT >= {thr} mm/day   (n={cnt}, 평균 GT={np.mean(gt[sel]):.1f})')
        for nm in names:
            e = results[nm][sel] - gt[sel]
            print(f'    {nm:<26} bias={np.mean(e):>+8.2f}  '
                  f'RMSE={np.sqrt(np.mean(e**2)):>7.2f}  '
                  f'mean_pred={np.mean(results[nm][sel]):>7.2f}')
        print()

    # ---- 요약 (baseline 대비) ----------------------------------------
    print('=' * 96)
    print('baseline 대비 변화 (음수 = 개선)')
    print('-' * 96)
    for nm in names[1:]:
        m = metrics(results[nm], gt, w_min, w_max)
        d_rmse = m['RMSE'] - base['RMSE']
        d_wmae = m['wMAE'] - base['wMAE']
        sel = gt >= 50
        e_b = results[names[0]][sel] - gt[sel]
        e_n = results[nm][sel] - gt[sel]
        d_hb = np.mean(e_n) - np.mean(e_b)
        print(f'  {nm:<26} dRMSE={d_rmse:>+7.3f}  dwMAE={d_wmae:>+8.2f}  '
              f'd(bias@GT>=50)={d_hb:>+7.2f}')
    print('=' * 96)


if __name__ == '__main__':
    main()
