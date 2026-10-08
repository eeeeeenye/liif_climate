""" ASOS 시간별 관측으로 MLP adapter 를 Bernoulli-Gamma 분포 출력(p, alpha, beta)으로 파인튜닝한다.

    기존 train_liif_finetune.py(MSE, 스칼라 출력 adapter) 와 같은 구조에서
      - adapter 출력이 1개 -> 3개 (models/bg_adapter.py)
      - 손실이 MSE -> Bernoulli-Gamma NLPD (losses_bernoulli_gamma.py, Cannon 2008 식 12)
    로만 바뀐다.  encoder / imnet 은 동결, add_layer(bg-adapter) 만 학습.  기존 파일은 수정하지 않음.

    주의: p(강수확률) 항은 '건조 시각'이 있어야 학습된다.  obs CSV 는 make_hsr_hourly_obs.py --fill zero
    로 만든 것(건조 = 0.0, 결측 = -2.0 아님)을 써야 한다.

    학습 시작 전 로그에 무조건부(기후학적) Bernoulli-Gamma NLPD 를 val 에 대해 출력한다 (Cannon 의 null 모델).
    val 지표: nll(및 skill = 1 - nll/nll_clim), 조건부 평균 mu=p*alpha*beta 의 rmse/bias/corr,
             LIIF 원출력 rmse(rmse_base), 강수 유무 POD/FAR/CSI/TSS (p > p_thr), mean_p vs wet_frac(보정).

    실행:
      python train_liif_finetune_bg.py \
        --config configs/train_climate/climate_edsr_radar_hsr_2x_minmax_liif_ft_bg_mlp_lag1h.yaml --gpu 2
"""

import argparse
import math
import os

import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import models.bg_adapter          # noqa: F401  ('bg-adapter' 레지스트리 등록)
import utils
from train_liif_finetune import pad_collate, set_seed   # 기존 로직 재사용 (main 은 실행되지 않음)
from losses_bernoulli_gamma import (bg_params, bg_mean, bg_nll, bg_nll_elem, fit_climatology)


def make_data_loader(spec, tag=''):
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    log('{} dataset: size={}'.format(tag, len(dataset)))
    return DataLoader(dataset, batch_size=spec['batch_size'], shuffle=(tag == 'train'),
                      num_workers=8, pin_memory=True, collate_fn=pad_collate)


def denorm(x):
    """ [0,1] minmax -> mm/h """
    pn = config['physical_norm']
    assert pn['type'] == 'minmax'
    return x * (float(pn['max']) - float(pn['min'])) + float(pn['min'])


def to_cuda(batch):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.cuda(non_blocking=True)
    return batch


def forward_bg(model, batch, adapter=True):
    """ adapter 는 local ensemble 전에(방향별, liif_baseline.py L219) 1회 적용되고 그 raw 로짓이
        가중 blend 된다 -> raw [B,q,3].  adapter=False 면 adapter 를 건너뛴 LIIF 원출력 [B,q,1]
        (rmse_base 용).  forward 끝(L287)의 두 번째 correction_layer 호출은 prepare_training 에서
        만든 래퍼가 무시한다(이미 3채널이면 통과). """
    model._bg_state['on'] = adapter
    return model(batch['inp'], batch['level'], batch['coord'], batch['cell'])


def prepare_training():
    model = models.make(config['model']).cuda()
    # correction_layer 는 query_rgb(방향별, 입력 1채널)와 forward(blend 후) 양쪽에서 불리는데,
    # 방향별 1회만 적용하고 blend 후 호출(이미 3채널)은 통과시키는 래퍼로 인스턴스 속성 대체 (기존 파일 무수정)
    model._bg_state = {'on': True}
    model.correction_layer = lambda preds: (
        model.add_layer(preds) if (model._bg_state['on'] and preds.shape[-1] == 1) else preds)
    sv = torch.load(config['resume'])
    missing, unexpected = model.load_state_dict(sv['model']['sd'], strict=False)
    log('missing keys (새로 학습): {}'.format([k for k in missing]))
    log('unexpected keys: {}'.format(list(unexpected)))

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.imnet.parameters():
        p.requires_grad = False
    for p in model.add_layer.parameters():
        p.requires_grad = True

    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log('frozen param : {}'.format(total - trainable))
    log('trainable param : {}'.format(trainable))

    lr_scheduler = None
    if config.get('multi_step_lr') is not None:
        lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
    return model, optimizer, lr_scheduler


@torch.no_grad()
def climatology(loader, bgc):
    """ train obs 로 무조건부 Bernoulli-Gamma MLE 를 구한다 (충분통계량만 누적). """
    n = n_wet = sy = sly = 0.0
    for batch in tqdm(loader, leave=False, desc='clim'):
        if batch is None:
            continue
        y = denorm(batch['gt'].squeeze(-1))
        m = batch['mask'].bool()
        y = y[m]
        wet = y >= bgc['wet_thr']
        yw = y[wet].double()
        n += y.numel(); n_wet += yw.numel()
        sy += yw.sum().item(); sly += torch.log(yw.clamp(min=bgc['y_floor'])).sum().item()
    p, a, b = fit_climatology(n, n_wet, sy, sly, bgc['y_floor'])
    log('climatology (train obs): n={:.0f} wet_frac={:.4f} alpha={:.3f} beta={:.3f}mm mean_wet={:.3f}mm'.format(
        n, p, a, b, a * b))
    return p, a, b


def train(loader, model, optimizer, bgc):
    model.train()
    avg = utils.Averager()
    for batch in tqdm(loader, leave=False, desc='train'):
        if batch is None:
            continue
        batch = to_cuda(batch)
        raw = forward_bg(model, batch)   # [B,q,3]
        y = denorm(batch['gt'])
        loss = bg_nll(raw, y, batch['mask'], bgc['wet_thr'], bgc['y_floor'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg.add(loss.item())
    return avg.item()


@torch.no_grad()
def evaluate(loader, model, bgc, clim):
    model.eval()
    raws, ys, bases = [], [], []
    for batch in tqdm(loader, leave=False, desc='val'):
        if batch is None:
            continue
        batch = to_cuda(batch)
        raw = forward_bg(model, batch)
        m = batch['mask'].bool()
        raws.append(raw[m].cpu())
        ys.append(denorm(batch['gt'].squeeze(-1))[m].cpu())
        base = forward_bg(model, batch, adapter=False)          # LIIF 원출력 (adapter 없음)
        bases.append(denorm(base.squeeze(-1))[m].clamp(min=0).cpu())
    raw, y, base = torch.cat(raws), torch.cat(ys), torch.cat(bases)

    p, a, b = bg_params(raw)
    nll_e, wet = bg_nll_elem(p, a, b, y, bgc['wet_thr'], bgc['y_floor'])
    nll = nll_e.mean().item()

    cp, ca, cb = [torch.full_like(p, v) for v in clim]
    nll_clim = bg_nll_elem(cp, ca, cb, y, bgc['wet_thr'], bgc['y_floor'])[0].mean().item()

    mu = p * a * b
    res = {
        'nll': nll,
        'nll_clim': nll_clim,
        'skill': 1.0 - nll / nll_clim,
        'nll_dry': nll_e[~wet].mean().item(),
        'nll_wet': nll_e[wet].mean().item() if wet.any() else float('nan'),
        'rmse_mu': torch.sqrt(((mu - y) ** 2).mean()).item(),
        'rmse_base': torch.sqrt(((base - y) ** 2).mean()).item(),
        'bias_mu': (mu - y).mean().item(),
        'corr_mu': torch.corrcoef(torch.stack([mu, y]))[0, 1].item(),
        'corr_base': torch.corrcoef(torch.stack([base, y]))[0, 1].item(),
        'mean_p': p.mean().item(),
        'wet_frac': wet.float().mean().item(),
    }
    # 조건부 평균 mu 는 p 로 희석돼 wet 시 과소가 정상 -> wet 에서 alpha*beta(젖었을 때 평균)도 같이 본다
    if wet.any():
        res['wet_bias_ab'] = ((a * b)[wet] - y[wet]).mean().item()
    # 강수 유무 (p > p_thr)
    pred_wet = p > bgc['p_thr']
    hit = (pred_wet & wet).sum().item(); miss = (~pred_wet & wet).sum().item()
    fa = (pred_wet & ~wet).sum().item(); cn = (~pred_wet & ~wet).sum().item()
    res['POD'] = hit / max(hit + miss, 1)
    res['FAR'] = fa / max(hit + fa, 1)
    res['CSI'] = hit / max(hit + miss + fa, 1)
    res['TSS'] = res['POD'] - fa / max(fa + cn, 1)
    return res


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    bgc = {'wet_thr': 0.05, 'y_floor': 0.05, 'p_thr': 0.5}
    bgc.update(config.get('bg', {}))

    train_loader = make_data_loader(config['train_dataset'], 'train')
    val_loader = make_data_loader(config['val_dataset'], 'val')
    model, optimizer, lr_scheduler = prepare_training()

    clim = climatology(train_loader, bgc)
    r0 = evaluate(val_loader, model, bgc, clim)
    log('epoch 0 (init): ' + ', '.join('{}={:.4f}'.format(k, v) for k, v in r0.items()))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val', 1)
    epoch_save = config.get('epoch_save')
    best_nll = 1e18
    timer = utils.Timer()

    for epoch in range(1, epoch_max + 1):
        t0 = timer.t()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        train_nll = train(train_loader, model, optimizer, bgc)
        if lr_scheduler is not None:
            lr_scheduler.step()
        info = ['epoch {}/{}'.format(epoch, epoch_max), 'train nll={:.4f}'.format(train_nll)]
        writer.add_scalars('nll', {'train': train_nll}, epoch)

        spec = config['model']
        spec['sd'] = model.state_dict()
        opt_spec = config['optimizer']
        opt_spec['sd'] = optimizer.state_dict()
        sv = {'model': spec, 'optimizer': opt_spec, 'epoch': epoch, 'bg': bgc, 'clim': clim}
        torch.save(sv, os.path.join(save_path, 'epoch-last.pth'))
        if epoch_save is not None and epoch % epoch_save == 0:
            torch.save(sv, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if epoch % epoch_val == 0:
            r = evaluate(val_loader, model, bgc, clim)
            info.append('val nll={:.4f} (clim {:.4f}, skill {:.3f})'.format(r['nll'], r['nll_clim'], r['skill']))
            info.append('rmse mu/base={:.3f}/{:.3f} corr mu/base={:.3f}/{:.3f} bias_mu={:.3f}'.format(
                r['rmse_mu'], r['rmse_base'], r['corr_mu'], r['corr_base'], r['bias_mu']))
            info.append('POD={:.3f} FAR={:.3f} CSI={:.3f} TSS={:.3f} mean_p/wet_frac={:.4f}/{:.4f}'.format(
                r['POD'], r['FAR'], r['CSI'], r['TSS'], r['mean_p'], r['wet_frac']))
            writer.add_scalars('nll', {'val': r['nll']}, epoch)
            if r['nll'] < best_nll:
                best_nll = r['nll']
                torch.save(sv, os.path.join(save_path, 'epoch-best-nll.pth'))
            info.append('best val nll={:.4f}'.format(best_nll))

        t = timer.t()
        prog = epoch / epoch_max
        info.append('{} {}/{}'.format(utils.time_text(t - t0), utils.time_text(t), utils.time_text(t / prog)))
        log(', '.join(info))
        writer.flush()


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
