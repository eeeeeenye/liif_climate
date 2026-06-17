import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

import datasets
import models
import utils

  
def batched_predict(model, inp, level, coord, cell, area, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(level, coord[:, ql: qr, :], cell[:, ql: qr, :], area)
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()
    loss_fn = nn.L1Loss()
    eval_loss = utils.Averager()
    obs_loss = utils.Averager()

    alpha_imerg, alpha_obs = 0.1, 0.2

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div

        with torch.no_grad():
            if eval_bsize is None:
                pred = model(inp, batch['level'], batch['coord'], batch['cell'])
            else:
                pred = batched_predict(
                    model, inp, batch['level'], batch['coord'], batch['cell'], eval_bsize
                )

        gt = (batch['gt'] - gt_sub) / gt_div

        obs_mask = batch['is_obs'].squeeze(-1).bool()
        imerg_mask = ~obs_mask

        loss_obs = torch.tensor(0.0, device=pred.device)
        loss_imerg = torch.tensor(0.0, device=pred.device)

        if pred[obs_mask].numel() > 0:
            loss_obs = loss_fn(pred[obs_mask], gt[obs_mask])

        if pred[imerg_mask].numel() > 0:
            loss_imerg = loss_fn(pred[imerg_mask], gt[imerg_mask])

        loss = alpha_imerg * loss_imerg + alpha_obs * loss_obs
        eval_loss.add(loss.item())
        eval_loss.add(loss_obs)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None:  # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]

            pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape).permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item(), eval_loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='2')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res, eval_loss = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))