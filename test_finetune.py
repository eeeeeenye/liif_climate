import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, obs_loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

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

    loader = zip(loader, obs_loader)

    pbar = tqdm(loader, leave=False, desc='val')
    for (sr_batch, obs_batch) in pbar:
        for k, v in sr_batch.items():
            sr_batch[k] = v.cuda()
        for k, v in obs_batch.items():
            obs_batch[k] = v.cuda()

        inp = (sr_batch['inp'] - inp_sub) / inp_div

        # === 마스킹된 obs 값 처리 ===
        mask = (obs_batch['gt'] != -1).squeeze(-1)
        masked = obs_batch['gt'].squeeze(-1)
        obs_value = torch.where(mask, masked, torch.tensor(0.0, device=masked.device))

    # === 모델 예측 ===
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, sr_batch['coord'], sr_batch['cell'], obs_batch['coord'])
        else:
            pred = batched_predict(model, inp,
                sr_batch['coord'], sr_batch['cell'], eval_bsize)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None:  # shaving eval용 reshape
            ih, iw = sr_batch['inp'].shape[-2:]
            s = math.sqrt(sr_batch['coord'].shape[1] / (ih * iw))
            shape = [sr_batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
            sr_batch['gt'] = sr_batch['gt'].view(*shape).permute(0, 3, 1, 2).contiguous()
            res = metric_fn(pred, sr_batch['gt'])  # shaving-eval인 경우만 여기서 끝
        else:
        # === obs_gt도 정규화 및 확장해서 concat ===
            obs_gt = (obs_value - gt_sub) / gt_div          
            obs_gt = obs_gt.permute(1,2,0).repeat(1, 1, 3)               
            # print(obs_gt.shape, sr_batch['gt'].shape)
            gt = torch.cat([sr_batch['gt'], obs_gt], dim=1) # (B, Q1 + Q2, 3)
            res = metric_fn(pred, gt)

        val_res.add(res.item(), inp.shape[0])


        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
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

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))
