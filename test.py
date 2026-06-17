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
    # gradient 계산 비활성화 (추론용)
    with torch.no_grad():

        # Encoder 통과 → feature map 생성
        model.gen_feat(inp)

        # 전체 query point 개수
        n = coord.shape[1]

        # query 시작 위치
        ql = 0

        # 예측 결과 저장 리스트
        preds = []

        # query를 bsize 단위로 나눠서 예측
        while ql < n:

            # 현재 batch 끝 위치
            qr = min(ql + bsize, n)

            # 현재 query chunk만 예측
            pred = model.query_rgb(
                level,
                coord[:, ql: qr, :],
                cell[:, ql: qr, :],
                area
            )

            preds.append(pred)

            # 다음 chunk로 이동
            ql = qr

        # 모든 chunk 결과 합치기
        pred = torch.cat(preds, dim=1)

    return pred


def eval_psnr(loader, model,
              data_norm=None,
              eval_type=None,
              eval_bsize=None,
              verbose=False):

    # evaluation mode
    model.eval()

    # MAE loss
    loss_fn = nn.L1Loss()

    # validation loss 평균 계산용
    eval_loss = utils.Averager()

    # normalization 정보가 없으면 기본값 사용
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    # input normalization 파라미터
    t = data_norm['inp']

    inp_sub = torch.FloatTensor(
        t['sub']
    ).view(1, -1, 1, 1).cuda()

    inp_div = torch.FloatTensor(
        t['div']
    ).view(1, -1, 1, 1).cuda()

    # GT normalization 파라미터
    t = data_norm['gt']

    gt_sub = torch.FloatTensor(
        t['sub']
    ).view(1, 1, -1).cuda()

    gt_div = torch.FloatTensor(
        t['div']
    ).view(1, 1, -1).cuda()

    # 평가 metric 설정
    if eval_type is None:

        metric_fn = utils.calc_psnr

    elif eval_type.startswith('div2k'):

        scale = int(eval_type.split('-')[1])

        metric_fn = partial(
            utils.calc_psnr,
            dataset='div2k',
            scale=scale
        )

    elif eval_type.startswith('benchmark'):

        scale = int(eval_type.split('-')[1])

        metric_fn = partial(
            utils.calc_psnr,
            dataset='benchmark',
            scale=scale
        )

    else:
        raise NotImplementedError

    # 평균 PSNR 저장용
    val_res = utils.Averager()

    # progress bar
    pbar = tqdm(loader, leave=False, desc='val')

    for batch in pbar:

        # 모든 batch tensor를 GPU로 이동
        for k, v in batch.items():
            batch[k] = v.cuda()

        # input normalization
        inp = (batch['inp'] - inp_sub) / inp_div

        # 한번에 예측 가능한 경우
        if eval_bsize is None:

            with torch.no_grad():

                pred = model(
                    inp,
                    batch['level'],
                    batch['coord'],
                    batch['cell']
                )

        # query가 많으면 chunk 단위 추론
        else:

            pred = batched_predict(
                model,
                inp,
                batch['level'],
                batch['coord'],
                batch['cell'],
                eval_bsize
            )

        # GT normalization
        gt = (batch['gt'] - gt_sub) / gt_div

        # MAE 계산
        loss = loss_fn(pred, gt)

        # loss 누적
        eval_loss.add(loss.item())

        # denormalization
        pred = pred * gt_div + gt_sub

        # 0~1 범위 제한
        pred.clamp_(0, 1)

        # benchmark 평가용 reshape
        if eval_type is not None:

            # LR 크기
            ih, iw = batch['inp'].shape[-2:]

            # scale factor 계산
            s = math.sqrt(
                batch['coord'].shape[1] / (ih * iw)
            )

            shape = [
                batch['inp'].shape[0],
                round(ih * s),
                round(iw * s),
                3
            ]

            # (N,HW,C) → (N,C,H,W)
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2) \
                .contiguous()

            batch['gt'] = batch['gt'] \
                .view(*shape) \
                .permute(0, 3, 1, 2) \
                .contiguous()

        # PSNR 계산
        res = metric_fn(pred, batch['gt'])

        # 평균 PSNR 누적
        val_res.add(
            res.item(),
            inp.shape[0]
        )

        # 진행중 출력
        if verbose:
            pbar.set_description(
                'val {:.4f}'.format(
                    val_res.item()
                )
            )

    return val_res.item(), eval_loss.item()


if __name__ == '__main__':

    # 명령행 인자
    parser = argparse.ArgumentParser()

    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='2')

    args = parser.parse_args()

    # 사용할 GPU 지정
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # yaml config 읽기
    with open(args.config, 'r') as f:

        config = yaml.load(
            f,
            Loader=yaml.FullLoader
        )

    # test dataset 설정
    spec = config['test_dataset']

    # dataset 생성
    dataset = datasets.make(
        spec['dataset']
    )

    # wrapper 적용
    dataset = datasets.make(
        spec['wrapper'],
        args={'dataset': dataset}
    )

    # dataloader 생성
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        num_workers=8,
        pin_memory=True
    )

    # checkpoint 로드
    model_spec = torch.load(
        args.model
    )['model']

    # 모델 생성 + weight 로드
    model = models.make(
        model_spec,
        load_sd=True
    ).cuda()

    # validation/test 수행
    res, eval_loss = eval_psnr(
        loader,
        model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True
    )

    # 최종 PSNR 출력
    print(
        'result: {:.4f}'.format(res)
    )