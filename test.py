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


def batched_predict(model, inp, level, coord, cell, bsize):
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
            )

            preds.append(pred)

            # 다음 chunk로 이동
            ql = qr

        # 모든 chunk 결과 합치기
        pred = torch.cat(preds, dim=1)

    return pred


def eval_psnr(
    loader,
    model,
    physical_norm=None,
    eval_type=None,
    eval_bsize=None,
    verbose=False,
    loss_type='mse'
):
    # evaluation mode
    model.eval()

    # normalized 공간에서 validation loss 계산 (train 손실과 동일하게 맞춤: 'mse' -> nn.MSELoss(기본), 'l1' -> nn.L1Loss)
    loss_fn = nn.L1Loss() if loss_type == 'l1' else nn.MSELoss()

    # validation loss 평균
    eval_loss = utils.Averager()

    # 평가 metric 설정
    if eval_type is None:
        metric_fn = utils.calc_psnr

    else:
        raise NotImplementedError(
            f'Unknown eval_type: {eval_type}'
        )

    # 평균 PSNR 저장
    val_res = utils.Averager()

    # progress bar
    pbar = tqdm(
        loader,
        leave=False,
        desc='val'
    )

    for batch in pbar:
        # pad_collate가 배치 내 유효 관측소가 하나도 없으면 None을 반환함
        # (예: val이 shuffle 안 되고 시간순이라, 연속 몇 시간 동안 비가 전혀 안 온
        # 구간이면 그 배치 전체가 결측 -> None). train()과 동일하게 건너뜀.
        if batch is None:
            continue

        # Tensor만 GPU로 이동
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)

        inp = batch['inp']

        # 모델 추론
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(
                    inp,
                    batch.get('level'),  # topo_nc_path 없는 config면 None
                    batch['coord'],
                    batch['cell']
                )

        else:
            with torch.no_grad():
                pred = batched_predict(
                    model,
                    inp,
                    batch.get('level'),
                    batch['coord'],
                    batch['cell'],
                    eval_bsize
                )

        gt = batch['gt']

        # ASOS station 수가 날짜마다 달라 padding된 batch(mask 포함)가 들어올 수 있음
        # padding된 가짜 지점(mask=0)이 loss/psnr에 섞이면 값이 실제보다 크게 나오므로 제외
        mask = batch.get('mask')
        if mask is not None:
            valid_mask = mask.bool()
            pred = pred[valid_mask]
            gt = gt[valid_mask]

        pred_base = pred
        gt_base = gt

        norm_type = physical_norm['type']

        if norm_type == "minmax":
            pmin = float(physical_norm['min'])
            pmax = float(physical_norm['max'])

            # [0,1] -> mm/day
            pred_mm = pred_base * (pmax - pmin) + pmin
            gt_mm = gt_base * (pmax - pmin) + pmin
        elif norm_type == "zscore":
            mean = float(physical_norm["mean"])
            std = float(physical_norm['std'])

            # zscore -> mm/day
            pred_mm = pred_base * std + mean
            gt_mm = gt_base * std + mean
        else:
            raise ValueError(
                f"Unknown physical_norm type: {norm_type}"
            )
        
        # normalized 공간, masked(유효 지점만) MSE
        loss = loss_fn(pred, gt)
        eval_loss.add(loss.item(), inp.shape[0])

        # z-score로 바꿔주기
        # pred_mm = pred * std + mean
        # gt_mm = gt * std + mean

        # 음수 강수량 방지
        pred_mm = torch.clamp(pred_mm, min=0.0)
        gt_mm = torch.clamp(gt_mm, min=0.0)

        # benchmark 평가용 reshape
        if eval_type is not None:

            # LR 크기
            ih, iw = batch['inp'].shape[-2:]

            # 출력 해상도 배율
            s = math.sqrt(
                batch['coord'].shape[1] / (ih * iw)
            )

            # 강수는 출력 채널 1개
            out_c = pred_mm.shape[-1]

            shape = [
                batch['inp'].shape[0],
                round(ih * s),
                round(iw * s),
                out_c
            ]

            # (B, H*W, C) -> (B, C, H, W)
            pred_mm = (
                pred_mm
                .view(*shape)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            gt_mm = (
                gt_mm
                .view(*shape)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        # mm 공간에서 PSNR 계산
        res = metric_fn(
            pred_mm,
            gt_mm,
            rgb_range=pmax
        )

        # 평균 PSNR 누적
        val_res.add(
            res.item(),
            inp.shape[0]
        )

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
        physical_norm=config.get("physical_norm"),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True
    )

    # 최종 PSNR 출력
    print(
        'result: {:.4f}'.format(res)
    )