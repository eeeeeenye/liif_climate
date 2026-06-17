""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

""" LIIF 모델 학습 코드
    이미지 입력 → implicit representation을 생성하는 모델을 학습한다.
"""

import argparse              
import os                    

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


def make_data_loader(spec, tag=''):
    # dataset 설정이 없으면 loader를 만들지 않음
    if spec is None:
        return None

    # 실제 dataset 생성
    dataset = datasets.make(spec['dataset'])

    # wrapper dataset 적용
    # LIIF에서는 보통 이미지를 coord, cell, gt 형태로 바꿔주는 역할
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    # dataset 크기 출력
    log('{} dataset: size={}'.format(tag, len(dataset)))

    # dataset[0]의 각 항목 shape 출력
    # 예: inp, coord, cell, gt, level 등
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    # DataLoader 생성
    # train일 때만 shuffle=True
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=(tag == 'train'),
        num_workers=8,
        pin_memory=True
    )

    return loader


def make_data_loaders():
    # train dataset loader 생성
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')

    # validation dataset loader 생성
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')

    return train_loader, val_loader


def prepare_training():
    # pretrained checkpoint에서 이어서 학습하는 경우
    if config.get('resume') is not None:
        # checkpoint 불러오기
        sv_file = torch.load(config['resume'])

        # 저장된 model state_dict 가져오기
        pretrained_sd = sv_file["model"]["sd"]

        # config에 적힌 모델 구조 생성 후 GPU로 이동
        model = models.make(config["model"]).cuda()

        # pretrained weight 로드
        # strict=False이므로 구조가 일부 달라도 로드 가능
        missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)

        # 로드되지 않은 key, 예상 밖 key 확인
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)

        # encoder는 freeze
        # 학습 중 weight 업데이트 안 됨
        for p in model.encoder.parameters():
            p.requires_grad = False

        # imnet도 freeze
        for p in model.imnet.parameters():
            p.requires_grad = False

        # add_layer만 학습  (2026년 6월 이후 업데이트)
        for p in model.add_layer.parameters():
            p.requires_grad = True

        # optimizer는 add_layer 파라미터만 받음
        optimizer = utils.make_optimizer(
            model.add_layer.parameters(),
            config['optimizer']
        )

        # epoch은 1부터 새로 시작
        epoch_start = 1

        # learning rate scheduler가 없으면 None
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            # 특정 epoch에서 learning rate 감소
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    # resume이 없는 경우: 처음부터 전체 모델 학습
    else:
        # 모델 생성
        model = models.make(config['model']).cuda()

        # 전체 model parameter를 optimizer에 넣음
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])

        # 시작 epoch
        epoch_start = 1

        # scheduler 설정
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    # 모델 파라미터 수 출력
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    # train mode
    # Dropout/BatchNorm 등이 학습 모드로 동작
    model.train()

    # L1 loss 사용
    # pred와 gt의 절댓값 차이 평균
    loss_fn = nn.L1Loss()

    # loss 평균 계산용 객체
    train_loss = utils.Averager()

    # normalization 설정 가져오기
    data_norm = config['data_norm']

    # input normalization 값
    # shape: (1, C, 1, 1)
    # 이미지 채널별 정규화를 위해 broadcasting 가능하게 만듦
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

    # gt normalization 값
    # shape: (1, 1, C)
    # gt는 보통 (B, N, C)이므로 broadcasting 가능하게 만듦
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    # batch 단위 학습
    for batch in tqdm(train_loader, leave=False, desc='train'):

        # batch 안의 모든 tensor를 GPU로 이동
        for k, v in batch.items():
            batch[k] = v.cuda()

        # 입력 이미지 정규화
        inp = (batch['inp'] - inp_sub) / inp_div

        # 모델 forward
        # inp: low-resolution input image
        # level: scale 또는 추가 조건 정보로 보임
        # coord: query coordinate
        # cell: query cell size
        pred = model(inp, batch['level'], batch['coord'], batch['cell'])

        # ground truth 정규화
        gt = (batch['gt'] - gt_sub) / gt_div

        # prediction과 gt 사이 loss 계산
        loss = loss_fn(pred, gt)

        # loss 평균에 추가
        train_loss.add(loss.item())

        # 이전 gradient 초기화
        optimizer.zero_grad()

        # backpropagation
        loss.backward()

        # parameter update
        optimizer.step()

        # 메모리 정리용
        pred = None
        loss = None

    # epoch 전체 평균 train loss 반환
    return train_loss.item()


def main(config_, save_path):
    # 전역변수로 config, log, writer 사용
    global config, log, writer

    # config 저장
    config = config_

    # 저장 경로 만들고 logger, tensorboard writer 설정
    log, writer = utils.set_save_path(save_path)

    # 사용한 config를 save_path에 저장
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # train/validation dataloader 생성
    train_loader, val_loader = make_data_loaders()

    # data_norm 설정이 없으면 기본값 사용
    # 즉 normalization 안 함
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    # 모델, optimizer, scheduler 준비
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    # 사용할 GPU 개수 확인
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    # GPU가 여러 개면 DataParallel 사용
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    # 총 epoch 수
    epoch_max = config['epoch_max']

    # validation 주기
    epoch_val = config.get('epoch_val')

    # checkpoint 저장 주기
    epoch_save = config.get('epoch_save')

    # best PSNR 저장용
    max_val_v = -1e18

    # best validation loss 저장용
    min_val_loss = 1e18

    # 시간 측정용
    timer = utils.Timer()

    # epoch loop
    for epoch in range(epoch_start, epoch_max + 1):

        # epoch 시작 시간
        t_epoch_start = timer.t()

        # log 문자열 리스트
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        # 현재 learning rate 기록
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # 한 epoch 학습
        train_loss = train(train_loader, model, optimizer)

        # scheduler가 있으면 learning rate 업데이트
        if lr_scheduler is not None:
            lr_scheduler.step()

        # train loss 로그 추가
        log_info.append('train: loss={:.4f}'.format(train_loss))

        # tensorboard에 train loss 기록
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        # DataParallel이면 내부 model 꺼내기
        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model

        # 저장할 model config
        model_spec = config['model']

        # 현재 모델 파라미터 저장
        model_spec['sd'] = model_.state_dict()

        # optimizer config
        optimizer_spec = config['optimizer']

        # optimizer 상태 저장
        optimizer_spec['sd'] = optimizer.state_dict()

        # checkpoint dictionary
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        # 항상 마지막 epoch checkpoint 저장
        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        # epoch_save 주기마다 checkpoint 저장
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(
                sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch))
            )

        # validation 주기마다 평가
        if (epoch_val is not None) and (epoch % epoch_val == 0):

            # 평가용 model 선택
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model

            # validation PSNR과 validation loss 계산
            val_res, val_loss = eval_psnr(
                val_loader,
                model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize')
            )

            # validation 결과 로그
            log_info.append('val: psnr={:.4f}'.format(val_res))
            log_info.append('val loss={:.4f}'.format(val_loss))

            # tensorboard에 PSNR 기록
            writer.add_scalars('psnr', {'val': val_res}, epoch)

            # PSNR이 가장 좋으면 저장
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best-psnr.pth'))

            # validation loss가 가장 낮으면 저장
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(sv_file, os.path.join(save_path, 'epoch-best-loss.pth'))

            # best validation loss 기록
            log_info.append('best val loss={:.4f}'.format(min_val_loss))

        # 시간 정보 계산
        t = timer.t()

        # 전체 진행률
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)

        # 이번 epoch 걸린 시간
        t_epoch = utils.time_text(t - t_epoch_start)

        # 경과 시간, 예상 전체 시간
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)

        # 로그에 시간 추가
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        # 로그 출력
        log(', '.join(log_info))

        # tensorboard writer flush
        writer.flush()


# 이 파일을 직접 실행할 때만 아래 코드 실행
if __name__ == '__main__':

    # command line argument parser
    parser = argparse.ArgumentParser()

    # config yaml 파일 경로
    parser.add_argument('--config')

    # save 이름
    parser.add_argument('--name', default=None)

    # 추가 tag
    parser.add_argument('--tag', default=None)

    # 사용할 GPU 번호
    parser.add_argument('--gpu', default='2')

    # 인자 파싱
    args = parser.parse_args()

    # 사용할 GPU 지정
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # yaml config 파일 읽기
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    # 저장 폴더 이름 설정
    save_name = args.name

    # name을 따로 안 주면 config 파일명 기반으로 저장 이름 생성
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]

    # tag가 있으면 저장 이름 뒤에 붙임
    if args.tag is not None:
        save_name += '_' + args.tag

    # 최종 저장 경로
    save_path = os.path.join('./save', save_name)

    # main 함수 실행
    main(config, save_path)
