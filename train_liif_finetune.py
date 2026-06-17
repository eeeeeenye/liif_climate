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
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

# 가장 큰 샘플 길이를 따라감
def pad_collate(batch):
    lengths = [b['coord'].shape[0] for b in batch]
    max_len = max(lengths)

    if max_len == 0:
        return None

    padded_inp = []
    padded_gt = []
    padded_coord=[]
    padded_cell = []
    padded_level=[]
    padded_area = []
    masks = []

    for b, q in zip(batch, lengths):
        inp = b['inp']
        gt = b['gt']
        coord = b['coord']
        cell = b['cell']
        level = b['level']
        # area = b['area']
        # print(area.shape, coord.shape)
        pad_len = max_len-q

        coord_padded = F.pad(coord, (0,0,0,pad_len))
        gt_padded = F.pad(gt, (0, 0, 0, pad_len))
        cell_padded = F.pad(cell, (0, 0, 0, pad_len)) 
        # area_padded = F.pad(area, (0, 0, 0, pad_len), value=0.0)

        mask = torch.zeros(max_len, dtype=torch.float32)
        mask[:q] = 1.0

        padded_inp.append(inp)
        padded_gt.append(gt_padded)
        padded_coord.append(coord_padded)
        padded_cell.append(cell_padded)
        padded_level.append(level)
        # padded_area.append(area_padded)
        masks.append(mask)
    
    batch_out = {
            'inp' : torch.stack(padded_inp, dim=0),
            'gt' : torch.stack(padded_gt, dim=0),
            'coord': torch.stack(padded_coord, dim=0),
            'cell': torch.stack(padded_cell, dim=0),
            'level': torch.stack(padded_level, dim=0),
            'mask': torch.stack(masks, dim=0),
            # 'area': torch.stack(padded_area, dim=0)
        }

    return batch_out

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(
        dataset, 
        batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), 
        num_workers=8, 
        pin_memory=True,
        collate_fn=pad_collate)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        pretrained_sd = sv_file['model']['sd']

        model = models.make(config['model']).cuda()
        missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)

        print("missing keys:", missing)
        print("unexpected keys:", unexpected)
        print("has add_layer:", hasattr(model, "add_layer"))

        for p in model.parameters():
            p.requires_grad = False

        for p in model.add_layer.parameters():
            p.requires_grad = True

        optimizer = utils.make_optimizer(
            model.add_layer.parameters(),
            config['optimizer']
        )

        # 학습시작 전 파라미터 직접 확인
        print("=" * 50)
        print("Trainable parameters:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(f" [TRAIN] {name}")
            else:
                print(f" [FROZEN] {name}")
        print("=" * 50)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable

        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,}  ← 이게 add_layer만이어야 함")
        print(f"Frozen params:    {frozen:,}")

        epoch_start = sv_file['epoch'] + 1

        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
            for _ in range(epoch_start - 1):
                lr_scheduler.step()

    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        epoch_start = 1

        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    return model, optimizer, epoch_start, lr_scheduler



def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss(reduction='none')
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        if batch is None:
            continue

        for k, v in batch.items():
            batch[k] = v.cuda()

        #  추가
        inp = (batch['inp'] - inp_sub) / inp_div

        pred = model(inp, batch['level'], batch['coord'], batch['cell'])  # , batch['area']
        gt = (batch['gt'] - gt_sub) / gt_div

        loss = loss_fn(pred, gt)

        mask = batch['mask']
        mask_expanded = mask.unsqueeze(-1)
        masked_loss = loss * mask_expanded

        loss = masked_loss.sum() / mask_expanded.sum()
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None

    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18
    min_val_loss = 1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res, val_loss = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            log_info.append('val loss: {:.4f}'.format(val_loss))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(sv_file, os.path.join(save_path, 'epoch-best-loss.pth'))
            log_info.append('best val loss={:.4f}'.format(min_val_loss))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

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
