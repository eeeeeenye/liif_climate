import os
import time
import shutil
import math
import json

import torch
import numpy as np
from torch.optim import SGD, Adam
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        # 전체 좌표 범위 [-1, 1]
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        
        r = (v1 - v0) / (2 * n)                             # -1 ~ 1 사이를 n(해상도)로 분활 후 중심점 계산
        seq = v0 + r + (2 * r) * torch.arange(n).float()    # 0~99를 곱해서 각 분활 그리드의 중심 좌표를 구함
        # d = coord_seqs.append(seq)
        coord_seqs.append(seq)
    
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)  # 마지막 축에 새로 만들어진 텐서의 차원을 추가 # 같은 위치(인덱스)에 있는 H(y), W(x)를 stack으로 묶어서 새로운 텐서 생성
    if flatten:
    # 앞쪽의 모든 차원을 flatten 해서 마지막 차원만 남겨둠
    # 예를 들면 shape = (60, 60, 2) -> (3600, 2)
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])      #이미지 shape 중 H, W 중 하나만 가져옴
    rgb = img.view(1, -1).permute(1, 0)
    return coord, rgb

# flipped default 값 False로 변경 -> xarray 원본 lat이 남->북 증가라면 true
def latlon_to_pixel_idx(lat, lon, lat_min, lat_max, lon_min, lon_max, H, W, flipped=False):
    """
    위성 강수장 이미지로 변환 시 min-max 스케일링 기법 사용
    따라서, asos도 같은 scaling 기법 사용 후 인덱스 추출 (j, i)
    """
    j = (lon - lon_min) / (lon_max - lon_min) * (W-1)

    if flipped:
        i = (lat - lat_min) / (lat_max - lat_min) * (H - 1)
        i = (H - 1) - i
    else:
        # 일반적인 이미지는 바로 매핑
        i = (lat_max - lat) / (lat_max - lat_min) * (H - 1)

    return i,j

def pixel_to_liif_coord(i, j, H, W):
    """
    j : x축 인덱스 
    i : y축 인덱스
    H, W : hr 해상도
    --방법 --
    1. ((j + 0.5) / W) -> 픽셀 인덱스에서 중심좌표로 shift 후 해상도로 나눔
    2. 원하는 좌표공간은 -1 ~ 1 하지만 1번 식은 0 ~ 1이므로 0 ~ 2 공간으로 변경
    3. -1을 빼줌으로써 -1 ~ 1사이 범위로 만듦
    """
    x = ((j + 0.5) / W) * 2.0 - 1.0 
    y = ((i + 0.5) / H) * 2.0 - 1.0
    return x, y

# get the grid idx and obs value
def obs_get_idx_val(data):
    idx = (data != 0).nonzero(as_tuple=False)
    vals = data[data != 0]

    return idx, vals

def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)
