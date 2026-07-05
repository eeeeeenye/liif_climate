import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples


# level + observation
# 이 wrapper는 이미 LR/HR pair가 존재하는 dataset을 받아서
# LIIF 학습에 필요한 inp, coord, cell, gt 형태로 바꿔주는 클래스
# 제 모델에서는 SRImplicitPaired를 사용하였습니다!
@register('sr-implicit-paired')  # 이 클래스를 datasets registry에 'sr-implicit-paired' 이름으로 등록
class SRImplicitPaired(Dataset):  # PyTorch Dataset을 상속받는 LIIF용 wrapper 클래스

    def __init__(self, dataset, inp_size=None, augment=False,
                 sample_q=None, obs_state=False):
        self.dataset = dataset        # 원본 dataset 저장
        self.inp_size = inp_size      # LR patch 크기, None이면 전체 이미지 사용
        self.augment = augment        # data augmentation 사용 여부
        self.sample_q = sample_q      # HR query point 중 sampling할 개수
        self.obs_state = obs_state    # 관측소 supervision 사용 여부

    def __len__(self):
        return len(self.dataset)      # 원본 dataset 길이를 그대로 반환

    def __getitem__(self, idx):
        if self.obs_state:            # 관측소 자료를 사용하는 경우
            img_lr, img_hr, obs_, level = self.dataset[idx]
            # img_lr : (1, 50, 50)      # LR 입력 이미지
            # img_hr : (1, 100, 100)    # HR 정답 이미지
            # obs_   : 관측소 dict      # {'coord', 'gt'} 포함
            # level  : (1, 100, 100)    # topo / level 정보

        else:                         # 관측소 자료를 사용하지 않는 경우
            img_lr, img_hr, level = self.dataset[idx]
            obs_ = None               # observation 없음

        s = img_hr.shape[-2] // img_lr.shape[-2]
        # scale factor 계산
        # 100 // 50 = 2

        if self.inp_size is None:     # patch crop 없이 전체 이미지 사용
            h_lr, w_lr = img_lr.shape[-2:]
            # h_lr = 50, w_lr = 50

            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            # img_hr를 LR 크기 * scale에 맞게 자름
            # img_hr : (1, 100, 100)

            crop_lr, crop_hr = img_lr, img_hr
            # crop_lr : (1, 50, 50)
            # crop_hr : (1, 100, 100)

            crop_level = level
            # crop_level : (1, 100, 100)

        else:                         # inp_size가 있으면 random crop 사용
            w_lr = self.inp_size      # LR patch 크기

            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            # LR 이미지에서 crop 시작 height index 선택

            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            # LR 이미지에서 crop 시작 width index 선택

            crop_lr = img_lr[:, x0:x0+w_lr, y0:y0+w_lr]
            # LR patch crop
            # crop_lr : (1, w_lr, w_lr)

            w_hr = w_lr * s
            # HR patch 크기 = LR patch 크기 * scale

            x1, y1 = x0 * s, y0 * s
            # LR crop 위치를 HR 좌표계로 변환

            crop_hr = img_hr[:, x1:x1+w_hr, y1:y1+w_hr]
            # HR patch crop
            # crop_hr : (1, w_hr, w_hr)

            crop_level = level[:, x1:x1+w_hr, y1:y1+w_hr]
            # level도 HR과 같은 위치로 crop
            # crop_level : (1, w_hr, w_hr)

        if self.augment:              # augmentation을 사용하는 경우
            hflip = random.random() < 0.5  # 50% 확률로 height 방향 flip
            vflip = random.random() < 0.5  # 50% 확률로 width 방향 flip
            dflip = random.random() < 0.5  # 50% 확률로 H/W transpose

            def augment(x):           # crop_lr, crop_hr에 같은 변환 적용
                if x is None:
                    return None       # None이면 그대로 반환

                if hflip:
                    x = x.flip(-2)    # height 방향 flip

                if vflip:
                    x = x.flip(-1)    # width 방향 flip

                if dflip:
                    x = x.transpose(-2, -1)  # height와 width 축 교환

                return x              # augmentation된 tensor 반환

            crop_lr = augment(crop_lr)
            # crop_lr shape 유지: (1, 50, 50)

            crop_hr = augment(crop_hr)
            # crop_hr shape 유지: (1, 100, 100)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        # HR 이미지를 LIIF query 형태로 변환
        # crop_hr  : (1, 100, 100)
        # hr_coord : (10000, 2)
        # hr_rgb   : (10000, 1)

        if self.sample_q is not None:  # query point를 일부만 sampling하는 경우
            n = len(hr_coord)          # 전체 HR pixel 개수, 여기서는 10000

            q = min(self.sample_q, n)  # sample_q가 n보다 크면 n개만 사용

            sample_idx = np.random.choice(n, q, replace=False)
            # 중복 없이 q개 index 선택

            hr_coord = hr_coord[sample_idx]
            # 선택된 좌표만 사용
            # hr_coord : (sample_q, 2)

            hr_rgb = hr_rgb[sample_idx]
            # 선택된 좌표의 HR 값만 사용
            # hr_rgb : (sample_q, 1)

        if obs_ is not None:           # 관측소 supervision을 사용하는 경우
            obs_coord = obs_['coord']
            # 관측소 좌표
            # obs_coord : (stations, 2)

            obs_val = obs_['gt']
            # 관측소 관측값
            # obs_val : (stations,)

            if obs_val.ndim == 1:
                obs_val = obs_val.unsqueeze(-1)
                # channel 차원 추가
                # obs_val : (stations, 1)

            valid_mask = obs_val.squeeze(-1) != -2.0
            # 결측값 -2.0이 아닌 관측소만 True

            obs_coord = obs_coord[valid_mask]
            # 유효한 관측소 좌표만 남김
            # obs_coord : (valid_stations, 2)

            obs_val = obs_val[valid_mask]
            # 유효한 관측값만 남김
            # obs_val : (valid_stations, 1)

            obs_val = obs_val.expand(-1, hr_rgb.shape[1])
            # gt channel 수를 hr_rgb channel 수와 맞춤
            # 현재 channel=1이므로 shape 유지
            # obs_val : (valid_stations, 1)

            hr_coord = obs_coord
            # 기존 HR pixel 좌표를 관측소 좌표로 교체
            # hr_coord : (valid_stations, 2)

            hr_rgb = obs_val
            # 기존 HR pixel gt를 관측소 관측값으로 교체
            # hr_rgb : (valid_stations, 1)

        cell = torch.ones_like(hr_coord)
        # coord와 같은 shape로 cell 생성
        # obs 사용 X : (sample_q, 2) 또는 (10000, 2)
        # obs 사용 O : (valid_stations, 2)

        cell[:, 0] *= 2 / crop_hr.shape[-2]
        # y 방향 cell 크기 설정
        # crop_hr height = 100이면 2 / 100 = 0.02

        cell[:, 1] *= 2 / crop_hr.shape[-1]
        # x 방향 cell 크기 설정
        # crop_hr width = 100이면 2 / 100 = 0.02

        return {
            'inp': crop_lr,        # LR input, shape: (1, 50, 50)
            'coord': hr_coord,     # query 좌표, obs 사용 시: (valid_stations, 2)
            'cell': cell,          # query cell 크기, obs 사용 시: (valid_stations, 2)
            'gt': hr_rgb,          # 정답값, obs 사용 시: (valid_stations, 1)
            'level': crop_level,   # topo/level input, shape: (1, 100, 100)
        }

def resize_fn(img, size):
    # Tensor image를 PIL image로 변환
    # BICUBIC resize 수행
    # 다시 Tensor로 변환
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)
        )
    )


# HR 이미지만 있는 dataset에서
# bicubic downsampling으로 LR을 만든 뒤 LIIF 학습 형태로 바꾸는 wrapper
@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        # 원본 dataset
        self.dataset = dataset

        # LR patch 크기
        self.inp_size = inp_size

        # 최소 scale
        self.scale_min = scale_min

        # 최대 scale이 없으면 고정 scale로 사용
        if scale_max is None:
            scale_max = scale_min

        self.scale_max = scale_max

        # augmentation 여부
        self.augment = augment

        # query sampling 개수
        self.sample_q = sample_q

    def __len__(self):
        # 원본 dataset 길이
        return len(self.dataset)

    def __getitem__(self, idx):
        # 원본 HR image 하나를 가져옴
        img = self.dataset[idx]

        # scale_min~scale_max 사이에서 랜덤 배율 선택
        s = random.uniform(self.scale_min, self.scale_max)

        # inp_size가 없으면 전체 이미지를 사용
        if self.inp_size is None:
            # scale s에 맞는 LR 크기 계산
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)

            # HR 크기를 LR*s에 맞도록 자름
            img = img[:, :round(h_lr * s), :round(w_lr * s)]

            # HR을 bicubic downsample해서 LR 생성
            img_down = resize_fn(img, (h_lr, w_lr))

            crop_lr, crop_hr = img_down, img

        # inp_size가 있으면 HR에서 patch를 뽑고 downsample해서 LR 생성
        else:
            # LR patch 크기
            w_lr = self.inp_size

            # scale을 고려한 HR patch 크기
            w_hr = round(w_lr * s)

            # HR patch 시작 위치 랜덤 선택
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)

            # HR patch crop
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]

            # HR patch를 LR 크기로 bicubic downsample
            crop_lr = resize_fn(crop_hr, w_lr)

        # data augmentation
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                # H 방향 flip
                if hflip:
                    x = x.flip(-2)

                # W 방향 flip
                if vflip:
                    x = x.flip(-1)

                # H/W transpose
                if dflip:
                    x = x.transpose(-2, -1)

                return x

            # LR/HR에 같은 augmentation 적용
            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # HR patch를 좌표와 값으로 펼침
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # 일부 query만 sampling
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord),
                self.sample_q,
                replace=False
            )

            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        # LIIF cell 크기 생성
        cell = torch.ones_like(hr_coord)

        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,      # bicubic으로 만든 LR input
            'coord': hr_coord,   # HR query 좌표 
            'cell': cell,        # query cell 크기
            'gt': hr_rgb         # HR 정답값
        }


# 다양한 HR size를 사용하기 위한 wrapper
@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        # 원본 dataset
        self.dataset = dataset

        # 최소 HR 크기
        self.size_min = size_min

        # 최대 HR 크기
        # 없으면 size_min으로 고정
        if size_max is None:
            size_max = size_min

        self.size_max = size_max

        # augmentation 여부
        self.augment = augment

        # GT를 최종적으로 특정 크기로 resize할지
        self.gt_resize = gt_resize

        # query sampling 개수
        self.sample_q = sample_q

    def __len__(self):
        # 원본 dataset 길이
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        # dataset index에 따라 0~1 사이 값 생성
        p = idx / (len(self.dataset) - 1)

        # index가 커질수록 HR 크기를 size_min에서 size_max까지 증가시킴
        w_hr = round(
            self.size_min +
            (self.size_max - self.size_min) * p
        )

        # HR을 해당 크기로 resize
        img_hr = resize_fn(img_hr, w_hr)

        # augmentation
        if self.augment:
            # 50% 확률로 W 방향 flip
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        # gt_resize가 있으면 HR을 다시 지정 크기로 resize
        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        # HR을 좌표와 값으로 펼침
        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        # 일부 query만 sampling
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord),
                self.sample_q,
                replace=False
            )

            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        # LIIF cell 크기 생성
        cell = torch.ones_like(hr_coord)

        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,      # LR input
            'coord': hr_coord,  # HR query 좌표
            'cell': cell,       # query cell 크기
            'gt': hr_rgb        # HR 정답값
        }
