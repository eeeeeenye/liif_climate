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
@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False,
                 sample_q=None, obs_state=False):
        # 원본 dataset
        self.dataset = dataset

        # LR patch 크기
        # None이면 전체 LR/HR 사용
        self.inp_size = inp_size

        # flip augmentation 사용 여부
        self.augment = augment

        # HR query point 중 몇 개만 sampling할지
        self.sample_q = sample_q

        # 관측소 observation supervision을 사용할지 여부
        self.obs_state = obs_state

    def __len__(self):
        # 원본 dataset 길이 그대로 사용
        return len(self.dataset)

    def __getitem__(self, idx):
        # obs_state=True이면 원본 dataset이
        # LR, HR, observation, level을 반환한다고 가정
        if self.obs_state:
            img_lr, img_hr, obs_, level = self.dataset[idx]

        # obs_state=False이면
        # LR, HR, level만 반환
        else:
            img_lr, img_hr, level = self.dataset[idx]
            obs_ = None

        # HR/LR 해상도 배율 계산
        # LR 10x10, HR 100x100이면 s=10
        s = img_hr.shape[-2] // img_lr.shape[-2]

        # inp_size가 없으면 전체 이미지를 사용
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]

            # HR 크기를 LR*s에 정확히 맞춤
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]

            # crop 없이 전체 사용
            crop_lr, crop_hr = img_lr, img_hr

            # level도 그대로 사용
            crop_level = level

        # inp_size가 있으면 LR에서 랜덤 patch crop
        else:
            # LR patch 크기
            w_lr = self.inp_size

            # LR patch의 시작 위치 랜덤 선택
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)

            # LR patch crop
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]

            # 대응되는 HR patch 크기
            w_hr = w_lr * s

            # HR에서 대응되는 시작 위치
            x1, y1 = x0 * s, y0 * s

            # HR patch crop
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

            # 주의:
            # 이 코드에서는 inp_size가 있을 때 crop_level을 따로 정의하지 않음
            # 아래 return에서 crop_level을 쓰므로 실제 실행 시 에러 가능성이 있음

        # data augmentation
        if self.augment:
            # 각각 50% 확률로 flip 여부 결정
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                # observation처럼 None인 경우는 그대로 반환
                if x is None:
                    return None

                # -2축 flip
                # tensor shape이 C,H,W이면 H 방향 flip
                if hflip:
                    x = x.flip(-2)

                # -1축 flip
                # tensor shape이 C,H,W이면 W 방향 flip
                if vflip:
                    x = x.flip(-1)

                # H,W transpose
                if dflip:
                    x = x.transpose(-2, -1)

                return x

            # LR/HR에 동일한 augmentation 적용
            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # -------------------------------------------------
        # Pixel samples 생성
        # -------------------------------------------------
        # crop_hr: C,H,W
        # hr_coord: H*W,2
        # hr_rgb: H*W,C
        #
        # 즉 HR grid 전체를
        # 좌표 coord와 해당 위치의 값 gt로 펼침
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # sample_q가 지정되어 있으면
        # 전체 HR pixel 중 일부 query만 랜덤 선택
        if self.sample_q is not None:
            n = len(hr_coord)

            # sample_q가 전체 pixel 수보다 크면 n개만 사용
            q = min(self.sample_q, n)

            # 중복 없이 q개 sampling
            sample_idx = np.random.choice(n, q, replace=False)

            # 선택된 좌표와 값만 사용
            hr_coord = hr_coord[sample_idx]
            hr_rgb = hr_rgb[sample_idx]

        # observation supervision을 사용하는 경우
        if obs_ is not None:
            # 관측소 좌표
            obs_coord = obs_['coord']

            # 관측값
            obs_val = obs_['gt']

            # obs_val이 1D이면 마지막 channel 차원 추가
            # 예: N -> N,1
            if obs_val.ndim == 1:
                obs_val = obs_val.unsqueeze(-1)

            # 결측값 제거
            # 여기서는 -2.0을 invalid value로 사용
            valid_mask = obs_val.squeeze(-1) != -2.0

            obs_coord = obs_coord[valid_mask]
            obs_val = obs_val[valid_mask]

            # obs_val channel 수를 hr_rgb channel 수에 맞춤
            obs_val = obs_val.expand(-1, hr_rgb.shape[1])

            # 기존 HR 전체 pixel supervision 대신
            # 관측소 위치와 관측값만 supervision으로 사용
            hr_coord = obs_coord
            hr_rgb = obs_val

            # area 관련 실험 코드였던 것으로 보임
            # obs_area = torch.full_like(hr_rgb, 500)
            # area_full = obs_area / 25000

        # -------------------------------------------------
        # Cell 크기 생성
        # -------------------------------------------------
        # LIIF에서 cell은 query point 하나가 차지하는 상대적 픽셀 크기
        # 좌표계가 [-1,1] 범위라서 전체 길이가 2
        cell = torch.ones_like(hr_coord)

        # height 방향 cell 크기
        cell[:, 0] *= 2 / crop_hr.shape[-2]

        # width 방향 cell 크기
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            # LR input patch
            'inp': crop_lr,

            # HR query 좌표
            'coord': hr_coord,

            # 각 query 좌표의 cell size
            'cell': cell,

            # HR 정답값
            'gt': hr_rgb,

            # level 입력
            'level': crop_level,

            # area 실험용
            # 'area': area_full
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
            # bicubic으로 만든 LR input
            'inp': crop_lr,

            # HR query 좌표
            'coord': hr_coord,

            # query cell 크기
            'cell': cell,

            # HR 정답값
            'gt': hr_rgb
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
        # 디버깅용 출력
        print(self.dataset[idx].shape[1:])

        # 원본 dataset에서 LR/HR pair를 가져온다고 가정
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
            # LR input
            'inp': img_lr,

            # HR query 좌표
            'coord': hr_coord,

            # query cell 크기
            'cell': cell,

            # HR 정답값
            'gt': hr_rgb
        }