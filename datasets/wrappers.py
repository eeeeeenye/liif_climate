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


from datasets import register
from utils import to_pixel_samples


# level + observation 
@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):
    # LR-HR paired dataset을 LIIF 학습용 형태로 감싸는 wrapper
    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None, obs_state=False):
        self.dataset = dataset        # 원본 dataset
        self.inp_size = inp_size      # LR crop 크기. None이면 전체 사용
        self.augment = augment        # flip augmentation 여부
        self.sample_q = sample_q      # HR pixel 중 몇 개 query point만 뽑을지
        self.obs_state = obs_state    # 관측값 obs를 같이 쓸지 여부

    def __len__(self):
        # dataset 전체 길이 반환
        return len(self.dataset)

    def __getitem__(self, idx):
        # idx번째 샘플을 가져옴
        if self.obs_state:
            # observation을 쓰는 경우
            img_lr, img_hr, obs_, level = self.dataset[idx]
        else:
            # observation을 안 쓰는 경우
            img_lr, img_hr, level = self.dataset[idx]
            obs_ = None

        # HR/LR scale factor 계산
        # 예: HR height 128, LR height 32이면 s=4
        s = img_hr.shape[-2] // img_lr.shape[-2]

        if self.inp_size is None:
            # crop 없이 전체 LR/HR 사용
            h_lr, w_lr = img_lr.shape[-2:]

            # HR 크기를 LR 크기 * scale에 맞춰 자름
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]

            # crop 결과는 전체 이미지
            crop_lr, crop_hr = img_lr, img_hr

            # level도 그대로 사용
            crop_level = level
        else:
            # LR에서 inp_size 크기만큼 random crop
            w_lr = self.inp_size

            # LR crop 시작 위치 랜덤 선택
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)

            # LR crop 추출
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]

            # HR crop 크기 = LR crop 크기 * scale
            w_hr = w_lr * s

            # LR crop 시작점을 HR 좌표로 변환
            x1, y1 = x0 * s, y0 * s

            # 대응되는 HR crop 추출
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

            # 주의: 원 코드에서는 crop_level이 여기서 정의되지 않음
            # return에서 crop_level을 사용하므로 inp_size가 None이 아니면 에러 가능
            # 필요하면 crop_level = level 또는 level도 crop해야 함

        if self.augment:
            # augmentation 여부를 랜덤으로 결정
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                # x가 None이면 그대로 반환
                if x is None:
                    return None

                # -2축 flip: 보통 H 방향
                if hflip:
                    x = x.flip(-2)

                # -1축 flip: 보통 W 방향
                if vflip:
                    x = x.flip(-1)

                # H/W transpose
                if dflip:
                    x = x.transpose(-2, -1)

                return x

            # LR과 HR에 같은 augmentation 적용
            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # HR 이미지를 좌표와 값으로 펼침
        # crop_hr: (C,H,W)
        # hr_coord: (H*W,2)
        # hr_rgb: (H*W,C)
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # observation 개수 초기값
        obs_coord_len = 0
       
        if self.sample_q is not None:
            # 전체 HR pixel 수
            n = len(hr_coord)

            # sample_q가 전체 개수보다 크면 전체 개수까지만 사용
            q = min(self.sample_q, n)

            # q개 좌표를 중복 없이 랜덤 선택
            sample_idx = np.random.choice(n, q, replace=False)

            # 좌표와 gt를 같은 인덱스로 샘플링
            hr_coord = hr_coord[sample_idx]
            hr_rgb = hr_rgb[sample_idx]

        if obs_ is not None:
            # observation 좌표와 값 가져오기
            obs_coord = obs_['coord']
            obs_val = obs_['gt']

            # obs_val이 (N,)이면 (N,1)로 변경
            if obs_val.ndim == 1:
                obs_val = obs_val.unsqueeze(-1)

            # -2.0을 결측값으로 보고 제거
            mask = obs_val.squeeze(-1) != -2.0
            obs_coord = obs_coord[mask]
            obs_val = obs_val[mask]

            # 유효 observation 개수
            obs_coord_len, obs_val_len = obs_coord.shape[0], obs_val.shape[0]

            # observation 값의 channel 수를 hr_rgb channel 수와 맞춤
            obs_val = obs_val.expand(-1, hr_rgb.shape[1])

            # 기존 HR 샘플 일부를 빼고 observation 좌표를 뒤에 붙임
            hr_coord = torch.cat([hr_coord[obs_coord_len:,:], obs_coord], dim=0)

            # 값도 동일하게 observation 값을 뒤에 붙임
            hr_rgb = torch.cat([hr_rgb[obs_val_len:,:], obs_val])

            # observation 여부를 표시하는 mask
            # 0: 일반 HR pixel
            # 1: observation point
            is_obs = torch.cat([
                torch.zeros(hr_coord.shape[0] - obs_coord_len, 1),
                torch.ones(obs_coord_len, 1)
            ], dim=0)
        
        # 각 query point의 cell 크기 생성
        cell = torch.ones_like(hr_coord)

        # LIIF 좌표계가 [-1,1]이므로 전체 길이 2
        # height 방향 cell 크기 = 2 / H
        cell[:, 0] *= 2 / crop_hr.shape[-2]

        # width 방향 cell 크기 = 2 / W
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        # LIIF 모델 학습에 필요한 dictionary 반환
        return {
            'inp': crop_lr,        # LR 입력
            'coord': hr_coord,     # query 좌표
            'cell': cell,          # query cell 크기
            'gt': hr_rgb,          # 정답 HR 값
            'level' : crop_level,  # 추가 조건 정보
            # 'is_obs' : is_obs
            # 'area' : area_norm
        }


def resize_fn(img, size):
    # tensor image를 PIL로 바꾸고 bicubic resize 후 다시 tensor로 변환
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):
    # HR 이미지에서 LR 이미지를 downsample해서 만드는 wrapper

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset      # 원본 dataset
        self.inp_size = inp_size    # LR crop 크기
        self.scale_min = scale_min  # 최소 scale factor

        # scale_max가 없으면 scale_min과 같게 설정
        if scale_max is None:
            scale_max = scale_min

        self.scale_max = scale_max  # 최대 scale factor
        self.augment = augment      # augmentation 여부
        self.sample_q = sample_q    # query sample 개수

    def __len__(self):
        # dataset 길이 반환
        return len(self.dataset)

    def __getitem__(self, idx):
        # 원본 dataset에서 이미지와 topo 가져오기
        img, topo = self.dataset[idx]

        # scale factor를 scale_min~scale_max 사이에서 랜덤 선택
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            # 전체 이미지를 사용할 경우 LR 크기 계산
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)

            # LR 크기에 대응되는 HR 크기 계산
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)

            # HR 이미지와 topo를 대응 크기로 자름
            img = img[:, :h_hr, :w_hr]
            topo = topo[:, :h_hr, :w_hr]

            # HR 이미지를 bicubic downsample해서 LR 생성
            img_down = resize_fn(img, (h_lr, w_lr))

            # crop 결과 저장
            crop_lr = img_down
            crop_hr = img
            crop_topo = topo

        else:
            # LR crop 크기
            w_lr = self.inp_size

            # HR crop 크기 = LR crop 크기 * scale
            w_hr = round(w_lr * s)

            # HR 이미지에서 crop 시작 위치 랜덤 선택
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)

            # HR crop 추출
            crop_hr = img[:, x0:x0 + w_hr, y0:y0 + w_hr]

            # topo도 같은 위치에서 crop
            crop_topo = topo[:, x0:x0 + w_hr, y0:y0 + w_hr]

            # HR crop을 LR 크기로 downsample
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            # augmentation 여부 랜덤 결정
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

            # LR, HR, topo 모두 같은 augmentation 적용
            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            crop_topo = augment(crop_topo)

        # HR crop을 query coordinate와 gt value로 펼침
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # topo도 같은 pixel 순서로 펼침
        # 좌표는 이미 있으므로 버림
        _, hr_topo = to_pixel_samples(crop_topo.contiguous())

        if self.sample_q is not None:
            # query point 일부만 랜덤 선택
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False
            )

            # coord, gt, topo를 같은 인덱스로 샘플링
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            hr_topo = hr_topo[sample_lst]

        # cell size 생성
        cell = torch.ones_like(hr_coord)

        # [-1,1] 좌표계에서 height 방향 cell 크기
        cell[:, 0] *= 2 / crop_hr.shape[-2]

        # [-1,1] 좌표계에서 width 방향 cell 크기
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        # LIIF 학습용 dictionary 반환
        return {
            'inp': crop_lr,      # downsampled LR 입력
            'coord': hr_coord,   # HR query 좌표
            'cell': cell,        # cell 크기
            'gt': hr_rgb,        # HR 정답값
            'level': hr_topo     # topo를 추가 조건으로 사용
        }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):
    # HR 크기를 sample index에 따라 다양하게 바꾸는 wrapper

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset      # 원본 dataset
        self.size_min = size_min    # 최소 HR size

        # size_max가 없으면 size_min과 같게 설정
        if size_max is None:
            size_max = size_min

        self.size_max = size_max    # 최대 HR size
        self.augment = augment      # augmentation 여부
        self.gt_resize = gt_resize  # GT resize 크기
        self.sample_q = sample_q    # query sample 개수

    def __len__(self):
        # dataset 길이 반환
        return len(self.dataset)

    def __getitem__(self, idx):
        # 디버깅용 출력으로 보임
        # 주의: self.dataset[idx]가 tuple이면 .shape가 없어서 에러 가능
        print(self.dataset[idx].shape[1:])

        # LR, HR 이미지 가져오기
        img_lr, img_hr = self.dataset[idx]

        # idx를 0~1 사이 비율로 변환
        p = idx / (len(self.dataset) - 1)

        # idx에 따라 HR 크기를 size_min~size_max 사이에서 선형적으로 결정
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)

        # HR 이미지를 w_hr 크기로 resize
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            # 50% 확률로 W 방향 flip
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            # gt_resize가 지정되면 HR을 다시 해당 크기로 resize
            img_hr = resize_fn(img_hr, self.gt_resize)

        # HR 이미지를 query coordinate와 gt value로 펼침
        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            # query point 일부만 랜덤 선택
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)

            # coord와 gt를 같은 인덱스로 샘플링
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        # cell size 생성
        cell = torch.ones_like(hr_coord)

        # [-1,1] 좌표계에서 height 방향 cell 크기
        cell[:, 0] *= 2 / img_hr.shape[-2]

        # [-1,1] 좌표계에서 width 방향 cell 크기
        cell[:, 1] *= 2 / img_hr.shape[-1]

        # LIIF 학습용 dictionary 반환
        return {
            'inp': img_lr,       # LR 입력
            'coord': hr_coord,   # query 좌표
            'cell': cell,        # cell 크기
            'gt': hr_rgb         # 정답 HR 값
        }
