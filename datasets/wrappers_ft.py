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
@register('sr-implicit-paired-ft')  # 이 클래스를 datasets registry에 'sr-implicit-paired' 이름으로 등록
class SRImplicitPaired(Dataset):  # PyTorch Dataset을 상속받는 LIIF용 wrapper 클래스

    def __init__(self, dataset, inp_size=None, augment=False,
                 sample_q=None, obs_state=False, return_global_coord=False):
        self.dataset = dataset        # 원본 dataset 저장
        self.inp_size = inp_size      # LR patch 크기, None이면 전체 이미지 사용
        self.augment = augment        # data augmentation 사용 여부
        self.sample_q = sample_q      # HR query point 중 sampling할 개수
        self.obs_state = obs_state    # 관측소 supervision 사용 여부
        # False면 기존 wrapper 출력과 완전히 동일하다. True일 때만 crop
        # 좌표와 별개로 whole-domain 좌표를 residual adapter에 전달한다.
        self.return_global_coord = return_global_coord
        if self.return_global_coord and not self.obs_state:
            raise ValueError('return_global_coord requires obs_state=True')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.obs_state:            # 관측소 자료를 사용하는 경우
            img_lr, img_hr, obs_, level = self.dataset[idx]
            # img_lr : (1, 50, 50)      # LR 입력 이미지
            # img_hr : (1, 100, 100)    # HR 정답 이미지
            # obs_   : 관측소 dict      # {'coord', 'gt'} 포함
            # level  : (1, 100, 100)    # topo / level 정보

            # ---- 관측소 유효값 미리 뽑기 (crop 위치 결정에 필요해서 앞당김) ----
            obs_coord = obs_['coord']          # (N, 2), (y, x) 순서, HR grid 기준 [-1,1]
            obs_val = obs_['gt']               # (N,) 또는 (N,1)
            if obs_val.ndim == 1:
                obs_val = obs_val.unsqueeze(-1)
            valid_mask = obs_val.squeeze(-1) != -2.0   # 결측(-2.0) 제외
            obs_coord = obs_coord[valid_mask].clone()
            obs_val = obs_val[valid_mask]

            # [위경도 정렬 버그 교정] (기존 로직과 동일, crop 여부와 무관하게 먼저 적용)
            # ObsInMemory(latlon_to_pixel_idx)는 위도를 '북 -> row0 (y=-1)'로 매핑하는데
            # IMERG/HSR nc(NetCDFFolder)는 위도 오름차순이라 row0 = 남쪽 -> grid_sample이
            # 위도 대칭점의 feature/topo를 읽어 관측소와 어긋난 채로 adapter가 학습됨.
            # coord = stack([y, x]) 이므로 y(index 0) 부호만 뒤집으면 '남 -> y=-1'로 맞춰짐.
            obs_coord[:, 0] = -obs_coord[:, 0]
        else:                         # 관측소 자료를 사용하지 않는 경우
            img_lr, img_hr, level = self.dataset[idx]
            obs_ = None               # observation 없음

        s = img_hr.shape[-2] // img_lr.shape[-2]
        # scale factor 계산
        # 100 // 50 = 2

        H_lr, W_lr = img_lr.shape[-2:]
        H_hr, W_hr = img_hr.shape[-2:]
        if self.inp_size is None:
            # crop 없이 전체 LR/HR 사용
            h_lr, w_lr = img_lr.shape[-2:]

            # HR 크기를 LR 크기 * scale에 맞춰 자름
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]

            # crop 결과는 전체 이미지
            crop_lr, crop_hr = img_lr, img_hr

            # level도 그대로 사용
            crop_level = level

            if self.obs_state:
                # crop 없음 -> 관측소 좌표도 전체 도메인 기준 그대로 사용
                obs_coord_crop, obs_val_crop = obs_coord, obs_val
                global_coord_crop = obs_coord.clone()
            global_bbox = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=img_lr.dtype)
        else:
            # LR에서 inp_size 크기만큼 crop.
            # obs_state=True면(HSR finetune처럼 관측소가 도메인 전체에 흩어져 있는 경우)
            # 그냥 랜덤 위치를 crop하면 그 window 안에 관측소가 하나도 안 들어올 확률이
            # 높아서(빈 배치만 계속 나옴) 학습 신호가 거의 없어진다. 그래서 관측소 중
            # 하나를 anchor로 잡아 그 지점이 반드시 window 안에 들어오도록 crop 위치를
            # 정하고, 그 window에 같이 들어오는 다른 관측소들도 좌표를 다시 계산해 포함시킴.
            # (관측소가 없는 시각/비 obs_state 모드는 기존처럼 완전 랜덤 위치로 crop.)
            """
                DEM이 ETOPO 기준 [1, 600, 480] 일 경우의 내용

                LR / HR/ DEM을 동일한 자리 영역으로 crop
                LR, HR, DEM은 해상도(픽셀 수)는 다르지만
                같은 위경도 영역을 나타내는 데이터임.,
                따라서 lr에서 Random crop한 위치를 기준으로 HR과 dem에서도 동일한
                공간 영역을 crop 해야함

                이를 위해 LR crop 위치를 전체 LR 이미지에서의 비율로 변환한 뒤,
                그 비율을 HR과 DEM 크기에 적용하여 대응되는 영역을 계산
            """

            # LR 이미지에서 선택된 위치를 inp_size만큼 crop
            # crop lr shape : [C, inp_size, inp_size]
            crop_h_lr = self.inp_size
            crop_w_lr = self.inp_size

            if self.obs_state and obs_coord.shape[0] > 0:
                # 관측소 좌표([-1,1])는 해상도와 무관한 물리 위치이므로, LR 격자에
                # 그대로 대입해서 LR 픽셀 인덱스로 역변환할 수 있다
                # (pixel_to_liif_coord의 역함수: coord -> pixel index).
                k = random.randrange(obs_coord.shape[0])
                ay = float(obs_coord[k, 0]); ax = float(obs_coord[k, 1])
                i_anchor = (ay + 1.0) / 2.0 * (H_lr - 1)
                j_anchor = (ax + 1.0) / 2.0 * (W_lr - 1)
                x0 = int(round(i_anchor - crop_h_lr / 2.0))
                y0 = int(round(j_anchor - crop_w_lr / 2.0))
                x0 = min(max(x0, 0), H_lr - crop_h_lr)
                y0 = min(max(y0, 0), W_lr - crop_w_lr)
            else:
                # 관측소 없음(비 안 온 시각) 또는 obs_state=False -> 기존처럼 랜덤 crop
                x0 = random.randint(0, H_lr - crop_h_lr)
                y0 = random.randint(0, W_lr - crop_w_lr)

            crop_lr = img_lr[
                :,
                x0:x0 + crop_h_lr,
                y0:y0 + crop_w_lr
            ]

            # [y_min, y_max, x_min, x_max] in the whole-domain [-1,1]
            # convention. LTE itself still uses crop-local coordinates.
            global_bbox = torch.tensor([
                2.0 * x0 / (H_lr - 1) - 1.0,
                2.0 * (x0 + crop_h_lr - 1) / (H_lr - 1) - 1.0,
                2.0 * y0 / (W_lr - 1) - 1.0,
                2.0 * (y0 + crop_w_lr - 1) / (W_lr - 1) - 1.0,
            ], dtype=img_lr.dtype)

            # HR crop
            # LR에서 선택한 시작 위치를 HR 좌표계의 시작 위치로 변환
            # 만약 LR : 40, HR : 80이면, lr X=10 -> HR X=20에 대응
            x1_hr = round(x0 / H_lr * H_hr)
            y1_hr = round(y0 / W_lr * W_hr)

            crop_h_hr = round(crop_h_lr / H_lr * H_hr)
            crop_w_hr = round(crop_w_lr / W_lr * W_hr)

            crop_hr = img_hr[
                :,
                x1_hr:x1_hr + crop_h_hr,
                y1_hr:y1_hr + crop_w_hr
            ]

            # topo_nc_path 없는 config(model B 등)면 level이 None -> DEM crop 스킵
            if level is not None:
                H_topo, W_topo = level.shape[-2:]
                x1_topo = round(x0 / H_lr * H_topo)
                y1_topo = round(y0 / W_lr * W_topo)

                crop_h_topo = round(crop_h_lr / H_lr * H_topo)
                crop_w_topo = round(crop_w_lr / W_lr * W_topo)

                # boundary check
                x1_topo = min(x1_topo, H_topo - crop_h_topo)
                y1_topo = min(y1_topo, W_topo - crop_w_topo)

                crop_level = level[
                    ...,
                    x1_topo:x1_topo + crop_h_topo,
                    y1_topo:y1_topo + crop_w_topo
                ]
            else:
                crop_level = None

            if self.obs_state:
                if obs_coord.shape[0] > 0:
                    # 전체 도메인 기준 관측소들의 LR 픽셀 인덱스를 구해서, 이번 crop
                    # window 안에 들어오는 것만 남기고, 좌표를 window 기준으로 재정규화
                    i_all = (obs_coord[:, 0] + 1.0) / 2.0 * (H_lr - 1)
                    j_all = (obs_coord[:, 1] + 1.0) / 2.0 * (W_lr - 1)
                    in_win = (i_all >= x0) & (i_all < x0 + crop_h_lr) & \
                             (j_all >= y0) & (j_all < y0 + crop_w_lr)
                    i_win = i_all[in_win] - x0
                    j_win = j_all[in_win] - y0
                    y_new = (i_win / (crop_h_lr - 1)) * 2.0 - 1.0
                    x_new = (j_win / (crop_w_lr - 1)) * 2.0 - 1.0
                    obs_coord_crop = torch.stack([y_new, x_new], dim=-1)
                    global_coord_crop = obs_coord[in_win].clone()
                    obs_val_crop = obs_val[in_win]
                else:
                    obs_coord_crop, obs_val_crop = obs_coord, obs_val  # 둘 다 빈 (0,2)/(0,1)
                    global_coord_crop = obs_coord.clone()


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

        if self.obs_state:           # 관측소 supervision을 사용하는 경우
            # obs_coord_crop/obs_val_crop은 위 crop 단계에서 이미 유효값 필터링 +
            # 좌표 교정(남/북 부호) + (crop 적용 시) window 기준 재정규화까지 끝난 상태.
            hr_coord = obs_coord_crop
            # hr_coord : (valid_stations_in_window, 2)

            hr_rgb = obs_val_crop.expand(-1, hr_rgb.shape[1])
            # gt channel 수를 hr_rgb channel 수와 맞춤
            # hr_rgb : (valid_stations_in_window, 1)

        cell = torch.ones_like(hr_coord)
        # coord와 같은 shape로 cell 생성
        # obs 사용 X : (sample_q, 2) 또는 (10000, 2)
        # obs 사용 O : (valid_stations, 2)

        if obs_ is not None:
            # ASOS는 격자 평균이 아닌 지점(point) 관측이므로
            # IMERG grid(crop_hr 해상도)보다 훨씬 조밀한 가상 해상도로 취급
            cell[:, 0] *= 2 / 1000
            cell[:, 1] *= 2 / 1000
        else:
            # grid 기반 query는 실제 crop_hr 해상도를 그대로 사용
            cell[:, 0] *= 2 / crop_hr.shape[-2]
            cell[:, 1] *= 2 / crop_hr.shape[-1]

        out = {
            'inp': crop_lr,        # LR input, shape: (1, 50, 50)
            'coord': hr_coord,     # query 좌표, obs 사용 시: (valid_stations, 2)
            'cell': cell,          # query cell 크기, obs 사용 시: (valid_stations, 2)
            'gt': hr_rgb,          # 정답값, obs 사용 시: (valid_stations, 1)
            'level': crop_level,   # topo/level input, shape: (1, 100, 100)
        }
        if self.return_global_coord:
            out['global_coord'] = global_coord_crop
            out['global_bbox'] = global_bbox
        return out

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
