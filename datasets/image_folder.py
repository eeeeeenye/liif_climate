import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import pixel_to_liif_coord, latlon_to_pixel_idx
import xarray as xr
import torch.nn.functional as F

from datasets import register
from .obs_folder import ObsFolder

@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path)) 
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('L')))
                # print(self.files)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('L'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x

####################
# 실측 데이터 전처리#
####################
@register('obs-folder')
class ObsInMemory_original(Dataset):
    def __init__(self, root_path=None, meta_path=None, hr_size=100,
                 flipped=False, normalize=None, first_k=None,
                 coord_mode='pixel_center'):
        """
        Args
        ----
        root_path : str
            관측 raw 텍스트 파일 경로
        meta_path : str
            {"lat_range":[lat_max,lat_min], "lon_range":[lon_max,lon_min]} JSON 경로
        hr_size   : int
            HR 한 변 크기 (예: 60)
        flipped   : bool
            전처리에서 상하 flip이 적용되었다면 True (좌표계 정합)
        normalize : None | ('subdiv', sub, div) | ('minmax',)
            gt 정규화 규칙
        first_k   : Optional[int]
            앞에서 K개 블록만 파싱(디버그)
        coord_mode: 'pixel_center' | 'norm_linear'
            - 'pixel_center': lat/lon -> (i,j) -> 픽셀센터 기반 [-1,1]
            - 'norm_linear' : lat/lon -> 바로 [-1,1] 선형 매핑
        """
        assert root_path is not None and meta_path is not None
        assert coord_mode in ('pixel_center', 'norm_linear')

        self.hr_h = int(hr_size)
        self.hr_w = int(hr_size)
        self.flipped = bool(flipped)
        self.normalize = normalize
        self.coord_mode = coord_mode
        self.MISSING = -2.0

        # meta (bbox)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # 이 부분 2026-01-05 수정
        self.lat_max = float(meta['lat_range'][0])
        self.lat_min = float(meta['lat_range'][1])
        self.lon_max = float(meta['lon_range'][1])
        self.lon_min = float(meta['lon_range'][0])

        # self.value_min    = float(meta['min_val'])
        # self.value_max    = float(meta['max_val'])

        # raw parse
        with open(root_path, 'r') as f:
            raw_text = f.read()
        blocks = raw_text.split('/\\')
        if first_k is not None:
            blocks = blocks[:first_k]

        self.data = {}  # date -> list[(lat, lon, val)]
        vmin, vmax = np.inf, -np.inf

        for blk in blocks:
            lines = [ln.strip() for ln in blk.strip().split('\n') if ln.strip()]
            if not lines:
                continue
            # 헤더 1줄 스킵 가정
            for line in lines[1:]:
                cols = line.split(',')
                if len(cols) < 5:
                    continue
                date = cols[1].strip()
                try:
                    lat = float(cols[2]); lon = float(cols[3])
                except Exception:
                    continue

                raw_v = cols[4].strip()
                if raw_v in ['', 'nan', 'NaN', 'None', None]:
                    val = self.MISSING              # 결측은 NaN으로 유지
                else:
                    try:
                        val = float(raw_v)
                    except Exception:
                        val = np.nan

                self.data.setdefault(date, []).append((lat, lon, val))
                if np.isfinite(val) and val != self.MISSING:
                    vmin = min(vmin, val)
                    vmax = max(vmax, val)

        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = 1.0
        self.value_min = float(vmin)
        self.value_max = float(vmax)
        print('image_folder/ObsInMemory ::: ',self.value_min, self.value_max)

        # in-memory 캐싱
        self.keys = sorted(self.data.keys())
        self.date2idx = {d: i for i, d in enumerate(self.keys)}
        self.obs_mem = []

        for date in self.keys:
            arr = np.array(self.data[date], dtype=np.float64)  # (N,3)
            if arr.size == 0:
                self.obs_mem.append({
                    'coord': torch.zeros(0, 2, dtype=torch.float32),
                    'gt':    torch.zeros(0,     dtype=torch.float32),
                    'mask':  torch.zeros(0,     dtype=torch.bool),
                    'cell':  torch.zeros(0, 2,  dtype=torch.float32),
                    'date':  date
                })
                continue

            lat = arr[:, 0]; lon = arr[:, 1]; val = arr[:, 2]

            # 좌표 생성
            if self.coord_mode == 'norm_linear':
                rx = (lon - self.lon_min) / max(self.lon_max - self.lon_min, 1e-12)
                ry = (lat - self.lat_min) / max(self.lat_max - self.lat_min, 1e-12)
                x = 2.0 * rx - 1.0
                y = 2.0 * ry - 1.0
                if self.flipped:
                    y = -y
            else:  # 'pixel_center'
                i, j = latlon_to_pixel_idx(
                    lat, lon,
                    self.lat_min, self.lat_max,
                    self.lon_min, self.lon_max,
                    self.hr_h, self.hr_w,
                    flipped=self.flipped
                )
                x, y = pixel_to_liif_coord(i, j, self.hr_h, self.hr_w)

            coord = torch.tensor(np.stack([x, y], axis=-1), dtype=torch.float32)  # (N,2)

            # 값 / 마스크 / 정규화
            v_t  = torch.tensor(val, dtype=torch.float32)   # (N,)
            nan_mask = ~torch.isfinite(v_t)
            v_t[nan_mask] = -2.0

            # -2을 결측으로 간주하는 마스크 (유효값만 True)
            mask = (v_t != -2.0)
            
            if self.normalize is None:
                gt = v_t.clone()
                if mask.any():
                    denom = max(self.value_max - self.value_min, 1e-12)
                    gt[mask] = (gt[mask] - self.value_min) / denom
                    gt[mask].clamp_(0.0, 1.0)                           
                gt[~mask] = self.MISSING
            else:
                mode = self.normalize[0]
                if mode == 'subdiv':
                    sub, div = float(self.normalize[1]), float(self.normalize[2])
                    if mask.any():
                        gt[mask] = (gt[mask] - sub) / max(div, 1e-12)
                    gt[~mask] = self.MISSING
                elif mode == 'minmax':
                    denom = max(self.value_max - self.value_min, 1e-12)
                    if mask.any():
                        gt[mask] = (gt[mask] - self.value_min) / denom
                        gt[mask].clamp_(0.0, 1.)
                    gt[~mask] = self.MISSING
                else:
                    raise ValueError(f'unknown normalize: {self.normalize}')

            # LIIF cell
            cell = torch.zeros_like(coord)
            cell[:, 0] = 2.0 / max(self.hr_w, 1)
            cell[:, 1] = 2.0 / max(self.hr_h, 1)

            # 여기에 obs 면적 크기 정보를 더 넣으면 좋을 것 같다 .. ! 
            # print(gt)
            self.obs_mem.append({
                'coord': coord,   # (N,2) in [-1,1]
                'gt':    gt,      # (N,)  (float, 보통 0~1)
                'mask':  mask,    # (N,)  bool
                'cell':  cell,    # (N,2)
                'date':  date
            })
    def __len__(self):
        return len(self.obs_mem)

    def __getitem__(self, idx):
        # int | str(date) 둘 다 허용
        if isinstance(idx, str):
            if idx not in self.date2idx:
                raise KeyError(f"[ObsInMemory] idx='{idx}' is not a valid date key.")
            idx = self.date2idx[idx]
        return self.obs_mem[int(idx)]

@register('obs-folder')
class ObsInMemory(Dataset):
    def __init__(self, root_path=None, meta_path=None, hr_size=100,
                 flipped=False, normalize=None, first_k=None,
                 coord_mode='pixel_center'):
        """
        Args
        ----
        root_path : str
            관측 raw 텍스트 파일 경로
        meta_path : str
            {"lat_range":[lat_max,lat_min], "lon_range":[lon_max,lon_min]} JSON 경로
        hr_size   : int
            HR 한 변 크기 (예: 60)
        flipped   : bool
            전처리에서 상하 flip이 적용되었다면 True (좌표계 정합)
        normalize : None | ('subdiv', sub, div) | ('minmax',)
            gt 정규화 규칙
        first_k   : Optional[int]
            앞에서 K개 블록만 파싱(디버그)
        coord_mode: 'pixel_center' | 'norm_linear'
            - 'pixel_center': lat/lon -> (i,j) -> 픽셀센터 기반 [-1,1]
            - 'norm_linear' : lat/lon -> 바로 [-1,1] 선형 매핑
        """
        assert root_path is not None and meta_path is not None
        assert coord_mode in ('pixel_center', 'norm_linear')

        self.hr_h = int(hr_size)
        self.hr_w = int(hr_size)
        self.flipped = bool(flipped)
        self.normalize = normalize
        self.coord_mode = coord_mode
        self.MISSING = -2.0

        # meta (bbox)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # 이 부분 2026-01-05 수정
        self.lat_max = float(meta['lat_range'][0])
        self.lat_min = float(meta['lat_range'][1])
        self.lon_max = float(meta['lon_range'][1])
        self.lon_min = float(meta['lon_range'][0])

        self.value_min    = float(meta['min_val'])
        self.value_max    = float(meta['max_val'])

        # raw parse
        with open(root_path, 'r') as f:
            raw_text = f.read()
        blocks = raw_text.split('/\\')
        if first_k is not None:
            blocks = blocks[:first_k]

        self.data = {}  # date -> list[(lat, lon, val)]

        for blk in blocks:
            lines = [ln.strip() for ln in blk.strip().split('\n') if ln.strip()]
            if not lines:
                continue
            # 헤더 1줄 스킵 가정
            for line in lines[1:]:
                cols = line.split(',')
                if len(cols) < 5:
                    continue
                date = cols[1].strip()
                try:
                    lat = float(cols[2]); lon = float(cols[3])
                except Exception:
                    continue

                raw_v = cols[4].strip()
                if raw_v in ['', 'nan', 'NaN', 'None', None]:
                    val = self.MISSING              # 결측은 NaN으로 유지
                else:
                    try:
                        val = float(raw_v)
                    except Exception:
                        val = np.nan

                self.data.setdefault(date, []).append((lat, lon, val))

        # in-memory 캐싱
        self.keys = sorted(self.data.keys())
        self.date2idx = {d: i for i, d in enumerate(self.keys)}
        self.obs_mem = []

        for date in self.keys:
            arr = np.array(self.data[date], dtype=np.float64)  # (N,3)
            if arr.size == 0:
                self.obs_mem.append({
                    'coord': torch.zeros(0, 2, dtype=torch.float32),
                    'gt':    torch.zeros(0,     dtype=torch.float32),
                    'mask':  torch.zeros(0,     dtype=torch.bool),
                    'cell':  torch.zeros(0, 2,  dtype=torch.float32),
                    'date':  date
                })
                continue

            lat = arr[:, 0]; lon = arr[:, 1]; val = arr[:, 2]

            # 좌표 생성
            if self.coord_mode == 'norm_linear':
                rx = (lon - self.lon_min) / max(self.lon_max - self.lon_min, 1e-12)
                ry = (lat - self.lat_min) / max(self.lat_max - self.lat_min, 1e-12)
                x = 2.0 * rx - 1.0
                y = 2.0 * ry - 1.0
                if self.flipped:
                    y = -y
            else:  # 'pixel_center'
                i, j = latlon_to_pixel_idx(
                    lat, lon,
                    self.lat_min, self.lat_max,
                    self.lon_min, self.lon_max,
                    self.hr_h, self.hr_w,
                    flipped=self.flipped
                )
                x, y = pixel_to_liif_coord(i, j, self.hr_h, self.hr_w)

            coord = torch.tensor(np.stack([x, y], axis=-1), dtype=torch.float32)  # (N,2)

            # 값 / 마스크 / 정규화
            v_t  = torch.tensor(val, dtype=torch.float32)   # (N,)
            nan_mask = ~torch.isfinite(v_t)
            v_t[nan_mask] = -2.0

            # -2을 결측으로 간주하는 마스크 (유효값만 True)
            mask = (v_t != -2.0)
            
            if self.normalize is None:
                gt = v_t.clone()
                if mask.any():
                    denom = max(self.value_max - self.value_min, 1e-12)
                    gt[mask] = (gt[mask] - self.value_min) / denom
                    gt[mask].clamp_(0.0, 1.7)                           
                gt[~mask] = self.MISSING
            else:
                mode = self.normalize[0]
                if mode == 'subdiv':
                    sub, div = float(self.normalize[1]), float(self.normalize[2])
                    if mask.any():
                        gt[mask] = (gt[mask] - sub) / max(div, 1e-12)
                    gt[~mask] = self.MISSING
                elif mode == 'minmax':
                    denom = max(self.value_max - self.value_min, 1e-12)
                    if mask.any():
                        gt[mask] = (gt[mask] - self.value_min) / denom
                        gt[mask].clamp_(0.0, 1.7)
                    gt[~mask] = self.MISSING
                else:
                    raise ValueError(f'unknown normalize: {self.normalize}')

            # LIIF cell
            cell = torch.zeros_like(coord)
            cell[:, 0] = 2.0 / max(self.hr_w, 1)
            cell[:, 1] = 2.0 / max(self.hr_h, 1)

            # area = torch.full_like(gt, 500)

            self.obs_mem.append({
                'coord': coord,   # (N,2) in [-1,1]
                'gt':    gt,      # (N,)  (float, 보통 0~1)
                'mask':  mask,    # (N,)  bool
                'cell':  cell,    # (N,2)
                'date':  date,
                # 'area': area
            })

    def __len__(self):
        return len(self.obs_mem)

    def __getitem__(self, idx):
        # int | str(date) 둘 다 허용
        if isinstance(idx, str):
            if idx not in self.date2idx:
                raise KeyError(f"[ObsInMemory] idx='{idx}' is not a valid date key.")
            idx = self.date2idx[idx]
        return self.obs_mem[int(idx)]

# topo(이미지 하나) 데이터를 가져와서 처리하고 배치마다 사용
class TopoInMemory(Dataset):
    """
    정적 지형(고도) NetCDF를 한번만 로드하여 (1, C, H, W) 텐서로 보관.
    __getitem__은 항상 동일 텐서를 반환 (길이=repeat_to 또는 1).

    채널 구성:
        - 기본: elevation(1채널, 표준화+tanh)
        - 선태기 slope_x, slope_y, relief, avg_pool 여러 스케일 (면적 평균 변수에 유용)
    """
    def __init__(self, path_1=None, lat_range=None, lon_range=None, normalize='standard_tanh', add_slope=False, add_relief=False, avg_pool_ks=(3, 7), dtype=torch.float32, repeat_to=1):
        super().__init__()
        self.repeat_to = int(repeat_to)

        # 읽기
        ds = xr.open_dataset(path_1)['z'].sel(
            lat=slice(lat_range[0], lat_range[1]),
            lon=slice(lon_range[0], lon_range[1])
        ).fillna(0)

        z = ds.values.astype(np.float32)

        if ds['lat'].values[0] > ds['lat'].values[-1]:
            z = np.flipud(z).copy()

        # 정규화
        if normalize == 'standard_tanh':
            mean, std = float(z.mean()) , float(z.std() + 1e-6)
            elev = np.tanh((z-mean)/std)
        elif normalize == 'minmax':
            lo, hi = float(z.min()), float(z.max())
            if hi == lo:
                elev = np.zeros_like(z, dtype=np.float32)
            else:
                elev = (z - lo) / (hi - lo)
        else:
            raise ValueError(f"unknown normalize: {normalize}")

        # torch tensor 변환
        level = torch.from_numpy(elev)[None, None, :, :].to(dtype=dtype)

        # 파생 특징 추가
        chans = [level]
        if add_slope or add_relief or (avg_pool_ks and len(avg_pool_ks) > 0):
            base = level

        if add_slope:
            # Sobel-like 커널
            kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=dtype).view(1,1,3,3)
            ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=dtype).view(1,1,3,3)
            slope_x = F.conv2d(base, kx, padding=1)
            slope_y = F.conv2d(base, ky, padding=1)
            chans += [slope_x, slope_y]

        if add_relief:
            # 국소 릴리프: (max - min) with window=11
            k = 11
            pad = k//2
            pool_max = F.max_pool2d(base, kernel_size=k, stride=1, padding=pad)
            pool_min = -F.max_pool2d(-base, kernel_size=k, stride=1, padding=pad)
            relief = pool_max - pool_min
            chans += [relief]

        if avg_pool_ks and len(avg_pool_ks) > 0:
            for k in avg_pool_ks: 
                pad = k//2
                avgk = F.avg_pool2d(base, kernel_size=k, stride=1, padding=pad)
                chans += [avgk]

        level_stack = torch.cat(chans, dim=1)  # (1, C, H, W)
        self.level = level_stack.squeeze().contiguous()  # 고정 텐서
        self.extra_in_dim = self.level.shape[1]  # imnet 입력 확장 차원 C

        # 보조: 위경도 값 보관(필요 시 참조)
        self.lat = ds['lat'].values
        self.lon = ds['lon'].values

    def __len__(self):
        return int(self.repeat_to)
    
    def __getitem__(self, idx):
        return self.level


# level + observation 추가 입력
# root_1 : 재분석장, root_2: 재분석장, root_3: level 이미지, obs_root_3: observation
@register('paired-image-folders')
class PairedImageFolders(Dataset):
    """
    root_path_1: 이미지 폴더 A
    root_path_2: 이미지 폴더 B
    topo_nc_path: 지형 NetCDF(정적 1장)  ← 새로 추가
    obs_path_3  : 관측 텍스트 (옵션)
    """
    def __init__(self,
                 root_path_1, root_path_2,
                 topo_nc_path=None, topo_kwargs=None,
                 obs_path_3=None, meta_path=None,
                 **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

        self.topo_ds = None
        if topo_nc_path is not None:
            topo_kwargs = topo_kwargs or {}
            self.topo_ds = TopoInMemory(
                path_1=topo_nc_path,
                repeat_to=len(self.dataset_1),  # 길이 맞춤
                lon_range=(124,134),
                lat_range=(33,43),
                **topo_kwargs
            )

        # 관측(옵션)
        if obs_path_3 is not None:
            self.dataset_3 = ObsInMemory(root_path=obs_path_3,
                                         meta_path=meta_path,
                                         flipped=True)
        else:
            self.dataset_3 = None

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        a = self.dataset_1[idx]
        b = self.dataset_2[idx]
        topo = self.topo_ds[idx] if self.topo_ds is not None else None

        if self.dataset_3 is not None:
            obs = self.dataset_3[idx]
            return a, b, obs, topo
        else:
            return a, b, topo