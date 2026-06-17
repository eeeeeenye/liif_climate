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
    def __init__(self,
                 root_path,
                 topo_nc_path=None,
                 topo_kwargs=None,
                 split_file=None,
                 split_key=None,
                 first_k=None,
                 repeat=1,
                 cache='none'):
        
        """
        Args
        ----
        root_path : str
            이미지 파일들이 저장된 디렉토리 경로
        topo_nc_path : str
            지형(고도) 정보가 저장된 NetCDF(.nc) 파일 경로
        topo_kwargs : dict, optional
            TopoInMemory 생성자에 전달할 추가 옵션
        split_file : str, optional
            데이터 분할(train/val/test 등)을 정의한 JSON 파일 경로
        split_key : str, optional
            split_file 내부에서 사용할 항목의 키 이름
            (예: 'train', 'val', 'test')
        first_k : Optional[int]
            데이터셋의 앞에서 K개 샘플만 사용
            (디버깅 또는 빠른 실험용)
        repeat : int, default=1
            데이터셋을 반복하여 길이를 확장하는 배수
        cache : str | bool
            이미지 데이터 캐싱 방식
            (예: 메모리 저장, 파일 저장, 비활성화 등)
        """

        # 데이터셋 반복 옵션 x, 
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

        # 파일 불러오기
        for filename in filenames:
            file = os.path.join(root_path, filename)

            # 캐시 사용방법 선택
            if cache == 'none':
                # none 선택시 실행되는 이미지 자체를 디스크에 적재 (리스트 append)
                self.files.append(file)

            # bin 선택 시 이미지 데이터를 텐서로 만든 후 바이너리 파일로 저장
            elif cache == 'bin':

                # 빈파일이 저장될 디렉토리 생성
                bin_root = os.path.join(
                    os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path)
                )
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)

                # 빈파일 생성(확장자: pkl)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl'
                )

                if not os.path.exists(bin_file):
                    # 이미지 -> 텐서화
                    img = Image.open(file).convert('L')
                    img = transforms.ToTensor()(img)
                    # 바이너리 파일에 저장
                    with open(bin_file, 'wb') as f:
                        pickle.dump(img, f)
                    print('dump', bin_file)
                # 바이너리파일 리스트에 적재
                self.files.append(bin_file)
            
            # 이미지를 텐서화하여 리스트에 적재
            elif cache == 'in_memory':
                self.files.append(
                    transforms.ToTensor()(Image.open(file).convert('L'))
                )

        # topo 데이터 처리
        self.topo_ds = None
        if topo_nc_path is not None:
            topo_kwargs = topo_kwargs or {}
            self.topo_ds = TopoInMemory(
                path_1=topo_nc_path,
                repeat_to=len(self.files) * self.repeat,
                lon_range=(124, 134),
                lat_range=(33, 43),
                **topo_kwargs
            )

    # Dataset의 전체 샘플 수를 반환하는 특수 메소드
    # * 특수 메소드 : 파이썬이 특정 상황에서 자동으로 호출하는 특수 메서드
    def __len__(self):
        return len(self.files) * self.repeat

    # Dataset의 특정 인덱스에 해당하는 데이터를 반환하는 특수 메서드
    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        # 특정 캐싱 방식에 따라 다르게 실행
        if self.cache == 'none':
            img = transforms.ToTensor()(Image.open(x).convert('L'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                img = pickle.load(f)

        elif self.cache == 'in_memory':
            img = x

        if self.topo_ds is not None:
            topo = self.topo_ds[idx]
            return img, topo

        return img

####################
# 실측 데이터 전처리#
####################
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

        * Obs 데이터는 dataframe 형태로 구성 (확장자 : csv)
        """

        # assert : 이 조건이 반드시 참이어야 한다고 확인하는 명령어
        # 입력값이 올바른지 확인하는 검증
        assert root_path is not None and meta_path is not None
        assert coord_mode in ('pixel_center', 'norm_linear')

        # class 변수 입력
        self.hr_h = int(hr_size)
        self.hr_w = int(hr_size)
        self.flipped = bool(flipped)
        self.normalize = normalize
        self.coord_mode = coord_mode
        self.MISSING = -2.0

        # meta 파일 open
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # 좌표 정규화를 위한 위경도 범위 추출
        self.lat_max = float(meta['lat_range'][0])
        self.lat_min = float(meta['lat_range'][1])
        self.lon_max = float(meta['lon_range'][1])
        self.lon_min = float(meta['lon_range'][0])

        # 메타데이터에서 위/경도 min/max 값 추출
        self.value_min    = float(meta['min_val'])
        self.value_max    = float(meta['max_val'])

        # raw parse -> line by line
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

            # 헤더 1줄 스킵 가정 -> 첫줄은 인덱스이기 때문
            # 파일 칼럼 구성 : [idx, time, x(lat), y(lon), v]
            for line in lines[1:]:                                          # 날짜, 위도, 경도 추출
                cols = line.split(',')
                if len(cols) < 5:
                    continue
                date = cols[1].strip()                                      # date 칼럼을읽고 앞뒤 공백 제거
                try:
                    lat = float(cols[2]); lon = float(cols[3])              # 좌표 컬럼 추출
                except Exception:
                    continue

                raw_v = cols[4].strip()                                     # 강수값 추출
                if raw_v in ['', 'nan', 'NaN', 'None', None]:       
                    val = self.MISSING                                      # 결측은 NaN으로 유지
                else:
                    try:
                        val = float(raw_v)
                    except Exception:
                        val = np.nan

                # 딕셔너리 안에 날짜별로 데이터를 모아 저장하는 코드
                # 날짜를 기준으로 데이터들이 들어가게 됨
                self.data.setdefault(date, []).append((lat, lon, val))

        # in-memory 캐싱
        self.keys = sorted(self.data.keys())                                # 날짜별로 정렬하여 리스트로 저장
        self.date2idx = {d: i for i, d in enumerate(self.keys)}             # 날짜별 인덱스 번호 부여
        self.obs_mem = []                                                   # 데이터 저장해줄 리스트

        for date in self.keys:                                              # 날짜별 데이터 텐서로 변환하여 memory 저장
            arr = np.array(self.data[date], dtype=np.float64)               # (N,3)
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

            # obs 상대좌표 생성
            if self.coord_mode == 'norm_linear':
                # 위/경도를 0~1 범위로 정규화            
                rx = (lon - self.lon_min) / max(self.lon_max - self.lon_min, 1e-12)
                ry = (lat - self.lat_min) / max(self.lat_max - self.lat_min, 1e-12)

                # 이후 -1 ~1로 선형 변환
                x = 2.0 * rx - 1.0
                y = 2.0 * ry - 1.0
                if self.flipped:
                    y = -y
            else:  # 'pixel_center'
                # LIIF의 좌표 체계와 같이 pixcel center 좌표로 변환
                i, j = latlon_to_pixel_idx(                                         # 좌표를 픽셀 인덱스로 변환
                    lat, lon,       
                    self.lat_min, self.lat_max,
                    self.lon_min, self.lon_max,
                    self.hr_h, self.hr_w,
                    flipped=self.flipped
                )
                x, y = pixel_to_liif_coord(i, j, self.hr_h, self.hr_w)              # 픽셀 인덱스를 픽셀의 중심좌표로 변환
            
            # x와 y좌표를 합쳐서 (N, 2) 형태의 Pytorch 텐서로 만드는 구조 (lon, lat)
            coord = torch.tensor(np.stack([y, x], axis=-1), dtype=torch.float32)    

            # 값 / 마스크 / 정규화
            v_t  = torch.tensor(val, dtype=torch.float32)                           # 값 텐서 생성
            nan_mask = ~torch.isfinite(v_t)                                         # nan인 데이터의 mask 행렬을 만들어줌
            v_t[nan_mask] = -2.0                                                    # nan인 데이터는 -2로 채워줌 (0과 nan을 구별하기 위해서)
            mask = (v_t != -2.0)                                                    # -2을 결측으로 간주하는 마스크 (유효값만 True)
            gt = v_t.clone()          

            if self.normalize is None:                                              # 기본 default                                              
                mode = "minmax"                                            
            else:
                mode = self.normalize[0]

            if mode == 'subdiv':                                                    # subdiv라면 z-score normalization
                sub, div = float(self.normalize[1]), float(self.normalize[2])   
                if mask.any():
                    gt[mask] = (gt[mask] - sub) / max(div, 1e-12)
                    gt[~mask] = self.MISSING
            elif mode == 'minmax':                                                  # min/max normalization
                denom = max(self.value_max - self.value_min, 1e-12)
                if mask.any():
                    gt[mask] = (gt[mask] - self.value_min) / denom
                    gt[mask].clamp_(0.0, 1.7)
            else:
                raise ValueError(f'unknown normalize: {self.normalize}')
            
            gt[~mask] = self.MISSING                                                # missing 데이터는 -2로 채워줌

            # LIIF cell
            cell = torch.zeros_like(coord)                                          # coord와 똑같은 shape, 똑같은 dtype를 가지는 0 텐서 생성
            cell[:, 0] = 2.0 / max(self.hr_h, 1)                                    # x축 좌표, 2/hr_size
            cell[:, 1] = 2.0 / max(self.hr_w, 1)                                    # y축 좌표, 2/he_size

            self.obs_mem.append({
                'coord': coord,   # (N,2) in [-1,1]
                'gt':    gt,      # (N,)  (float, 보통 0~1)
                'mask':  mask,    # (N,)  bool
                'cell':  cell,    # (N,2)
                'date':  date,
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
        """
            Args
        ----
        path_1 : str, optional
            지형(NetCDF) 파일 경로

        lat_range : tuple[float, float], optional
            사용할 위도 범위 (lat_min, lat_max)

        lon_range : tuple[float, float], optional
            사용할 경도 범위 (lon_min, lon_max)

        normalize : str
            지형 데이터 정규화 방식
            - 'standard_tanh' : 표준화 후 tanh 적용
            - 'standard'      : z-score 표준화
            - 'minmax'        : min-max 정규화
            - None            : 정규화 미적용

        add_slope : bool
            지형 경사도(slope) 채널 추가 여부

        add_relief : bool
            지형 기복도(relief) 채널 추가 여부

        avg_pool_ks : tuple[int, ...]
            relief 계산 시 사용할 평균 풀링 커널 크기

        dtype : torch.dtype
            출력 텐서 자료형

        repeat_to : int
            데이터셋 길이를 repeat_to 이상이 되도록 반복
        """
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

        # 파생 특징을 사용할 경우 기준 고도장 저장
        if add_slope or add_relief or (avg_pool_ks and len(avg_pool_ks) > 0):
            base = level

        if add_slope:
            # Sobel 필터를 이용한 x/y 방향 지형 경사도 계산
            kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=dtype).view(1,1,3,3)
            ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=dtype).view(1,1,3,3)

            slope_x = F.conv2d(base, kx, padding=1)
            slope_y = F.conv2d(base, ky, padding=1)

            chans += [slope_x, slope_y]

        if add_relief:
            # 국소 지형 기복도(max elevation - min elevation) 계산
            k = 11
            pad = k//2

            # 주변 영역의 최대 고도
            pool_max = F.max_pool2d(base, kernel_size=k, stride=1, padding=pad)

            # 주변 영역의 최소 고도
            pool_min = -F.max_pool2d(-base, kernel_size=k, stride=1, padding=pad)

            relief = pool_max - pool_min

            chans += [relief]

        if avg_pool_ks and len(avg_pool_ks) > 0:
            # 다중 스케일 평균 지형 정보 생성
            for k in avg_pool_ks:
                pad = k//2

                # k×k 영역의 평균 고도 계산
                avgk = F.avg_pool2d(base, kernel_size=k, stride=1, padding=pad)

                chans += [avgk]

        # 생성된 모든 지형 채널 결합
        level_stack = torch.cat(chans, dim=1)  # (1, C, H, W)

        # (C, H, W) 형태의 고정 지형 텐서 저장
        self.level = level_stack.squeeze().contiguous()

        # 지형 채널 수 (elevation + 파생 특징)
        self.extra_in_dim = self.level.shape[1]

        # 원본 위도/경도 좌표 저장
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
    obs_path_3  : 관측 CSV (옵션)
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
            # print(obs['coord'].shape)
            return a, b, obs, topo
        else:
            return a, b, topo