import os                                                   # 파일/디렉토리 경로 처리와 폴더 목록 조회에 사용
import json                                                 # split/meta 설정 파일(JSON)을 읽는 데 사용
from PIL import Image                                        # 이미지 파일을 열고 grayscale로 변환하는 데 사용

import pickle                                               # 텐서화한 이미지를 .pkl 바이너리 파일로 저장/로드하는 데 사용
import imageio                                              # 이미지 입출력 라이브러리 import; 현재 코드에서는 직접 사용되지 않음
import numpy as np                                          # 배열 연산, 좌표/관측값 처리, 정규화 계산에 사용
import torch                                                # PyTorch Tensor 생성 및 연산에 사용
from torch.utils.data import Dataset                        # PyTorch Dataset 클래스를 상속하기 위해 사용
from torchvision import transforms                          # PIL 이미지를 Tensor로 변환하기 위해 사용
from utils import pixel_to_liif_coord, latlon_to_pixel_idx   # 위경도↔픽셀좌표↔LIIF 좌표 변환 함수 import
import xarray as xr                                         # NetCDF 지형 자료를 읽기 위해 사용
import torch.nn.functional as F                             # conv2d, pooling 등 함수형 연산에 사용

from datasets import register                               # dataset 이름을 registry에 등록하기 위한 decorator import
from .obs_folder import ObsFolder                           # 같은 패키지의 ObsFolder import; 현재 코드에서는 직접 사용되지 않음

@register('image-folder')                                   # 이 Dataset 클래스를 'image-folder'라는 이름으로 등록
class ImageFolder(Dataset):                                 # 이미지 폴더를 PyTorch Dataset 형태로 다루는 클래스
    def __init__(self,                                      # Dataset 생성자 정의
                 root_path,                                 # 이미지 파일들이 들어 있는 폴더 경로
                 topo_nc_path=None,                         # 선택적으로 사용할 지형 NetCDF 파일 경로
                 topo_kwargs=None,                          # TopoInMemory에 넘길 추가 인자 딕셔너리
                 split_file=None,                           # train/val/test 분할 정보가 들어 있는 JSON 파일 경로
                 split_key=None,                            # split_file에서 사용할 key 이름
                 first_k=None,                              # 앞에서부터 일부 샘플만 사용할 때의 개수
                 repeat=1,                                  # Dataset 길이를 몇 배 반복할지 지정
                 cache='none'):                             # 이미지 캐싱 방식 지정('none', 'bin', 'in_memory')
        
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

        self.repeat = repeat                                # 같은 이미지 목록을 repeat번 반복해 Dataset 길이를 늘리기 위한 값 저장
        self.cache = cache                                  # __getitem__에서 어떤 방식으로 이미지를 읽을지 결정하기 위해 cache 방식 저장

        if split_file is None:                              # split 파일이 없으면 root_path 안의 모든 파일을 사용
            filenames = sorted(os.listdir(root_path))        # 폴더 안의 파일명을 정렬해서 순서를 고정
        else:                                               # split 파일이 있으면 지정된 split만 사용
            with open(split_file, 'r') as f:                 # JSON split 파일 열기
                filenames = json.load(f)[split_key]         # split_key에 해당하는 파일명 목록만 가져오기

        if first_k is not None:                             # 일부 샘플만 사용할 경우
            filenames = filenames[:first_k]                 # 앞에서부터 first_k개 파일만 남김

        self.files = []                                     # 실제로 __getitem__에서 사용할 파일 경로 또는 Tensor를 저장할 리스트

        for filename in filenames:                          # 사용할 파일명 목록을 하나씩 순회
            file = os.path.join(root_path, filename)         # root_path와 파일명을 합쳐 전체 파일 경로 생성

            if cache == 'none':                             # 캐싱하지 않고 매번 디스크에서 이미지를 읽는 방식
                self.files.append(file)                     # 이미지 파일 경로만 리스트에 저장

            elif cache == 'bin':                            # 이미지를 Tensor로 변환한 뒤 pkl 파일로 캐싱하는 방식

                bin_root = os.path.join(                    # pkl 캐시 파일들을 저장할 폴더 경로 생성
                    os.path.dirname(root_path),              # 원본 이미지 폴더의 상위 폴더
                    '_bin_' + os.path.basename(root_path)    # 원본 폴더명 앞에 _bin_을 붙인 캐시 폴더명
                )
                if not os.path.exists(bin_root):             # 캐시 폴더가 아직 없으면
                    os.mkdir(bin_root)                       # 캐시 폴더 생성
                    print('mkdir', bin_root)

                bin_file = os.path.join(                    # 현재 이미지에 대응되는 pkl 캐시 파일 경로 생성
                    bin_root, filename.split('.')[0] + '.pkl' # 확장자를 제거하고 .pkl 확장자로 저장
                )

                if not os.path.exists(bin_file):             # 해당 이미지의 캐시 파일이 아직 없으면
                    img = Image.open(file).convert('L')      # 이미지를 열고 grayscale 단일 채널로 변환
                    img = transforms.ToTensor()(img)         # PIL 이미지를 [C,H,W] 형태의 float Tensor로 변환
                    with open(bin_file, 'wb') as f:          # pkl 파일을 binary write 모드로 열기
                        pickle.dump(img, f)                  # 변환된 Tensor를 pkl 파일에 저장
                    print('dump', bin_file)
                self.files.append(bin_file)                  # 이후 로드할 수 있도록 pkl 파일 경로를 리스트에 저장
            
            elif cache == 'in_memory':                       # 모든 이미지를 처음부터 메모리에 올려두는 방식
                self.files.append(                           # 변환된 이미지 Tensor를 바로 리스트에 저장
                    transforms.ToTensor()(Image.open(file).convert('L')) # 이미지를 열고 grayscale 변환 후 Tensor화
                )

        self.topo_ds = None                                  # 지형 Dataset은 기본적으로 사용하지 않는 상태로 초기화
        if topo_nc_path is not None:                         # 지형 NetCDF 경로가 주어진 경우
            topo_kwargs = topo_kwargs or {}                  # topo_kwargs가 None이면 빈 dict로 대체
            self.topo_ds = TopoInMemory(                     # 정적 지형 데이터를 메모리에 올리는 Dataset 생성
                path_1=topo_nc_path,                         # NetCDF 파일 경로 전달
                repeat_to=len(self.files) * self.repeat,     # 이미지 Dataset 길이와 맞도록 지형 Dataset 길이 지정
                lon_range=(124, 134),                        # 사용할 경도 범위 지정
                lat_range=(33, 43),                          # 사용할 위도 범위 지정
                **topo_kwargs                                # 추가 지형 처리 옵션 전달
            )

    def __len__(self):                                       # Dataset의 전체 샘플 수를 반환하는 특수 메소드
        return len(self.files) * self.repeat                 # 실제 파일 개수에 repeat 배수를 곱한 길이 반환

    def __getitem__(self, idx):                              # 특정 index에 해당하는 샘플을 반환하는 특수 메소드
        x = self.files[idx % len(self.files)]                # repeat 때문에 idx가 커져도 원본 파일 목록 안에서 순환되도록 처리

        if self.cache == 'none':                             # 캐싱하지 않는 경우
            img = transforms.ToTensor()(Image.open(x).convert('L')) # 파일 경로 x에서 이미지를 읽고 grayscale Tensor로 변환

        elif self.cache == 'bin':                            # pkl 캐시를 사용하는 경우
            with open(x, 'rb') as f:                         # pkl 파일을 binary read 모드로 열기
                img = pickle.load(f)                         # 저장되어 있던 이미지 Tensor 로드

        elif self.cache == 'in_memory':                      # 이미지 Tensor가 이미 메모리에 올라와 있는 경우
            img = x                                          # self.files에 저장된 Tensor를 그대로 사용

        if self.topo_ds is not None:                         # 지형 Dataset을 함께 사용하는 경우
            topo = self.topo_ds[idx]                         # 현재 index에 대응되는 지형 Tensor 반환
            return img, topo                                 # 이미지 Tensor와 지형 Tensor를 함께 반환

        return img                                           # 지형 정보가 없으면 이미지 Tensor만 반환

####################
# 실측 데이터 전처리#
####################
@register('obs-folder')                                      # 이 Dataset 클래스를 'obs-folder'라는 이름으로 등록
class ObsInMemory(Dataset):                                  # 관측 자료를 날짜별로 메모리에 저장하는 Dataset 클래스
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

        assert root_path is not None and meta_path is not None # 관측 파일 경로와 메타 파일 경로가 반드시 있어야 함
        assert coord_mode in ('pixel_center', 'norm_linear')   # coord_mode는 두 가지 방식 중 하나만 허용

        self.hr_h = int(hr_size)                            # 고해상도 격자의 세로 크기 저장
        self.hr_w = int(hr_size)                            # 고해상도 격자의 가로 크기 저장
        self.flipped = bool(flipped)                        # y축 방향 뒤집힘 여부 저장
        self.normalize = normalize                          # 관측값 정규화 방식 저장
        self.coord_mode = coord_mode                        # 위경도를 LIIF 좌표로 변환하는 방식 저장
        self.MISSING = -2.0                                 # 결측값을 표시하기 위한 sentinel 값

        with open(meta_path, 'r') as f:                      # 좌표 범위와 값 범위가 들어 있는 meta JSON 파일 열기
            meta = json.load(f)                             # JSON 내용을 dict로 읽기
        
        self.lat_max = float(meta['lat_range'][0])           # meta에서 최대 위도 값 추출
        self.lat_min = float(meta['lat_range'][1])           # meta에서 최소 위도 값 추출
        self.lon_max = float(meta['lon_range'][1])           # meta에서 최대 경도 값 추출
        self.lon_min = float(meta['lon_range'][0])           # meta에서 최소 경도 값 추출

        self.value_min = float(meta['min_val'])              # min-max 정규화에 사용할 관측값 최소값 저장
        self.value_max = float(meta['max_val'])              # min-max 정규화에 사용할 관측값 최대값 저장

        with open(root_path, 'r') as f:                      # 관측 raw 텍스트 파일 열기
            raw_text = f.read()                              # 파일 전체 내용을 문자열로 읽기
        blocks = raw_text.split('/\\')                      # '/\' 구분자를 기준으로 날짜/시간별 블록 분리
        if first_k is not None:                              # 일부 블록만 사용할 경우
            blocks = blocks[:first_k]                        # 앞에서부터 first_k개 블록만 사용

        self.data = {}                                       # 날짜를 key로 하고 (lat, lon, val) 리스트를 value로 저장할 dict

        for blk in blocks:                                   # 분리된 관측 블록을 하나씩 순회
            lines = [ln.strip() for ln in blk.strip().split('\n') if ln.strip()] # 빈 줄을 제외하고 각 줄 앞뒤 공백 제거
            if not lines:                                    # 블록에 유효한 줄이 없으면
                continue                                     # 다음 블록으로 넘어감

            for line in lines[1:]:                           # 첫 줄은 헤더로 보고 제외한 뒤 데이터 행만 순회
                cols = line.split(',')                       # CSV 형식이므로 쉼표 기준으로 컬럼 분리
                if len(cols) < 5:                            # 필요한 컬럼(idx,time,lat,lon,value)이 부족하면
                    continue                                 # 해당 줄은 건너뜀
                date = cols[1].strip()                       # time/date 컬럼을 읽고 앞뒤 공백 제거
                try:                                         # 좌표 값 변환 중 오류가 날 수 있으므로 예외 처리
                    lat = float(cols[2]); lon = float(cols[3]) # 위도와 경도를 float로 변환
                except Exception:                            # 좌표 변환에 실패하면
                    continue                                 # 해당 줄은 건너뜀

                raw_v = cols[4].strip()                      # 관측값 컬럼을 문자열로 읽고 앞뒤 공백 제거
                if raw_v in ['', 'nan', 'NaN', 'None', None]: # 결측으로 간주할 문자열인지 확인
                    val = self.MISSING                       # 결측값은 -2.0 sentinel로 저장
                else:                                        # 결측 문자열이 아니면
                    try:                                     # 숫자 변환 시도
                        val = float(raw_v)                   # 관측값을 float로 변환
                    except Exception:                        # 숫자 변환에 실패하면
                        val = np.nan                         # 일단 NaN으로 저장하고 이후 mask 처리에서 결측 처리

                self.data.setdefault(date, []).append((lat, lon, val)) # 날짜별 리스트에 (위도, 경도, 관측값) 추가

        self.keys = sorted(self.data.keys())                 # 날짜 key를 정렬해 index 순서를 고정
        self.date2idx = {d: i for i, d in enumerate(self.keys)} # 날짜 문자열로도 접근할 수 있도록 날짜→index dict 생성
        self.obs_mem = []                                    # 날짜별 관측 Tensor dict를 저장할 리스트

        for date in self.keys:                               # 날짜별로 관측값을 Tensor 형태로 변환
            arr = np.array(self.data[date], dtype=np.float64) # 해당 날짜의 (lat, lon, val) 리스트를 (N,3) numpy 배열로 변환
            if arr.size == 0:                                # 해당 날짜에 관측 데이터가 하나도 없는 경우
                self.obs_mem.append({                        # 빈 관측 샘플을 같은 key 구조로 저장
                    'coord': torch.zeros(0, 2, dtype=torch.float32), # 관측 좌표가 없으므로 shape (0,2) Tensor 생성
                    'gt':    torch.zeros(0,     dtype=torch.float32), # 관측값이 없으므로 shape (0,) Tensor 생성
                    'mask':  torch.zeros(0,     dtype=torch.bool),    # 유효값 mask도 빈 bool Tensor 생성
                    'cell':  torch.zeros(0, 2,  dtype=torch.float32), # LIIF cell 정보도 빈 shape (0,2) Tensor 생성
                    'date':  date                                     # 날짜 정보는 유지
                })
                continue                                      # 빈 샘플 처리가 끝났으므로 다음 날짜로 이동

            lat = arr[:, 0]; lon = arr[:, 1]; val = arr[:, 2] # 배열에서 위도, 경도, 관측값 컬럼을 각각 분리

            if self.coord_mode == 'norm_linear':              # 위경도를 직접 [-1,1] 범위로 선형 변환하는 방식
                rx = (lon - self.lon_min) / max(self.lon_max - self.lon_min, 1e-12) # 경도를 0~1 범위로 정규화
                ry = (lat - self.lat_min) / max(self.lat_max - self.lat_min, 1e-12) # 위도를 0~1 범위로 정규화

                x = 2.0 * rx - 1.0                            # 정규화된 경도를 [-1,1] 범위의 x좌표로 변환
                y = 2.0 * ry - 1.0                            # 정규화된 위도를 [-1,1] 범위의 y좌표로 변환
                if self.flipped:                              # 이미지가 상하 반전된 좌표계라면
                    y = -y                                    # y좌표 방향을 반대로 맞춤
            else:                                             # 'pixel_center' 방식: 위경도→픽셀 index→LIIF 좌표 순서로 변환
                i, j = latlon_to_pixel_idx(                   # 위경도를 HR 격자의 픽셀 index로 변환
                    lat, lon,                                 # 관측소 위도/경도 배열 전달
                    self.lat_min, self.lat_max,               # 변환에 사용할 위도 범위 전달
                    self.lon_min, self.lon_max,               # 변환에 사용할 경도 범위 전달
                    self.hr_h, self.hr_w,                     # HR 이미지의 높이와 너비 전달
                    flipped=self.flipped                      # y축 뒤집힘 여부 반영
                )
                x, y = pixel_to_liif_coord(i, j, self.hr_h, self.hr_w) # 픽셀 index를 LIIF의 [-1,1] 픽셀 중심 좌표로 변환
            
            coord = torch.tensor(np.stack([y, x], axis=-1), dtype=torch.float32) # y,x 순서로 합쳐 각 관측점 좌표 Tensor (N,2) 생성

            v_t = torch.tensor(val, dtype=torch.float32)      # 관측값 배열을 float32 Tensor로 변환
            nan_mask = ~torch.isfinite(v_t)                   # NaN/inf처럼 유한하지 않은 값 위치를 True로 표시
            v_t[nan_mask] = self.MISSING                      # 유한하지 않은 값은 결측 sentinel 값(-2.0)으로 대체
            mask = (v_t != self.MISSING)                      # 결측이 아닌 위치만 True인 유효 관측값 mask 생성
            gt = v_t.clone()                                  # 정규화 전 관측값 Tensor를 복사해 gt로 사용

            if self.normalize is None:                        # 정규화 방식이 따로 지정되지 않았으면
                mode = 'minmax'                               # 기본 정규화 방식으로 minmax 사용
            else:                                             # 정규화 방식이 지정되어 있으면
                mode = self.normalize[0]                      # normalize tuple/list의 첫 번째 원소를 mode로 사용

            if mode == 'subdiv':                              # 평균/표준편차 방식의 정규화인 경우
                sub, div = float(self.normalize[1]), float(self.normalize[2]) # 정규화에 사용할 sub와 div 값 추출
                if mask.any():                                # 유효 관측값이 하나라도 있으면
                    gt[mask] = (gt[mask] - sub) / max(div, 1e-12) # 유효값에 대해서만 (값-sub)/div 정규화 수행
                    gt[~mask] = self.MISSING                  # 결측 위치는 sentinel 값으로 유지
            elif mode == 'minmax':                            # min-max 정규화 방식인 경우
                denom = max(self.value_max - self.value_min, 1e-12) # 0 나눗셈 방지를 위해 최소 분모를 보장
                if mask.any():                                # 유효 관측값이 하나라도 있으면
                    gt[mask] = (gt[mask] - self.value_min) / denom # 유효값을 min-max 방식으로 정규화
                    gt[mask].clamp_(0.0, 1.7)                 # 유효값 범위를 0.0~1.7 사이로 제한
            else:                                             # 지원하지 않는 정규화 방식이면
                raise ValueError(f'unknown normalize: {self.normalize}') # 어떤 normalize 값이 문제인지 에러 발생
            
            gt[~mask] = self.MISSING                          # 결측 위치는 정규화 이후에도 -2.0으로 유지

            cell = torch.zeros_like(coord)                     # coord와 같은 shape의 0 Tensor 생성; 각 관측점의 cell 크기 저장용
            cell[:, 0] = 2.0 / max(self.hr_h, 1)               # 첫 번째 좌표축 방향의 cell 크기를 LIIF 좌표계 기준으로 설정
            cell[:, 1] = 2.0 / max(self.hr_w, 1)               # 두 번째 좌표축 방향의 cell 크기를 LIIF 좌표계 기준으로 설정

            self.obs_mem.append({                             # 날짜 하나에 대한 관측 Tensor들을 메모리 리스트에 저장
                'coord': coord,                               # 관측소 위치 좌표 Tensor, shape (N,2), 값 범위는 대체로 [-1,1]
                'gt':    gt,                                  # 관측값 Tensor, shape (N,), 유효값은 정규화되고 결측은 -2.0
                'mask':  mask,                                # 유효 관측값 여부 Tensor, shape (N,), bool
                'cell':  cell,                                # LIIF cell 크기 Tensor, shape (N,2)
                'date':  date,                                # 현재 관측 날짜/시간 key
            })

    def __len__(self):                                        # Dataset에 저장된 날짜별 관측 샘플 수 반환
        return len(self.obs_mem)                              # obs_mem에 저장된 날짜 개수 반환

    def __getitem__(self, idx):                               # index 또는 날짜 문자열로 관측 샘플 하나를 반환
        if isinstance(idx, str):                              # idx가 정수가 아니라 날짜 문자열인 경우
            if idx not in self.date2idx:                      # 해당 날짜가 Dataset에 없으면
                raise KeyError(f"[ObsInMemory] idx='{idx}' is not a valid date key.") # 유효하지 않은 날짜 key 에러 발생
            idx = self.date2idx[idx]                          # 날짜 문자열을 정수 index로 변환
        return self.obs_mem[int(idx)]                         # 정수 index에 해당하는 관측 Tensor dict 반환

class TopoInMemory(Dataset):                                  # 정적 지형 NetCDF를 한 번만 읽어 메모리에 저장하는 Dataset 클래스
    """
    정적 지형(고도) NetCDF를 한번만 로드하여 (C, H, W) 텐서로 보관.
    __getitem__은 항상 동일 텐서를 반환 (길이=repeat_to 또는 1).

    채널 구성:
        - 기본: elevation(1채널, 표준화+tanh 또는 minmax)
        - 선택: slope_x, slope_y, relief
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
            - 'minmax'        : min-max 정규화

        ** slope, relief, avg_pool_ks는 사용안하였습니다***
        add_slope : bool
            지형 경사도(slope) 채널 추가 여부

        add_relief : bool
            지형 기복도(relief) 채널 추가 여부

        avg_pool_ks : tuple[int, ...]
            현재 코드에서는 평균 풀링 채널 생성 부분이 주석 처리되어 있어 실제로 사용되지 않음

        dtype : torch.dtype
            출력 텐서 자료형

        repeat_to : int
            Dataset 길이를 repeat_to로 맞추기 위한 값
        """
        super().__init__()                                    # Dataset 부모 클래스 초기화
        self.repeat_to = int(repeat_to)                       # __len__에서 반환할 Dataset 길이 저장

        ds = xr.open_dataset(path_1)['z'].sel(                # NetCDF에서 z 변수만 선택한 뒤 지정한 위경도 범위로 잘라 읽기
            lat=slice(lat_range[0], lat_range[1]),            # 사용할 위도 범위 적용
            lon=slice(lon_range[0], lon_range[1])             # 사용할 경도 범위 적용
        ).fillna(0)                                           # 지형 자료의 NaN 값을 0으로 대체

        z = ds.values.astype(np.float32)                      # xarray DataArray를 float32 numpy 배열로 변환

        if ds['lat'].values[0] > ds['lat'].values[-1]:        # 위도 배열이 큰 값에서 작은 값으로 내려가는 순서이면
            z = np.flipud(z).copy()                           # 이미지/격자 좌표와 맞추기 위해 상하 방향을 뒤집고 복사

        if normalize == 'standard_tanh':                      # 표준화 후 tanh를 적용하는 정규화 방식
            mean, std = float(z.mean()), float(z.std() + 1e-6) # 평균과 표준편차 계산; 0 나눗셈 방지를 위해 작은 값 추가
            elev = np.tanh((z - mean) / std)                  # z-score 표준화 후 tanh로 값 범위를 완만하게 제한
        elif normalize == 'minmax':                           # min-max 정규화 방식
            lo, hi = float(z.min()), float(z.max())           # 지형값 최소/최대 계산
            if hi == lo:                                      # 모든 값이 같아 분모가 0이 되는 경우
                elev = np.zeros_like(z, dtype=np.float32)     # 모든 값을 0으로 채운 배열 생성
            else:                                             # 최소값과 최대값이 다른 일반적인 경우
                elev = (z - lo) / (hi - lo)                   # 0~1 범위로 min-max 정규화
        else:                                                 # 지원하지 않는 정규화 방식이면
            raise ValueError(f"unknown normalize: {normalize}") # 어떤 normalize 값이 문제인지 에러 발생

        level = torch.from_numpy(elev)[None, None, :, :].to(dtype=dtype) # (H,W) 배열을 (1,1,H,W) Tensor로 변환

        chans = [level]                                       # 기본 elevation 채널을 채널 리스트에 먼저 저장

        if add_slope or add_relief or (avg_pool_ks and len(avg_pool_ks) > 0): # 파생 지형 특징 계산이 필요한 경우
            base = level                                      # 파생 특징 계산의 기준이 되는 고도 Tensor 저장

        if add_slope:                                         # 경사도 채널을 추가하는 경우
            kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=dtype).view(1,1,3,3) # x방향 Sobel kernel 생성
            ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=dtype).view(1,1,3,3) # y방향 Sobel kernel 생성

            slope_x = F.conv2d(base, kx, padding=1)           # 고도장에 x방향 Sobel filter를 적용해 x방향 경사 계산
            slope_y = F.conv2d(base, ky, padding=1)           # 고도장에 y방향 Sobel filter를 적용해 y방향 경사 계산

            chans += [slope_x, slope_y]                       # 계산된 경사도 채널 2개를 채널 리스트에 추가

        if add_relief:                                        # 기복도 채널을 추가하는 경우
            k = 11                                            # 주변 영역 크기 11x11 사용
            pad = k // 2                                      # 출력 크기를 유지하기 위한 padding 크기 계산

            pool_max = F.max_pool2d(base, kernel_size=k, stride=1, padding=pad) # 주변 11x11 영역의 최대 고도 계산
            pool_min = -F.max_pool2d(-base, kernel_size=k, stride=1, padding=pad) # 주변 11x11 영역의 최소 고도 계산

            relief = pool_max - pool_min                      # 국소 최대값과 최소값 차이로 지형 기복도 계산

            chans += [relief]                                 # 기복도 채널을 채널 리스트에 추가

        # if avg_pool_ks and len(avg_pool_ks) > 0:
        #     # 다중 스케일 평균 지형 정보 생성
        #     for k in avg_pool_ks:
        #         pad = k//2

        #         # k×k 영역의 평균 고도 계산
        #         avgk = F.avg_pool2d(base, kernel_size=k, stride=1, padding=pad)

        #         chans += [avgk]

        level_stack = torch.cat(chans, dim=1)                 # 채널 리스트를 channel 축(dim=1)으로 합쳐 (1,C,H,W) Tensor 생성

        self.level = level_stack.squeeze(0).contiguous()      # batch 차원만 제거해 (C,H,W) 형태의 고정 지형 Tensor로 저장

        self.extra_in_dim = self.level.shape[0]               # 지형 채널 수 C 저장; self.level이 (C,H,W)이므로 shape[0]이 채널 수

        self.lat = ds['lat'].values                           # 잘라낸 지형 자료의 위도 좌표 배열 저장
        self.lon = ds['lon'].values                           # 잘라낸 지형 자료의 경도 좌표 배열 저장

    def __len__(self):                                        # Dataset 길이를 반환하는 특수 메소드
        return int(self.repeat_to)                            # 이미지 Dataset과 길이를 맞추기 위해 repeat_to 반환
    
    def __getitem__(self, idx):                               # 어떤 index가 들어와도 동일한 정적 지형 Tensor 반환
        return self.level                                     # 저장된 지형 Tensor 반환, shape는 (C,H,W)


@register('paired-image-folders')                             # 이 Dataset 클래스를 'paired-image-folders'라는 이름으로 등록
class PairedImageFolders(Dataset):                            # 두 이미지 폴더와 선택적 관측/지형 자료를 묶어 반환하는 Dataset 클래스
    """
    root_path_1: 입력 이미지 폴더 A
    root_path_2: 정답 또는 대응 이미지 폴더 B
    topo_nc_path: 정적 지형 NetCDF 파일 경로
    obs_path_3  : 관측 CSV 또는 관측 raw 텍스트 파일 경로(옵션)
    """
    def __init__(self,                                        # paired Dataset 생성자 정의
                 root_path_1, root_path_2,                    # 서로 대응되는 두 이미지 폴더 경로
                 topo_nc_path=None, topo_kwargs=None,         # 선택적으로 사용할 지형 NetCDF 경로와 추가 옵션
                 obs_path_3=None, meta_path=None,             # 선택적으로 사용할 관측 파일 경로와 meta JSON 경로
                 **kwargs):                                   # ImageFolder로 넘길 추가 옵션들
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)   # 첫 번째 이미지 폴더를 ImageFolder Dataset으로 생성
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)   # 두 번째 이미지 폴더를 ImageFolder Dataset으로 생성

        self.topo_ds = None                                   # 지형 Dataset은 기본적으로 없는 상태로 초기화
        if topo_nc_path is not None:                          # 지형 NetCDF 경로가 주어진 경우
            topo_kwargs = topo_kwargs or {}                   # topo_kwargs가 None이면 빈 dict로 대체
            self.topo_ds = TopoInMemory(                      # 정적 지형 Dataset 생성
                path_1=topo_nc_path,                          # NetCDF 파일 경로 전달
                repeat_to=len(self.dataset_1),                # 첫 번째 이미지 Dataset과 길이를 맞춤
                lon_range=(124,134),                          # 사용할 경도 범위 지정
                lat_range=(33,43),                            # 사용할 위도 범위 지정
                **topo_kwargs                                 # 추가 지형 처리 옵션 전달
            )

        if obs_path_3 is not None:                            # 관측 파일 경로가 주어진 경우
            self.dataset_3 = ObsInMemory(root_path=obs_path_3, # 관측 자료를 메모리에 올리는 Dataset 생성
                                         meta_path=meta_path,  # 좌표/값 범위가 들어 있는 meta JSON 경로 전달
                                         flipped=True)         # 이미지 좌표계에 맞추기 위해 y축 flip 적용
        else:                                                 # 관측 파일이 없는 경우
            self.dataset_3 = None                             # 관측 Dataset을 사용하지 않도록 None 저장

    def __len__(self):                                        # paired Dataset 전체 길이 반환
        return len(self.dataset_1)                            # 기준 Dataset인 첫 번째 이미지 Dataset 길이를 반환

    def __getitem__(self, idx):                               # index에 해당하는 paired 샘플 반환
        a = self.dataset_1[idx]                               # 첫 번째 이미지 Dataset에서 입력 이미지 Tensor 가져오기
        b = self.dataset_2[idx]                               # 두 번째 이미지 Dataset에서 대응 이미지 Tensor 가져오기
        topo = self.topo_ds[idx] if self.topo_ds is not None else None # 지형 Dataset이 있으면 지형 Tensor를 가져오고 없으면 None

        if self.dataset_3 is not None:                        # 관측 Dataset을 함께 사용하는 경우
            obs = self.dataset_3[idx]                         # 현재 index에 해당하는 관측 dict 가져오기
            # print(obs['coord'].shape)
            return a, b, obs, topo                            # 이미지 2개, 관측 dict, 지형 Tensor를 함께 반환
        else:                                                 # 관측 Dataset을 사용하지 않는 경우
            return a, b, topo                                 # 이미지 2개와 지형 Tensor만 반환
