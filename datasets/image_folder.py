import os                                                   # 파일/디렉토리 경로 처리와 폴더 목록 조회에 사용
import json                                                 # split/meta 설정 파일(JSON)을 읽는 데 사용
from PIL import Image                                        # 이미지 파일을 열고 grayscale로 변환하는 데 사용

import pickle                                               # 텐서화한 이미지를 .pkl 바이너리 파일로 저장/로드하는 데 사용
import imageio                                              # 이미지 입출력 라이브러리 import; 현재 코드에서는 직접 사용되지 않음
import numpy as np                                          # 배열 연산, 좌표/관측값 처리, 정규화 계산에 사용
import torch                                                # PyTorch Tensor 생성 및 연산에 사용
import torch.nn.functional as F                             # TopoInMemory의 Sobel/pooling 연산에 사용
from torch.utils.data import Dataset                        # PyTorch Dataset 클래스를 상속하기 위해 사용
from torchvision import transforms                          # PIL 이미지를 Tensor로 변환하기 위해 사용
from utils import pixel_to_liif_coord, latlon_to_pixel_idx   # 위경도↔픽셀좌표↔LIIF 좌표 변환 함수 import
import xarray as xr                                         # NetCDF 지형 자료를 읽기 위해 사용
from netCDF4 import Dataset as NCDataset

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

@register("netcdf-folder")                 # config에서 name: netcdf-folder로 호출할 수 있도록 Dataset 등록
class NetCDFFolder(Dataset):               # NetCDF 파일을 PyTorch Dataset 형태로 사용하는 클래스

    def __init__(                          # Dataset 생성자
        self,
        nc_path,                           # NetCDF(.nc) 파일 경로
        var_name="precipitation",          # 사용할 변수 이름 (기본: precipitation)
        start_idx=0,                       # 사용할 시작 시간 index
        end_idx=None,                      # 사용할 마지막 시간 index (None이면 끝까지)
        repeat=1,                          # Dataset을 반복해서 사용할 횟수
        load_into_memory=True              # True이면 모든 데이터를 메모리에 미리 로드
    ):
        super().__init__()                 # 부모(Dataset) 초기화

        self.repeat = int(repeat)          # repeat를 정수형으로 저장
        self.start_idx = int(start_idx)    # 시작 index 저장
        self.end_idx = None if end_idx is None else int(end_idx)  # end_idx가 있으면 정수형으로 저장
        self.load_into_memory = bool(load_into_memory)            # 메모리 로드 여부 저장

        # ----------------------------------------------------
        # NetCDF 파일 열기
        # ----------------------------------------------------
        self.nc = NCDataset(nc_path, mode="r")   # 읽기 모드로 NetCDF 파일 열기

        # ----------------------------------------------------
        # 원하는 변수가 존재하는지 확인
        # ----------------------------------------------------
        if var_name not in self.nc.variables:
            available = list(self.nc.variables.keys())  # 존재하는 변수 목록 가져오기
            self.nc.close()                             # 파일 닫기
            raise KeyError(
                f"'{var_name}' 변수가 없습니다. "
                f"사용 가능한 변수: {available}"
            )

        self.var = self.nc.variables[var_name]         # 사용할 변수 객체 저장

        # 위경도 좌표 보관 (지형을 레이더 격자에 위경도로 맞출 때 사용; 없으면 None)
        self.lat = np.asarray(self.nc.variables['lat'][:], dtype=np.float64) if 'lat' in self.nc.variables else None
        self.lon = np.asarray(self.nc.variables['lon'][:], dtype=np.float64) if 'lon' in self.nc.variables else None

        # ----------------------------------------------------
        # 변수 차원 확인
        # 반드시 (time, lat, lon) 형태여야 함
        # ----------------------------------------------------
        if self.var.ndim != 3:
            shape = self.var.shape                     # 현재 shape 저장
            self.nc.close()                            # 파일 닫기
            raise ValueError(
                f"'{var_name}'은 (time, lat, lon) 3차원이어야 합니다. "
                f"현재 shape={shape}"
            )

        total_len = self.var.shape[0]                  # 전체 시간 개수(time dimension)

        # ----------------------------------------------------
        # end_idx가 지정되지 않았으면 마지막까지 사용
        # ----------------------------------------------------
        if self.end_idx is None:
            self.end_idx = total_len

        # ----------------------------------------------------
        # index 범위가 올바른지 검사
        # ----------------------------------------------------
        if not (0 <= self.start_idx < self.end_idx <= total_len):
            self.nc.close()
            raise ValueError(
                f"잘못된 인덱스 범위입니다: "
                f"start_idx={self.start_idx}, "
                f"end_idx={self.end_idx}, "
                f"total={total_len}"
            )

        # 실제 사용할 시간 개수
        self.base_len = self.end_idx - self.start_idx

        # ----------------------------------------------------
        # 메모리에 모두 올리는 경우
        # ----------------------------------------------------
        if self.load_into_memory:

            # (time, lat, lon) 데이터를 한 번에 읽음
            data = self.var[self.start_idx:self.end_idx]

            # MaskedArray이면 결측값을 0으로 채움
            if np.ma.isMaskedArray(data):
                data = data.filled(0.0)

            # numpy float32 배열로 변환
            self.data = np.asarray(data, dtype=np.float32)

            # 이미 메모리에 다 올렸으므로 파일은 닫음
            self.nc.close()
            self.nc = None
            self.var = None

        else:
            # 메모리에 올리지 않을 경우
            # __getitem__에서 매번 NetCDF를 읽음
            self.data = None

    # ========================================================
    # Dataset 길이
    # ========================================================
    def __len__(self):

        # repeat가 2라면 길이도 두 배가 됨
        return self.base_len * self.repeat

    # ========================================================
    # 샘플 하나 반환
    # ========================================================
    def __getitem__(self, idx):

        # repeat 때문에 index가 넘어갈 수 있으므로
        # 원래 데이터 길이 안으로 순환
        local_idx = int(idx) % self.base_len

        # --------------------------------------------
        # 메모리에 이미 로드된 경우
        # --------------------------------------------
        if self.data is not None:

            # numpy array (H,W)
            arr = self.data[local_idx]

        # --------------------------------------------
        # 메모리에 없으면 NetCDF에서 직접 읽기
        # --------------------------------------------
        else:

            arr = self.var[self.start_idx + local_idx]

            # MaskedArray이면 결측값을 0으로 변경
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(0.0)

            # float32 numpy 배열로 변환
            arr = np.asarray(arr, dtype=np.float32)

        # --------------------------------------------
        # PyTorch Tensor로 변환
        # --------------------------------------------

        # (H,W)
        # ↓
        # unsqueeze(0)
        # ↓
        # (1,H,W)
        # CNN 입력 형태(Channel 추가)
        return torch.from_numpy(arr).unsqueeze(0)

    # ========================================================
    # 객체가 삭제될 때 NetCDF 파일 닫기
    # ========================================================
    def __del__(self):

        # nc 객체가 존재하는지 확인
        nc = getattr(self, "nc", None)

        if nc is not None:
            try:
                nc.close()                 # 파일 닫기
            except Exception:
                pass                       # 이미 닫혀있으면 무시

####################
# 실측 데이터 전처리#
####################
@register('obs-folder')                                      # 이 Dataset 클래스를 'obs-folder'라는 이름으로 등록
class ObsInMemory(Dataset):                                  # 관측 자료를 날짜별로 메모리에 저장하는 Dataset 클래스
    def __init__(self, root_path=None, meta_path=None, hr_size=100,
                 flipped=False, first_k=None,
                 coord_mode='pixel_center', normalize=None):
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

        self.flipped = bool(flipped)                        # y축 방향 뒤집힘 여부 저장
        self.normalize = normalize                          # 관측값 정규화 방식 저장
        self.coord_mode = coord_mode                        # 위경도를 LIIF 좌표로 변환하는 방식 저장
        self.MISSING = -2.0                                 # 결측값을 표시하기 위한 sentinel 값

        with open(meta_path, 'r') as f:                      # 좌표 범위와 값 범위가 들어 있는 meta JSON 파일 열기
            meta = json.load(f)                             # JSON 내용을 dict로 읽기

        if 'original_shape' in meta:                        # meta에 실제 HR (H, W)가 기록되어 있으면 그 값을 사용
            self.hr_h = int(meta['original_shape'][-2])      # GT는 정사각형이 아닐 수 있으므로(예: 100x80) H, W를 따로 저장
            self.hr_w = int(meta['original_shape'][-1])
        else:                                               # original_shape이 없는 meta에 대한 하위호환
            self.hr_h = int(hr_size)
            self.hr_w = int(hr_size)

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
                if len(cols) < 5:                            # 필요한 컬럼(idx,time,lon,lat,value)이 부족하면
                    continue                                 # 해당 줄은 건너뜀
                date = cols[1].strip()                       # time/date 컬럼을 읽고 앞뒤 공백 제거
                try:                                         # 좌표 값 변환 중 오류가 날 수 있으므로 예외 처리
                    lon = float(cols[2]); lat = float(cols[3]) # x=경도, y=위도 컬럼 순서로 float 변환
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
                    gt[mask] = (gt[mask] - sub) / div # 유효값에 대해서만 (값-sub)/div 정규화 수행
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

    채널 구성 (순서 고정):
        - elevation(1채널, 표준화+tanh 또는 minmax)
        - slope_x, slope_y (add_slope=True일 때 +2채널, 정규화된 elevation에 Sobel 적용, 재정규화 없음)
        - relief           (add_relief=True일 때 +1채널, 정규화된 elevation의 11x11 max-min, 재정규화 없음)
    """
    def __init__(self, path_1=None, lat_range=None, lon_range=None, normalize='standard_tanh', add_slope=False, add_relief=False, dtype=torch.float32, repeat_to=1, grid_lat=None, grid_lon=None, grid_scale=4, **legacy_kwargs):
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

        add_slope : bool
            True면 slope_x, slope_y 2채널 추가 (3x3 Sobel, padding=1)

        add_relief : bool
            True면 relief 1채널 추가 (11x11 창의 max - min)

        dtype : torch.dtype
            출력 텐서 자료형

        repeat_to : int
            Dataset 길이를 repeat_to로 맞추기 위한 값

        grid_lat, grid_lon : 1-D array, optional
            레이더 LR 격자의 위도/경도 (오름차순, 픽셀 중심). 주어지면 lat_range/lon_range 는 무시하고,
            지형을 레이더 LR 격자 영역(픽셀 가장자리 기준)에 위경도로 맞춰 nearest 리샘플한다.
            결과 크기는 (len(grid_lat)*grid_scale, len(grid_lon)*grid_scale) 이라서 LR 픽셀 1개가 정확히
            grid_scale x grid_scale 지형 픽셀에 대응 -> wrapper 의 비율 crop 이 위경도 기준으로 정확히 맞고
            LR crop 이 정사각이면 지형 crop 도 정사각이 된다. 지형 파일 밖 영역은 고도 0 m(바다)로 채운다.

        grid_scale : int
            grid_lat/grid_lon 사용 시 LR 픽셀당 지형 픽셀 수 (한 변). 기본 4 (HSR 기준 약 250 m)

        legacy_kwargs :
            avg_pool_ks 등 그 외 인자는 warn 후 무시 (평균풀링 채널은 옛 버전에서도 주석 처리였음).
        """
        super().__init__()                                    # Dataset 부모 클래스 초기화
        if legacy_kwargs:                                     # 제거된 slope/relief 실험 인자가 넘어오면
            import warnings
            warnings.warn(
                f"[TopoInMemory] 제거된 인자 무시: {sorted(legacy_kwargs)} "
                "(지원하지 않는 인자)"
            )
        self.repeat_to = int(repeat_to)                       # __len__에서 반환할 Dataset 길이 저장

        raw_ds = xr.open_dataset(path_1)                      # NetCDF 파일 열기 (파일마다 고도 변수명이 다를 수 있음)

        # 좌표 이름 표준화: 파일마다 latitude/longitude 로 되어 있을 수 있으므로 lat/lon 으로 통일
        rename_map = {}
        for cand in ('latitude', 'Latitude', 'LAT', 'y'):
            if cand in raw_ds.variables or cand in raw_ds.dims:
                rename_map[cand] = 'lat'
                break
        for cand in ('longitude', 'Longitude', 'LON', 'x'):
            if cand in raw_ds.variables or cand in raw_ds.dims:
                rename_map[cand] = 'lon'
                break
        if rename_map:
            raw_ds = raw_ds.rename(rename_map)

        if 'z' in raw_ds.data_vars:                           # 예: ETOPO1_on_IMERG_Land.nc
            var_name = 'z'
        elif 'elev' in raw_ds.data_vars:                      # 예: ETOPO1_Ice_g_gmt4_kor.nc (raw ETOPO1)
            var_name = 'elev'
        else:
            raise KeyError(
                f"[TopoInMemory] 'z' 또는 'elev' 변수를 찾을 수 없습니다. "
                f"사용 가능한 변수: {list(raw_ds.data_vars)}"
            )

        if grid_lat is not None and grid_lon is not None:     # 레이더 격자에 위경도로 맞춰 리샘플하는 경우
            z, out_lat, out_lon = self._resample_to_grid(raw_ds[var_name], grid_lat, grid_lon, int(grid_scale))
        else:
            ds = raw_ds[var_name].sel(                        # 고도 변수를 선택한 뒤 지정한 위경도 범위로 잘라 읽기
                lat=slice(lat_range[0], lat_range[1]),        # 사용할 위도 범위 적용
                lon=slice(lon_range[0], lon_range[1])         # 사용할 경도 범위 적용
            ).fillna(0)                                       # 지형 자료의 NaN 값을 0으로 대체

            if 'time' in ds.dims:                             # raw ETOPO1 파일처럼 길이 1짜리 time 차원이 남아있으면
                ds = ds.isel(time=0)                          # (H, W)만 남도록 제거

            z = ds.values.astype(np.float32)                  # xarray DataArray를 float32 numpy 배열로 변환

            if ds['lat'].values[0] > ds['lat'].values[-1]:    # 위도 배열이 큰 값에서 작은 값으로 내려가는 순서이면
                z = np.flipud(z).copy()                       # 이미지/격자 좌표와 맞추기 위해 상하 방향을 뒤집고 복사
            out_lat, out_lon = ds['lat'].values, ds['lon'].values

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
        base = level                                          # 파생 특징 계산의 기준이 되는 (정규화된) 고도 Tensor

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

            chans += [pool_max - pool_min]                    # 국소 최대-최소 차이로 지형 기복도 채널 추가

        level_stack = torch.cat(chans, dim=1)                 # 채널 리스트를 channel 축(dim=1)으로 합쳐 (1,C,H,W) Tensor 생성

        self.level = level_stack.squeeze(0).contiguous()      # batch 차원만 제거해 (C,H,W) 형태의 고정 지형 Tensor로 저장

        self.extra_in_dim = self.level.shape[0]               # 지형 채널 수 C 저장; self.level이 (C,H,W)이므로 shape[0]이 채널 수

        self.lat = out_lat                                    # 지형 격자의 위도 좌표 배열 저장
        self.lon = out_lon                                    # 지형 격자의 경도 좌표 배열 저장

    @staticmethod
    def _resample_to_grid(da, grid_lat, grid_lon, k):
        """ 지형 DataArray 를 레이더 LR 격자 영역에 맞춰 (len(grid_lat)*k, len(grid_lon)*k) 로 nearest 리샘플.
            LR 픽셀 i 는 [lat0 + (i-0.5)d, lat0 + (i+0.5)d] 를 덮고, 그 안을 k 등분한 중심점마다 가장 가까운 지형 픽셀을 취한다. """
        if 'time' in da.dims:
            da = da.isel(time=0)
        da = da.sortby('lat').sortby('lon')                   # 오름차순(row0=남쪽, col0=서쪽) 보장
        src_lat, src_lon = da['lat'].values.astype(np.float64), da['lon'].values.astype(np.float64)
        full = da.fillna(0).values.astype(np.float32)

        def targets(g):
            g = np.asarray(g, dtype=np.float64)
            d = (g[-1] - g[0]) / (len(g) - 1)                 # LR 픽셀 간격
            lo = g[0] - d / 2                                 # 첫 LR 픽셀의 가장자리
            return lo + (np.arange(len(g) * k) + 0.5) * (d / k)

        def nearest(src, tgt):
            step = (src[-1] - src[0]) / (len(src) - 1)
            idx = np.rint((tgt - src[0]) / step).astype(np.int64)
            valid = (idx >= 0) & (idx < len(src))             # 지형 파일 밖이면 invalid
            return np.clip(idx, 0, len(src) - 1), valid

        t_lat, t_lon = targets(grid_lat), targets(grid_lon)
        ri, rv = nearest(src_lat, t_lat)
        ci, cv = nearest(src_lon, t_lon)
        z = full[np.ix_(ri, ci)]
        z[~rv, :] = 0.0                                       # 영역 밖은 바다(0 m)로
        z[:, ~cv] = 0.0
        return z, t_lat, t_lon

    def __len__(self):                                        # Dataset 길이를 반환하는 특수 메소드
        return int(self.repeat_to)                            # 이미지 Dataset과 길이를 맞추기 위해 repeat_to 반환
    
    def __getitem__(self, idx):                               # 어떤 index가 들어와도 동일한 정적 지형 Tensor 반환
        return self.level                                     # 저장된 지형 Tensor 반환, shape는 (C,H,W)

# 이미지 자체를 넣어주지 말고
@register('paired-image-folders')                             # 이 Dataset 클래스를 'paired-image-folders'라는 이름으로 등록
class PairedImageFolders(Dataset):                            # 두 이미지 폴더와 선택적 관측/지형 자료를 묶어 반환하는 Dataset 클래스
    """
    root_path_1: 입력 이미지 폴더 A
    root_path_2: 정답 또는 대응 이미지 폴더 B
    topo_nc_path: 정적 지형 NetCDF 파일 경로
    obs_path_3  : 관측 CSV 또는 관측 raw 텍스트 파일 경로(옵션)
    """
    def __init__(self,                                        # paired Dataset 생성자 정의
                 root_path_1, root_path_2,                    # 서로 대응되는 두 입력/정답 경로
                 topo_nc_path=None, topo_kwargs=None,         # 선택적으로 사용할 지형 NetCDF 경로와 추가 옵션
                 obs_path_3=None, meta_path=None,             # 선택적으로 사용할 관측 파일 경로와 meta JSON 경로
                 loader=None,                                 # 'netcdf' | 'image' (None이면 경로로 자동 판별)
                 **kwargs):                                   # 하위 Dataset(NetCDFFolder/ImageFolder)로 넘길 추가 옵션들

        # loader 자동 판별: root_path_1이 디렉토리면 이미지 폴더, 그 외(.nc 파일)면 NetCDF
        if loader is None:
            loader = 'image' if os.path.isdir(root_path_1) else 'netcdf'

        if loader == 'image':
            # 이미지 폴더 입력: ImageFolder가 자체 topo_ds를 만들지 않도록 topo_nc_path는 넘기지 않음
            self.dataset_1 = ImageFolder(root_path_1, **kwargs)   # 입력 이미지 폴더
            self.dataset_2 = ImageFolder(root_path_2, **kwargs)   # 정답 이미지 폴더
        elif loader == 'netcdf':
            self.dataset_1 = NetCDFFolder(root_path_1, **kwargs)   # 입력 NetCDF
            self.dataset_2 = NetCDFFolder(root_path_2, **kwargs)   # 정답 NetCDF
        else:
            raise ValueError(f"[PairedImageFolders] 알 수 없는 loader: {loader!r} (netcdf | image)")

        self.topo_ds = None                                   # 지형 Dataset은 기본적으로 없는 상태로 초기화
        if topo_nc_path is not None:                          # 지형 NetCDF 경로가 주어진 경우
            topo_kwargs = dict(topo_kwargs or {})             # topo_kwargs가 None이면 빈 dict로 대체 (원본 config dict는 건드리지 않음)
            # align_to_lr: true 일 때만 지형을 입력(LR) nc 의 위경도에 맞춘다 (기본 False = 기존 lon_range/lat_range 방식 유지;
            #   범위를 안 준 기존 config(IMERG/GPP 등)의 지형이 조용히 바뀌어 기존 체크포인트와 어긋나는 것을 막기 위함)
            align_to_lr = bool(topo_kwargs.pop('align_to_lr', False))
            if align_to_lr and 'lon_range' not in topo_kwargs and 'lat_range' not in topo_kwargs \
                    and getattr(self.dataset_1, 'lat', None) is not None and getattr(self.dataset_1, 'lon', None) is not None:
                # 입력(LR) nc 의 위경도에 지형을 맞춘다 (grid_scale 은 topo_kwargs 로 조절)
                topo_kwargs['grid_lat'] = self.dataset_1.lat
                topo_kwargs['grid_lon'] = self.dataset_1.lon
            else:
                topo_kwargs.setdefault('lon_range', (124, 132))   # 기존 방식: 범위를 직접 지정 (IMERG 기본값)
                topo_kwargs.setdefault('lat_range', (33, 43))
            self.topo_ds = TopoInMemory(                      # 정적 지형 Dataset 생성
                path_1=topo_nc_path,                          # NetCDF 파일 경로 전달
                repeat_to=len(self.dataset_1),                # 첫 번째 이미지 Dataset과 길이를 맞춤
                **topo_kwargs                                 # 범위/격자/정규화 등 추가 지형 처리 옵션 전달
            )

        if obs_path_3 is not None:                            # 관측 파일 경로가 주어진 경우
            self.dataset_3 = ObsInMemory(root_path=obs_path_3, # 관측 자료를 메모리에 올리는 Dataset 생성
                                         meta_path=meta_path,  # 좌표/값 범위가 들어 있는 meta JSON 경로 전달
                                         flipped=False)        
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
            return a, b, obs, topo                            # 이미지 2개, 관측 dict, 지형 Tensor를 함께 반환
        else:                                                 # 관측 Dataset을 사용하지 않는 경우
            return a, b, topo                                 # 이미지 2개와 지형 Tensor만 반환
