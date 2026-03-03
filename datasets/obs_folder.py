from datasets import register
from torch.utils.data import Dataset
from utils import pixel_to_liif_coord, latlon_to_pixel_idx
import numpy as np
import os

####################
# 실측 데이터 전처리#
####################

@register('obs-folder')
class ObsFolder(Dataset):
    def __init__(self, root_path=None, meta_path=None, hr_size=60, flipped=False, normalize=None, first_k=None):
        import json

        self.hr_h = int(hr_size)
        self.hr_w = int(hr_size)
        self.flipped = bool(flipped)
        self.normalize = normalize
        
        if meta_path is not None:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
    
        # Lat_range = [Lat_max, Lat_min]
        self.lat_max = float(meta['lat_range'][0])
        self.lat_min = float(meta['lat_range'][1])
        self.lon_max = float(meta['lon_range'][0])
        self.lon_min = float(meta['lon_range'][1])
        # # print(f'this is obs_folder. {self.lat_max}, {self.lat_min}, {self.lon_max}, {self.lon_min}')

        # 파일에서 min/max를 구해서 정의해야 함
        self.value_min = 0.0
        self.value_max = 0.0

        # 파일 파싱
        with open(root_path, 'r') as f:
            raw_text = f.read()
        
        blocks = raw_text.split('/\\')

        if first_k is not None:
            blocks = blocks[:first_k]
        
        self.data = dict() # 날짜별로

        for block in blocks:
            lines = [ln.strip() for ln in block.strip().split('\n') if ln.strip()]
            if not lines:
                continue

            for line in lines[1:]:
                if line.strip():
                    row = line.strip().split(',')
                    if len(row) < 5:
                        continue
                    date = row[1].strip()
                    try:
                        lat = float(row[2])
                        lon = float(row[3])
                    except ValueError:
                        continue
                    
                    v = row[4].strip()
                    if v in ['', 'nan', 'NaN','None', None]:
                        v = 0.0
                    else:
                        try: v = float(v)
                        except Exception: v = None
                    
                    if float(v) > self.value_max:
                        self.value_max = float(v)
                    self.data.setdefault(date, []).append((lat, lon, v))

        self.keys = sorted(self.data.keys())

    def __len__(self):
        return len(self.data)
    
    def get_minmax_value(self):
        return self.value_min, self.value_max
    
    def __getitem__(self, idx):
        import torch

        date = self.keys[idx]
        rows = self.data[date]

        coords_xy = []
        values = []

        for lat, lon, v in rows:
            i, j = latlon_to_pixel_idx(
                lat, lon,
                self.lat_min, self.lat_max,
                self.lon_min, self.lon_max,
                self.hr_h, self.hr_w,
                flipped=self.flipped)
            x, y = pixel_to_liif_coord(i, j, self.hr_h, self.hr_w)
            coords_xy.append((x, y))

            if v is None:
                v = 0.0
            
            values.append(v)

        # obs 데이터 전처리
        values = np.array(values)
        coord_tensor = torch.tensor(coords_xy, dtype=torch.float32)
        normed_data = torch.from_numpy(values).float() / 255.0
        value_tensor = torch.tensor(normed_data, dtype=torch.float32)
        # print(value_tensor)

        # LIIF cell 
        cell = torch.ones_like(coord_tensor)
        if cell.numel() > 0:
            cell[:, 0] *= 2.0 / self.hr_w 
            cell[:, 1] *= 2.0 / self.hr_h
        
        return {
            'coord': coord_tensor,
            'gt': value_tensor,
            'cell': cell
        }