import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import models
from models import register
from utils import make_coord


@register('lte')
class LTE(nn.Module):
    """
    LTE (Local Texture Estimator) 를 다른 liif_* 래퍼들과 동일한 인터페이스로 맞춘 버전.

    - forward(inp, topo, coord, cell) 시그니처 통일 (topo 없는 config 면 None 이 넘어옴)
    - local_ensemble / cell_decode 플래그 지원
    - topo/level 을 query 좌표에서 샘플링해 decoder 입력에 concat (extra_in_dim)
    - layer_spec 으로 correction(adapter) layer 부착 가능
    - self.le_preds / self.le_weights 로 방향별 중간값 보관 (분석/추출용)
    """

    def __init__(self, encoder_spec, imnet_spec=None, layer_spec=None,
                 hidden_dim=256, local_ensemble=True, cell_decode=True,
                 extra_in_dim=0, adapter_per_direction=False):
        super().__init__()

        # adapter_per_direction=True : LIIF(liif_baseline.py)와 같이 방향별 imnet 출력에 adapter 적용 -> 4방향 blend -> (+ LR bilinear 잔차, adapter 없음)
        # adapter_per_direction=False(기본, 기존 동작) : blend + LR bilinear 잔차 뒤에 adapter 1회 적용
        self.adapter_per_direction = adapter_per_direction
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode
        self.hidden_dim = hidden_dim

        # Encoder 생성
        # inp: [16, 1, 50, 50]
        # encoder feat: [16, 256, 50, 50]
        self.encoder = models.make(encoder_spec)

        # LTE 고유 모듈
        # coef / freq: encoder feature -> 진폭 / 주파수 성분
        # phase: cell(2) -> 위상 (hidden_dim//2)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)

        if imnet_spec is not None:
            # LTE decoder 입력: cos/sin 푸리에 basis * coef -> hidden_dim
            # topo/level 정보(q_level: [16, 59, 1]) 를 추가로 concat
            imnet_in_dim = hidden_dim + extra_in_dim
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

        if layer_spec is not None:
            # imnet 출력 차원
            # pred: [16, 59, 1]
            adapter_in_dim = self.imnet.out_dim

            # correction layer 생성
            self.add_layer = models.make(layer_spec, args={'in_dim': adapter_in_dim})

    def gen_feat(self, inp):
        # inp: [16, 1, 50, 50]
        self.inp = inp

        # encoder feature 및 LTE 진폭/주파수 성분 생성
        self.feat = self.encoder(inp)          # [16, 256, 50, 50]
        self.coeff = self.coef(self.feat)      # [16, hidden_dim, 50, 50]
        self.freqq = self.freq(self.feat)      # [16, hidden_dim, 50, 50]

        # feature map 각 위치의 정규화 좌표
        # feat_coord: [16, 2, 50, 50]
        self.feat_coord = make_coord(self.feat.shape[-2:], flatten=False) \
            .to(self.feat.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(self.feat.shape[0], 2, *self.feat.shape[-2:])

        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat            # [16, 256, 50, 50]
        coef = self.coeff           # [16, hidden_dim, 50, 50]
        freq = self.freqq           # [16, hidden_dim, 50, 50]
        feat_coord = self.feat_coord

        # topo/level: [16, 1, 100, 100] (topo_nc_path 없는 config면 None)
        level = getattr(self, 'topo', None)
        if level is not None and level.dim() == 5:
            level = level.squeeze(2)

        if self.local_ensemble:
            # query 주변 4개 방향 사용: (-1,-1), (-1,1), (1,-1), (1,1)
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # feature grid 반 칸 크기 (global 좌표계 [-1, 1] 기준)
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # topo/level 은 encoder feature grid 와 무관한 별도 해상도이므로
        # local ensemble 로 이동된 coord_ 가 아니라 실제 query 좌표(coord)에서 샘플링
        if level is not None:
            q_level = F.grid_sample(
                level, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)
            # q_level: [16, 59, 1]
        else:
            q_level = None

        preds = []   # 방향별 pred (총 4개 또는 1개)
        areas = []   # 방향별 area

        for vx in vx_lst:
            for vy in vy_lst:
                # query 좌표를 현재 방향으로 feature grid 반 칸만큼 이동
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # 이동된 좌표에서 coef / freq / feature 좌표 샘플링
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # 원래 query 좌표와 선택된 feature grid 좌표 사이의 상대좌표
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # cell 을 feature 해상도 기준으로 scaling (위상 입력)
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]

                bs, q = coord.shape[:2]

                # 푸리에 basis 생성
                # q_freq: [bs, q, hidden_dim] -> [bs, q, 2, hidden_dim//2]
                # freq는 인코더가 예측한 주파수 텐서이고, 채널 수가 hidden_dim
                # 이 주파수들을 좌표에 곱해서 위치별 위상을 만들고, 그 위상으로 cos/sin 푸리에 기저 생성 목적
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)  # 마지막 축의 주파수들을 x,y 축으로 나눔 [bs, q, 2, hidden_dim//2]
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))           # 주파수와 상대좌표를 내적   [bs, q, 2, hidden_dim//2]
                q_freq = torch.sum(q_freq, dim=-2)                            # 방금 만든 x기여분, y 기여분을 더해서 합침 -> 최종 shape [bs, q, 128]
                if self.cell_decode:
                    q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)

                # decoder 입력: basis * 진폭
                inp = torch.mul(q_coef, q_freq)     # [bs, q, hidden_dim]

                # topo/level concat (있을 때만)
                if q_level is not None:
                    inp = torch.cat([inp, q_level], dim=-1)

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                # 방향별 correction: adapter_per_direction=True 일 때만 (기본은 blend 후 1회 적용)
                if self.adapter_per_direction:
                    pred = self.correction_layer(pred)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)

        if self.local_ensemble:
            # bilinear interpolation 에서 반대편 면적이 weight 로 쓰이므로 순서 재배치
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        # 방향별 중간값 보관 (분석/추출용, forward 동작에는 영향 없음)
        self.le_preds = preds
        self.le_weights = [area / tot_area for area in areas]

        # LTE residual: 저해상도 입력의 bilinear 보간을 더함
        ret = ret + F.grid_sample(
            self.inp, coord.flip(-1).unsqueeze(1),
            mode='bilinear', padding_mode='border', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)

        return ret

    def correction_layer(self, preds):
        # preds before correction: [16, 59, 1]
        if hasattr(self, 'add_layer'):
            return self.add_layer(preds)
        return preds

    def forward(self, inp, topo, coord, cell):
        # inp:   [16, 1, 50, 50]
        # topo:  [16, 1, 100, 100] (없는 config면 None)
        # coord: [16, 59, 2]
        # cell:  [16, 59, 2]

        # encoder feature / coef / freq 생성
        self.gen_feat(inp)

        # query_rgb 에서 사용할 topo 저장
        self.topo = topo

        # adapter 적용 전 LTE 출력 (local ensemble blend 까지 반영)
        preds_before_adapter = self.query_rgb(coord, cell)
        self.preds_before_adapter = preds_before_adapter

        # 기본: blend 결과에 correction(adapter) 1회 적용 (add_layer 없으면 그대로 통과)
        # adapter_per_direction=True 면 query_rgb 안에서 이미 방향별로 적용됨 -> 다시 적용하지 않음
        #   (이때 self.preds_before_adapter 는 adapter 적용 후 값임에 주의)
        if self.adapter_per_direction:
            return preds_before_adapter
        return self.correction_layer(preds_before_adapter)
