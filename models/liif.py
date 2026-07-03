import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, layer_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, extra_in_dim=0):
        super().__init__()

        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        # Encoder 생성
        # inp: [16, 1, 50, 50]
        # encoder feat: [16, 256, 50, 50]
        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            # encoder 출력 channel 수: 256
            imnet_in_dim = self.encoder.out_dim

            # feature unfold 사용 시 3x3 주변 feature를 함께 사용
            # [16, 256, 50, 50] -> [16, 2304, 50, 50]
            # 2304 = 256 * 9
            if self.feat_unfold:
                imnet_in_dim *= 9

            # 상대좌표 rel_coord(x, y) 추가
            # q_coord: [16, 59, 2]
            # rel_coord: [16, 59, 2]
            imnet_in_dim += 2

            # cell 정보(x, y) 추가
            # cell: [16, 59, 2]
            if self.cell_decode:
                imnet_in_dim += 2

            # topo/level 정보 추가
            # q_level: [16, 59, 1]
            imnet_in_dim += extra_in_dim

            # 최종 decoder input dim
            # 2304 + 2 + 1 + 2 = 2309
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
        self.feat = self.encoder(inp)

        # self.feat: [16, 256, 50, 50]
        return self.feat

    def query_rgb(self, coord, cell=None):
        # feat: [16, 256, 50, 50]
        feat = self.feat

        # topo/level: [16, 1, 100, 100]
        level = self.topo

        device = feat.device

        # coord: [16, 59, 2]
        if self.imnet is None:
            ret = F.grid_sample(
                feat,
                coord.flip(-1).unsqueeze(1),  # [16, 1, 59, 2]
                mode='nearest',
                align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)

            # ret: [16, 59, 256]
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view( 
                feat.shape[0],
                feat.shape[1] * 9,
                feat.shape[2],
                feat.shape[3]
            )

            # feat after unfold: [16, 2304, 50, 50]

        if self.local_ensemble:
            # local ensemble 사용 시 query 주변 4개 방향 사용
            # (-1,-1), (-1,1), (1,-1), (1,1)
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # feature grid 반 칸 크기
        # feat 해상도 50x50이므로 rx = ry = 2/50/2 = 0.02
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # feature map 각 위치의 정규화 좌표 생성
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda(device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        # feat_coord: [16, 2, 50, 50]

        preds = []   # 각 local ensemble 방향별 pred 저장, 총 4개
        areas = []   # 각 local ensemble 방향별 area 저장, 총 4개

        for vx in vx_lst:
            for vy in vy_lst:
                # coord: [16, 59, 2]
                coord_ = coord.clone()

                # query 좌표를 현재 방향(vx, vy)으로 feature grid 반 칸만큼 이동
                # coord_: [16, 59, 2]
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift

                # grid_sample 입력 좌표가 [-1, 1] 범위를 넘지 않도록 제한
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # 이동된 query 좌표에서 feature 값 추출
                q_feat = F.grid_sample(
                    feat,
                    coord_.flip(-1).unsqueeze(1),  # [16, 1, 59, 2]
                    mode='nearest',
                    align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)

                # q_feat: [16, 59, 2304]

                # 이동된 query 좌표와 가장 가까운 feature grid 좌표 추출
                q_coord = F.grid_sample(
                    feat_coord,
                    coord_.flip(-1).unsqueeze(1),
                    mode='nearest',
                    align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)

                # q_coord: [16, 59, 2]

                # 이동된 query 좌표에 해당하는 topo/level 값 추출
                q_level = F.grid_sample(
                    level,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)

                # q_level: [16, 59, 1]

                # 원래 query 좌표와 선택된 feature grid 좌표 사이의 상대좌표 계산
                rel_coord = coord - q_coord

                # rel_coord: [16, 59, 2]

                # 정규화 좌표계의 상대좌표를 feature 해상도 기준으로 scaling
                # rel_coord: [16, 59, 2]
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # decoder 입력 구성
                # q_feat: [16, 59, 2304]
                # rel_coord: [16, 59, 2]
                # q_level: [16, 59, 1]
                inp = torch.cat([q_feat, rel_coord, q_level], dim=-1)

                # inp: [16, 59, 2307]

                if self.cell_decode:
                    # cell: [16, 59, 2]
                    rel_cell = cell.clone()

                    # cell도 feature 해상도 기준으로 scaling
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]

                    # rel_cell: [16, 59, 2]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                    # decoder input: [16, 59, 2309]
                    # 2309 = 2304(q_feat) + 2(rel_coord) + 1(q_level) + 2(cell)

                # bs = 16, q = 59
                bs, q = coord.shape[:2]

                # MLP는 2D 입력을 받으므로 batch와 query 차원을 합침
                # decoder input: [16, 59, 2309]
                # decoder input 2D: [944, 2309]
                # 944 = 16 * 59
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)

                # pred: [16, 59, 1]
                preds.append(pred)

                # local ensemble weight 계산을 위한 area
                # rel_coord[:, :, 0]: [16, 59]
                # rel_coord[:, :, 1]: [16, 59]
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])

                # area: [16, 59]
                areas.append(area + 1e-9)

        # areas는 local ensemble 4방향에 대해 저장됨
        # len(areas) = 4
        # torch.stack(areas): [4, 16, 59]
        # tot_area: [16, 59]
        tot_area = torch.stack(areas).sum(dim=0)

        if self.local_ensemble:
            # bilinear interpolation에서 반대편 면적이 weight로 사용되므로
            # area 순서를 pred 순서에 맞게 재배치
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

        # 최종 출력 초기화
        ret = 0

        for pred, area in zip(preds, areas):
            # pred: [16, 59, 1]
            # area: [16, 59]
            # area / tot_area: [16, 59]
            # unsqueeze(-1): [16, 59, 1]
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        # final ret: [16, 59, 1]
        return ret

    def correction_layer(self, preds):
        # preds before correction: [16, 59, 1]
        if hasattr(self, 'add_layer'):
            out = self.add_layer(preds)

            # after correction: [16, 59, 1]
            return out

        return preds

    def forward(self, inp, topo, coord, cell):
        # inp:   [16, 1, 50, 50]
        # topo:  [16, 1, 100, 100]
        # coord: [16, 59, 2]
        # cell:  [16, 59, 2]

        # encoder feature 생성
        self.gen_feat(inp)

        # query_rgb 안에서 topo/level을 사용하기 위해 저장
        self.topo = topo

        # LIIF decoder를 통해 query point별 예측
        preds = self.query_rgb(coord, cell)

        # preds: [16, 59, 1]

        # correction layer 적용 후 반환
        return self.correction_layer(preds)

        # output: [16, 59, 1]