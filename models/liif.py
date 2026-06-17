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
        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2
            if self.cell_decode:
                imnet_in_dim += 2
            imnet_in_dim += extra_in_dim
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

        if layer_spec is not None:
            adapter_in_dim = self.imnet.out_dim
            self.add_layer = models.make(layer_spec, args={'in_dim': adapter_in_dim})
    
    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat                        # 16, 256, 100, 100
        level = self.topo[:,:1,...]           # 16, 3, 100, 100

        device = feat.device

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # feat.shape : (16, 64, 30, 30) -> unfold (16, 576, 30, 30)

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda(device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # hr coord 샘플링한 좌표를 가져와서 clone
                coord_ = coord.clone() # (16 ,1024, 2)

                # hr coord를 feat의 중간 좌표에 맞게 shift -> 중간좌표 조정
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                # print(coord_.flip(-1).unsqueeze(1).shape)

                # flip 함수는 coord의 (B, sample_q, 2) 에서 2는 좌표(y,x)를 의미하는데 이 좌표의 순서를 (x,y)로 바꿔줌
                # unsqueeze(1)은 1번 인덱스에 축을 하나 추가해서 H=1, W=1024를 만드는 작업
                # coord_의 shape가 16, 1, 1024, 2로 되어있어서, H가 1인 한줄짜리 1024쌍의 (x,y) 상대좌표를 가진 텐서로 인식하고 자동으로 grid 위치랑 매핑하여, feat에서 맞는 grid 위치에 있는 데이터만을 뽑아옴 

                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),  # coord_ shape: 16, 1, 1024, 2, feat shape: 16, 567, 30, 30
                    mode='nearest', align_corners=False)[:,:,0,:] \
                    .permute(0, 2, 1)
                    
                # print("q_feat_shape",q_feat.shape)                   # torch.size([16, 1024, 576])
                
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                q_level = F.grid_sample(level, coord_.flip(-1).unsqueeze(1),
                             mode="nearest", align_corners=False)[:, :, 0, :]\
                            .permute(0,2,1)
                
                # print(level.shape)
                # level = level.permute(0,2,1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord, q_level], dim=-1) # q_feat : [16, 1024, 2304], [16, 1024,2], [16, 1024, 1]
                # print(inp.shape)                 # 16, 1024, 2307

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)
                
                bs, q = coord.shape[:2]
                # print(inp.view(bs * q, -1).shape) # 16384, 2309

                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)
                # print("pred shape : ",pred.shape)                 # 16, 1024, 2

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        """ 상대 좌표는 쿼리 좌표(q)와 기준 좌표(q_coord)의 차이를 통해,
         복원하려는 위치가 셀 내부에서 어디쯤 있는지를 나타냅니다.
         이 좌표의 x, y 성분을 곱하면 면적에 해당하는 값이 되어,
         bilinear weight 계산에 쓰이는 기여도를 표현할 수 있습니다.
         LIIF 논문에서 말하는 signal이 바로 이 상대 좌표(rel_coord)입니다.
         즉, 쿼리별 signal을 4개 꼭짓점 feature와 결합하여 후보 예측값을 만들고,
         쿼리 좌표의 위치에 따라 bilinear weight로 합성하여 최종 HR 값을 예측합니다. """

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0

        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret
    
    # adapter layer 추가
    def correction_layer(self, preds):
        if hasattr(self, 'add_layer'):
            # print(preds.shape)
            return self.add_layer(preds)
        return preds

    def forward(self, inp, topo, coord, cell):  #, area=None
        self.gen_feat(inp)
        self.topo = topo
        preds = self.query_rgb(coord, cell)
        # print(preds.shape)

        return self.correction_layer(preds)
