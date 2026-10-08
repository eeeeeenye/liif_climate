"""[새 파일] 문맥(context)을 보는 어댑터를 단 LTE:  모델 이름 'lte_ctx'.
원본 models/liif_lte.py 는 수정하지 않고 LTE 를 상속해 query_rgb/forward 만 다시 정의한다.
이 모듈을 import 하면 models registry 에 'lte_ctx' 가 등록된다 (models/__init__.py 를 고치지 않으려고 학습/평가 스크립트가 직접 import).

배경: 기존 어댑터는 방향별 imnet 출력 p_k '숫자 하나'만 보고 값을 바꾼다. 같은 p_k 라도 ASOS 가 0 인 시각과 비가 오는 시각이 섞여 있어
      구분할 정보가 없고, 평균 쪽으로 타협해 약한 비는 과대, 강한 비는 과소가 된다(실험 노트 참고).
      -> 방향별 어댑터 입력에 '주변 정보'를 더한다.

방향 k 의 어댑터 입력 (모두 정규화 공간 [0,1], in_scale 배):
   p_k            : 이 방향의 imnet 출력
   mean4/max4/min4: 4방향 imnet 출력의 평균/최대/최소  (방향 간 일관성: 모두 크면 확실한 비)
   lr             : 저해상도 입력의 bilinear 값 (질의 위치의 HSR)
   lrmax          : 저해상도 입력의 ctx_win x ctx_win 최대값 (주변에 비가 있는지)
출력:  g_k = p_k + f([입력])   (f = MLP, 마지막 Linear 0 초기화 -> 학습 시작 시점은 어댑터 없는 사전학습 출력과 같음)
최종:  ret = sum_k w_k g_k + lr   (원본 LTE adapter_per_direction=True 와 같은 결합 방식)
모듈 이름을 add_layer.asos_adapter.* 로 맞춰서 기존 학습 코드의 freeze/체크포인트 규약과 호환된다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register
from models.liif_lte import LTE

N_FEAT = 6


class CtxAdapter(nn.Module):
    def __init__(self, in_dim=N_FEAT, hidden=(64, 32)):
        super().__init__()
        dims = [in_dim] + list(hidden); layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        last = nn.Linear(dims[-1], 1); nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        layers.append(last)
        self.asos_adapter = nn.Sequential(*layers)

    def forward(self, feats):                         # feats [bs, q, 4, F] -> 잔차 [bs, q, 4, 1]
        return self.asos_adapter(feats)


@register('lte_ctx')
class LTEContext(LTE):
    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256, local_ensemble=True, cell_decode=True,
                 extra_in_dim=0, ctx_hidden=(64, 32), ctx_win=3, in_scale=10.0, **unused):
        super().__init__(encoder_spec, imnet_spec, layer_spec=None, hidden_dim=hidden_dim, local_ensemble=local_ensemble,
                         cell_decode=cell_decode, extra_in_dim=extra_in_dim, adapter_per_direction=False)
        assert local_ensemble, 'lte_ctx 는 4방향 local ensemble 을 전제로 함'
        self.ctx_win, self.in_scale = int(ctx_win), float(in_scale)
        self.add_layer = CtxAdapter(N_FEAT, ctx_hidden)

    def _context_feats(self, preds, coord):
        """preds [bs,q,4] 방향별 imnet 출력. 반환: feats [bs,q,4,F], lr [bs,q,1]"""
        grid = coord.flip(-1).unsqueeze(1)
        lr = F.grid_sample(self.inp, grid, mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        lrmax = F.grid_sample(F.max_pool2d(self.inp, self.ctx_win, stride=1, padding=self.ctx_win // 2), grid, mode='nearest',
                              padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        mean4, max4, min4 = preds.mean(-1, keepdim=True), preds.max(-1, keepdim=True)[0], preds.min(-1, keepdim=True)[0]
        ctx = torch.cat([mean4, max4, min4, lr, lrmax], dim=-1).unsqueeze(2).expand(-1, -1, 4, -1)    # [bs,q,4,5]
        feats = torch.cat([preds.unsqueeze(-1), ctx], dim=-1) * self.in_scale                          # [bs,q,4,6]
        return feats, lr

    def query_rgb(self, coord, cell=None):
        super().query_rgb(coord, cell)                                    # 부모: 방향별 원 imnet 출력 self.le_preds / 가중치 self.le_weights 를 채운다 (adapter_per_direction=False 라 어댑터 미적용)
        preds = torch.cat(self.le_preds, dim=-1)                          # [bs,q,4]  p_k
        w = torch.stack(self.le_weights, dim=-1)                          # [bs,q,4]
        feats, lr = self._context_feats(preds, coord)
        g = preds + self.add_layer(feats).squeeze(-1)                     # [bs,q,4]  g_k
        self.le_g = g
        return (g * w).sum(-1, keepdim=True) + lr

    def forward(self, inp, topo, coord, cell):
        self.gen_feat(inp)
        self.topo = topo
        out = self.query_rgb(coord, cell)
        self.preds_before_adapter = out
        return out
