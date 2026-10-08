""" Low-Rank Adaptation of Neural Fields (Truong et al.) 를 LTE imnet(MLP) 에 적용하기 위한 유틸.

    h_{l+1} = sigma( W_l h_l + (1/r) B_l A_l h_l + b_l )
      W_l, b_l : 사전학습 값 (freeze)
      A_l      : [r, d_in]  ~ N(0, 1/d_in)   (trainable)
      B_l      : [d_out, r] = 0              (trainable)  -> 학습 시작 시 LoRA 출력 == 원래 출력

    models/liif_lte.py 는 수정하지 않는다. LTE.imnet.layers(nn.Sequential) 안의 nn.Linear 를
    LoRALinear 로 갈아끼우는 방식이라 LTE.forward / query_rgb 는 그대로 동작한다.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        assert isinstance(base, nn.Linear), type(base)
        self.base = base
        self.rank = rank
        self.scale = 1.0 / rank            # 논문 정의: (1/r) B A
        self.enabled = True                # False 면 사전학습 원본 출력 (ablation / 초기 동치 검증용)
        d_out, d_in = base.weight.shape
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.randn(rank, d_in, device=dev, dtype=dt) / math.sqrt(d_in))  # N(0, 1/d_in)
        self.B = nn.Parameter(torch.zeros(d_out, rank, device=dev, dtype=dt))

    def delta_weight(self):
        return self.scale * (self.B @ self.A)          # [d_out, d_in]

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + F.linear(F.linear(x, self.A), self.B) * self.scale
        return y


def _linear_slots(imnet):
    """imnet.layers(nn.Sequential) 안의 (index, nn.Linear) 목록. MLP 순서 그대로 첫 층 -> 출력층."""
    return [(i, m) for i, m in enumerate(imnet.layers) if isinstance(m, nn.Linear)]


def inject_lora(model, rank, layers='all'):
    """model.imnet 의 Linear 에 LoRA 삽입. layers: 'all'(논문 방식) | 'no_first'(첫 층 제외) | 'last'(출력층만, ablation).
    첫 층은 (feature x 상대좌표 modulation) 이 들어오는 층이라, 여기에 LoRA 를 넣으면 좌표 의존 출력을 학습하기 쉬움.

    반환: 삽입된 LoRALinear 리스트. base 가중치는 모두 requires_grad=False 로 고정된다.
    """
    assert layers in ('all', 'no_first', 'last'), layers
    for p in model.parameters():
        p.requires_grad = False

    slots = _linear_slots(model.imnet)
    if layers == 'last':
        slots = slots[-1:]
    elif layers == 'no_first':
        slots = slots[1:]
    out = []
    for i, lin in slots:
        lora = LoRALinear(lin, rank)
        model.imnet.layers[i] = lora
        out.append(lora)
    return out


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def set_lora_enabled(model, enabled):
    for m in lora_modules(model):
        m.enabled = enabled


def lora_state_dict(model):
    """A, B 만 추출 (키는 imnet 내 위치). base 는 원본 체크포인트에서 다시 불러오므로 저장하지 않는다."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.endswith('.A') or k.endswith('.B')}


def load_lora_state_dict(model, sd):
    cur = model.state_dict()
    missing = [k for k in sd if k not in cur]
    if missing:
        raise KeyError(f'LoRA 키가 모델에 없음: {missing}')
    model.load_state_dict(sd, strict=False)


@torch.no_grad()
def merge_lora(model):
    """W <- W + (1/r) B A 로 합치고 LoRALinear 를 원래 nn.Linear 로 되돌린다 (일반 LTE 체크포인트로 export 용)."""
    for i, m in enumerate(model.imnet.layers):
        if isinstance(m, LoRALinear):
            m.base.weight += m.delta_weight()
            model.imnet.layers[i] = m.base
    return model


def build_lora_lte(base_ckpt, rank, layers='all', lora_sd=None, device='cuda'):
    """사전학습 LTE 체크포인트 로드 -> 전부 freeze -> LoRA 삽입 (-> lora_sd 있으면 A,B 복원)."""
    import models   # 순환 import 방지용 지연 import
    sv = torch.load(base_ckpt, map_location='cpu')
    spec = sv['model']
    if spec.get('args', {}).get('layer_spec') is not None:
        raise RuntimeError('체크포인트에 layer_spec(correction layer) 이 있습니다.')
    model = models.make(spec, load_sd=True).to(device)
    base_spec = {k: v for k, v in spec.items() if k != 'sd'}
    inject_lora(model, rank, layers)
    if lora_sd is not None:
        load_lora_state_dict(model, lora_sd)
    return model, base_spec
