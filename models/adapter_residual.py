import torch
import torch.nn as nn
from models import register


@register("adapter_res")
class ASOSAdapterResidual(nn.Module):
    """항등에서 시작하는 잔차형 어댑터:  out = x + f(x),  f = in-h1-h2-...-out MLP(ReLU), 마지막 Linear 를 0 으로 초기화.
    -> 학습 시작 시점의 출력 = 사전학습 LIIF 출력 그대로 (기존 adapter 는 랜덤 초기화라 시작점이 임의 함수였음).
       무강수 라벨이 대부분(~80%)인 손실에서 "전부 0" 으로 무너지는 것을 막기 위함.
    zero_anchor=True 이면 out = x + f(x) - f(0): 입력 0 은 항상 0 (무강수 픽셀에 오프셋이 생길 수 없음). 기본 False (데이터가 배우게 둠).
    state_dict 키는 기존 adapter 와 같은 asos_adapter.* 를 쓴다 (학습 대상 파라미터 필터 호환)."""

    def __init__(self, in_dim, out_dim, hidden_dims=None, zero_anchor=False):
        super().__init__()
        assert in_dim == out_dim, "잔차형이라 in_dim == out_dim 이어야 함"
        dims = [in_dim] + list(hidden_dims or [16])
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        last = nn.Linear(dims[-1], out_dim)
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        layers.append(last)
        self.asos_adapter = nn.Sequential(*layers)
        self.zero_anchor = zero_anchor

    def forward(self, x):
        shape = x.shape[:-1]
        x2 = x.view(-1, x.shape[-1])
        out = x2 + self.asos_adapter(x2)
        if self.zero_anchor:
            out = out - self.asos_adapter(torch.zeros_like(x2[:1]))
        return out.view(*shape, -1)
