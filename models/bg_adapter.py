import math

import torch
import torch.nn as nn

from models import register


@register("bg-adapter")
class BernoulliGammaAdapter(nn.Module):
    """ LIIF 출력(1채널) -> Bernoulli-Gamma 파라미터 로짓 3개 (Cannon 2008 BDN 출력층).

    기존 models/adapter.py(ASOSAdapter) 와 같은 MLP 구조(in_dim-hidden...-out)이고 출력만 3개.
        raw[..., 0] = o1 -> p     = sigmoid(o1)   (강수 확률)
        raw[..., 1] = o2 -> alpha = exp(o2)       (gamma shape)
        raw[..., 2] = o3 -> beta  = exp(o3)       (gamma scale, mm/h)
    변환(sigmoid/exp)은 losses_bernoulli_gamma.bg_params 에서 한다.  이 모듈은 raw 로짓만 반환.
    LIIF 안에서 local ensemble 전에 방향별로 적용되므로, 최종 raw 는 방향별 로짓의 면적 가중합
    (= 로짓 공간 blend) 이다.

    입력 x 는 [0,1] minmax 공간의 LIIF 출력. in_scale(= physical_norm.max) 을 곱해 mm 로 되돌린 뒤
    in_transform='log1p' 면 log1p(relu(mm)) 을 MLP 에 넣는다(0 근처에 몰린 입력을 펴주기 위함).
    in_transform='none' 이면 x 를 그대로 넣는다.

    초기화: 마지막 층 weight 를 작게, bias 를 [logit(init_p), log(init_alpha), log(init_beta)] 로 두어
    학습 시작 시 분포가 기후학적 값 근처에서 출발하게 한다.
    """

    def __init__(self, in_dim, out_dim=3, hidden_dims=(32, 16, 32),
                 in_scale=74.8784, in_transform='log1p',
                 init_p=0.1, init_alpha=1.0, init_beta=1.0):
        super().__init__()
        assert out_dim == 3, 'bg-adapter 출력은 (p, alpha, beta) 3개 고정'
        assert in_dim == 1, 'bg-adapter 입력은 LIIF 출력 1채널 가정'
        assert in_transform in ('log1p', 'none')
        self.in_scale = float(in_scale)
        self.in_transform = in_transform

        dims = [in_dim] + list(hidden_dims)
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU())
        last = nn.Linear(dims[-1], 3)
        with torch.no_grad():
            last.weight.mul_(0.1)
            last.bias.copy_(torch.tensor([
                math.log(init_p / (1.0 - init_p)),
                math.log(init_alpha),
                math.log(init_beta),
            ]))
        layers.append(last)
        self.asos_adapter = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        if self.in_transform == 'log1p':
            x = torch.log1p(torch.relu(x) * self.in_scale)
        return self.asos_adapter(x).view(*shape, 3)
