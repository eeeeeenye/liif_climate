""" Probabilistic CNP-style residual adapter (frozen LIIF/LTE 위에 얹는 확률적 보정기).

    아이디어
    --------
    사전학습된 LTE 는 그대로 얼려두고(Y0 = LTE(IMERG, x)), ASOS 관측소에서의
    잔차 e_i = y_i^ASOS - Y0(s_i) 를 조건으로 받아, 임의의 target 좌표 x_t 에서
    "보정량의 확률분포" p(dy_t | x_t, C) = N(mu_t, sigma_t^2) 를 출력한다.

        context 점 i : [coord_i(2), Y0(s_i)(1), e_i(1)]   (총 4-d)
            -> encoder MLP -> r_i
            -> (masked) mean 으로 집계 -> r            (Conditional Neural Process)
        target 점 t  : [coord_t(2), Y0(x_t)(1)] ++ r
            -> decoder MLP -> (mu_t, s_t)
            -> sigma_t = softplus(s_t) + min_sigma

    학습은 context/target split + Gaussian NLL (losses_cnp.py) 로 한다.
    inference 는 ASOS 전 지점을 context 로, 전체 격자를 target 으로 넣어
    correction map mu(x) 와 uncertainty map sigma(x) 를 얻는다.

    LTE 의 correction_layer(add_layer) 와는 완전히 별개다. 이 모듈은 LTE 를
    건드리지 않고 밖에서 감싸는 방식이라, 기존 학습 코드/체크포인트에 영향이 없다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register


def _mlp(in_dim, hidden_list, out_dim, act=nn.ReLU):
    layers = []
    last = in_dim
    for h in hidden_list:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


@register('cnp-adapter')
class ProbabilisticCNPAdapter(nn.Module):
    """ CNP 식 확률적 residual adapter.

    Args
    ----
    coord_dim : int          target/context 좌표 차원 (LIIF [-1,1] 좌표, 보통 2)
    y0_dim : int             LTE 출력 채널 수 (강수는 1)
    residual_dim : int       잔차 e 채널 수 (강수는 1)
    enc_hidden : list[int]   encoder MLP hidden 크기 목록
    r_dim : int              context representation r 차원
    dec_hidden : list[int]   decoder MLP hidden 크기 목록
    min_sigma : float        sigma 하한 (softplus 출력에 더함). 학습 공간 단위.
    init_sigma_bias : float  decoder 마지막 층 sigma 출력 bias 초기값.
                             음수로 주면 초기 sigma 가 작게 시작 -> NLL 초기 발산 완화.
    """

    def __init__(self,
                 coord_dim=2, y0_dim=1, residual_dim=1,
                 enc_hidden=(128, 128), r_dim=128,
                 dec_hidden=(128, 128, 128),
                 min_sigma=1e-4, init_sigma_bias=-2.0):
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.y0_dim = int(y0_dim)
        self.residual_dim = int(residual_dim)
        self.r_dim = int(r_dim)
        self.min_sigma = float(min_sigma)

        enc_in = self.coord_dim + self.y0_dim + self.residual_dim
        dec_in = self.coord_dim + self.y0_dim + self.r_dim
        self.encoder = _mlp(enc_in, list(enc_hidden), self.r_dim)
        self.decoder = _mlp(dec_in, list(dec_hidden), 2)

        # sigma head 가 큰 값에서 시작해 서서히 내려오도록 마지막 층 bias 초기화
        with torch.no_grad():
            self.decoder[-1].bias[0].zero_()
            self.decoder[-1].bias[1].fill_(float(init_sigma_bias))

    # ------------------------------------------------------------------
    def aggregate(self, r_i, ctx_mask=None):
        """ r_i: (B, Nc, r_dim) -> r: (B, r_dim).  ctx_mask: (B, Nc) 1/0. """
        if ctx_mask is None:
            return r_i.mean(dim=1)
        w = ctx_mask.unsqueeze(-1).to(r_i.dtype)          # (B, Nc, 1)
        return (r_i * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def encode(self, context, ctx_mask=None):
        """ context: (B, Nc, coord+y0+residual) -> r: (B, r_dim). """
        return self.aggregate(self.encoder(context), ctx_mask)

    def decode(self, target, r):
        """ target: (B, Nt, coord+y0), r: (B, r_dim) -> (mu, sigma) 각각 (B, Nt). """
        B, Nt, _ = target.shape
        r_exp = r.unsqueeze(1).expand(B, Nt, self.r_dim)
        out = self.decoder(torch.cat([target, r_exp], dim=-1))   # (B, Nt, 2)
        mu = out[..., 0]
        sigma = F.softplus(out[..., 1]) + self.min_sigma
        return mu, sigma

    def forward(self, context, target, ctx_mask=None):
        r = self.encode(context, ctx_mask)
        return self.decode(target, r)
