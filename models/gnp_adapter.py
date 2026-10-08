""" 거리 게이트(Gaussian-gated) attention 버전의 확률적 residual adapter.

    models/anp_adapter.py 의 attention 이 절대좌표만 봐서 관측소 구름 밖(코너/해상/북한)
    에서 "임베딩상 제일 가까운 관측소 하나"에 몰빵 -> Voronoi 쐐기 + 극단 외삽하는 문제를
    고친다. 핵심 3가지:

      1) 거리 게이트 : attention logit 에 -||coord_t - coord_i||^2 / (2*ell_h^2) 를 더한다.
         (ell_h = head 별 학습 가능 length scale). target 이 모든 context 에서 멀면
         실제 관측소 logit 이 다 같이 깔려서 attention 이 평평해짐.
      2) null token : 거리 페널티가 없는 학습 가능한 (key, value) 한 쌍을 항상 붙인다.
         관측소가 다 멀면 softmax 가 이 null 로 쏠려 "학습된 prior(보통 보정≈0)" 로 폴백.
      3) density feature : rho_t = log(1 + sum_i exp(-d^2/2ell^2)) 를 decoder 에 넣어
         "주변에 관측이 얼마나 있나" 를 알려준다 -> 데이터 희박한 곳에서 mu 축소 / sigma 확대.

    인터페이스는 cnp/anp-adapter 와 동일: forward(context, target, ctx_mask=None) -> (mu, sigma).
"""

import math

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


@register('gnp-adapter')
class GatedNPAdapter(nn.Module):
    """ 거리 게이트 attention + null token + density feature.

    Args
    ----
    coord_dim, y0_dim, residual_dim : 입력 채널 (2 / 1 / 1)
    enc_hidden : value encoder f_v hidden
    r_dim : value/representation 차원 (= attention embed_dim, r_dim % n_heads == 0)
    kq_hidden : key/query encoder hidden
    n_heads : head 수
    dec_hidden : decoder MLP hidden
    length_scale_init : head 별 게이트 length scale 초기값 (LIIF [-1,1] 좌표 단위).
                        0.3 ~ 도메인 폭(2)의 15% ~ 위도 1.5도.
    use_null : null token 사용 (권장 True)
    use_density : density feature 를 decoder 입력에 concat
    min_sigma, init_sigma_bias : cnp-adapter 와 동일
    """

    def __init__(self,
                 coord_dim=2, y0_dim=1, residual_dim=1,
                 enc_hidden=(128, 128), r_dim=128,
                 kq_hidden=(128,), n_heads=4,
                 dec_hidden=(128, 128, 128),
                 length_scale_init=0.3, use_null=True, use_density=True,
                 min_sigma=1e-4, init_sigma_bias=-2.0):
        super().__init__()
        assert r_dim % n_heads == 0
        self.coord_dim = int(coord_dim)
        self.y0_dim = int(y0_dim)
        self.residual_dim = int(residual_dim)
        self.r_dim = int(r_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.r_dim // self.n_heads
        self.use_null = bool(use_null)
        self.use_density = bool(use_density)
        self.min_sigma = float(min_sigma)

        loc_in = self.coord_dim + self.y0_dim
        val_in = loc_in + self.residual_dim

        self.value_encoder = _mlp(val_in, list(enc_hidden), self.r_dim)
        self.key_encoder = _mlp(loc_in, list(kq_hidden), self.r_dim)
        self.query_encoder = _mlp(loc_in, list(kq_hidden), self.r_dim)
        self.out_proj = nn.Linear(self.r_dim, self.r_dim)

        # head 별 length scale : softplus(param) = ell
        inv = math.log(math.expm1(float(length_scale_init)))
        self.ls_param = nn.Parameter(torch.full((self.n_heads,), inv))

        if self.use_null:
            self.null_k = nn.Parameter(torch.randn(self.r_dim) * 0.02)
            self.null_v = nn.Parameter(torch.zeros(self.r_dim))

        dec_in = loc_in + self.r_dim + (1 if self.use_density else 0)
        self.decoder = _mlp(dec_in, list(dec_hidden), 2)
        with torch.no_grad():
            self.decoder[-1].bias[0].zero_()
            self.decoder[-1].bias[1].fill_(float(init_sigma_bias))

    # ------------------------------------------------------------------
    def forward(self, context, target, ctx_mask=None):
        """ context: (B, Nc, coord+y0+residual),  target: (B, Nt, coord+y0). """
        B, Nc, _ = context.shape
        Nt = target.shape[1]
        H, hd = self.n_heads, self.head_dim
        ce = self.coord_dim + self.y0_dim

        coord_c = context[..., :self.coord_dim]        # (B, Nc, 2)
        coord_t = target[..., :self.coord_dim]         # (B, Nt, 2)

        v = self.value_encoder(context)                # (B, Nc, r)
        k = self.key_encoder(context[..., :ce])        # (B, Nc, r)
        q = self.query_encoder(target)                 # (B, Nt, r)

        qh = q.view(B, Nt, H, hd).transpose(1, 2)      # (B, H, Nt, hd)
        kh = k.view(B, Nc, H, hd).transpose(1, 2)      # (B, H, Nc, hd)
        vh = v.view(B, Nc, H, hd).transpose(1, 2)

        # 쌍별 제곱거리 (좌표 공간)
        d2 = ((coord_t[:, :, None, :] - coord_c[:, None, :, :]) ** 2).sum(-1)   # (B, Nt, Nc)

        ell = F.softplus(self.ls_param).view(1, H, 1, 1).clamp_min(1e-3)
        logits = (qh @ kh.transpose(-1, -2)) / math.sqrt(hd)                    # (B, H, Nt, Nc)
        logits = logits - d2.unsqueeze(1) / (2.0 * ell ** 2)                    # 거리 게이트

        if ctx_mask is not None:
            m = ctx_mask.bool()[:, None, None, :]                              # (B,1,1,Nc)
            logits = logits.masked_fill(~m, float('-inf'))

        if self.use_null:
            nk = self.null_k.view(1, H, 1, hd).expand(B, H, 1, hd)
            nv = self.null_v.view(1, H, 1, hd).expand(B, H, 1, hd)
            null_logit = (qh @ nk.transpose(-1, -2)) / math.sqrt(hd)           # (B,H,Nt,1) : 거리 페널티 없음
            logits = torch.cat([logits, null_logit], dim=-1)                   # (B,H,Nt,Nc+1)
            vh = torch.cat([vh, nv], dim=2)                                    # (B,H,Nc+1,hd)

        attn = torch.softmax(logits, dim=-1)
        r_star = (attn @ vh).transpose(1, 2).reshape(B, Nt, self.r_dim)        # (B, Nt, r)
        r_star = self.out_proj(r_star)

        dec_in = [target, r_star]
        if self.use_density:
            ell0 = F.softplus(self.ls_param).mean().clamp_min(1e-3)
            w = torch.exp(-d2 / (2.0 * ell0 ** 2))                            # (B, Nt, Nc)
            if ctx_mask is not None:
                w = w * ctx_mask[:, None, :].to(w.dtype)
            dec_in.append(torch.log1p(w.sum(-1, keepdim=True)))               # (B, Nt, 1)

        out = self.decoder(torch.cat(dec_in, dim=-1))
        mu = out[..., 0]
        sigma = F.softplus(out[..., 1]) + self.min_sigma
        return mu, sigma
