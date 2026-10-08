""" Attentive CNP(ANP) 식 확률적 residual adapter.

    models/cnp_adapter.py 의 mean aggregation 을 cross-attention 으로 바꾼 버전.
    target 좌표마다 "가까운/관련 있는 context 관측소" 에 더 큰 가중치를 주어
    국소적인 보정을 하도록 한다 (mean 은 모든 target 에 같은 r 하나만 줬음).

        value  v_i = f_v([coord_i, Y0(s_i), e_i])            (context, 잔차 포함)
        key    k_i = f_k([coord_i, Y0(s_i)])                 (context, 위치 임베딩)
        query  q_t = f_q([coord_t, Y0(x_t)])                 (target, 위치 임베딩)
        r*_t   = MultiHeadAttention(q_t, k_i, v_i)           (target 별 representation)
        (opt)  r_mean = mean_i v_i                           (전역 문맥, use_global)
        decoder([coord_t, Y0(x_t), r*_t (, r_mean)]) -> (mu_t, s_t)
        sigma_t = softplus(s_t) + min_sigma
        p(dy_t | x_t, C) = N(mu_t, sigma_t^2)

    학습/추론 인터페이스는 cnp-adapter 와 동일: forward(context, target, ctx_mask=None)
    -> (mu, sigma). train_anp_adapter.py / infer_prob_adapter.py 에서 그대로 쓴다.
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


@register('anp-adapter')
class AttentiveCNPAdapter(nn.Module):
    """ ANP 식 확률적 residual adapter.

    Args
    ----
    coord_dim, y0_dim, residual_dim : 입력 채널 (강수는 2 / 1 / 1)
    enc_hidden : list[int]   value encoder(f_v) hidden
    r_dim : int              value / representation 차원 (= attention embed_dim)
    kq_hidden : list[int]    key/query encoder(f_k, f_q) hidden
    n_heads : int            MultiHeadAttention head 수 (r_dim % n_heads == 0)
    dec_hidden : list[int]   decoder MLP hidden
    use_global : bool        전역 mean(v_i) 를 decoder 입력에 추가로 concat
    min_sigma : float        sigma 하한
    init_sigma_bias : float  decoder 마지막층 sigma bias 초기값 (음수 -> 초기 sigma 축소)
    """

    def __init__(self,
                 coord_dim=2, y0_dim=1, residual_dim=1,
                 enc_hidden=(128, 128), r_dim=128,
                 kq_hidden=(128,), n_heads=4,
                 dec_hidden=(128, 128, 128),
                 use_global=True,
                 min_sigma=1e-4, init_sigma_bias=-2.0):
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.y0_dim = int(y0_dim)
        self.residual_dim = int(residual_dim)
        self.r_dim = int(r_dim)
        self.use_global = bool(use_global)
        self.min_sigma = float(min_sigma)

        loc_in = self.coord_dim + self.y0_dim                 # [coord, Y0]
        val_in = loc_in + self.residual_dim                   # [coord, Y0, e]

        self.value_encoder = _mlp(val_in, list(enc_hidden), self.r_dim)
        self.key_encoder = _mlp(loc_in, list(kq_hidden), self.r_dim)
        self.query_encoder = _mlp(loc_in, list(kq_hidden), self.r_dim)
        self.attn = nn.MultiheadAttention(self.r_dim, int(n_heads), batch_first=True)

        dec_in = loc_in + self.r_dim + (self.r_dim if self.use_global else 0)
        self.decoder = _mlp(dec_in, list(dec_hidden), 2)

        with torch.no_grad():
            self.decoder[-1].bias[0].zero_()
            self.decoder[-1].bias[1].fill_(float(init_sigma_bias))

    # ------------------------------------------------------------------
    def forward(self, context, target, ctx_mask=None):
        """ context: (B, Nc, coord+y0+residual),  target: (B, Nt, coord+y0)
            ctx_mask: (B, Nc) 1=유효 / 0=padding (없으면 전부 유효). """
        ce = self.coord_dim + self.y0_dim
        ctx_loc = context[..., :ce]

        v = self.value_encoder(context)            # (B, Nc, r_dim)
        k = self.key_encoder(ctx_loc)              # (B, Nc, r_dim)
        q = self.query_encoder(target)             # (B, Nt, r_dim)

        key_padding_mask = None
        if ctx_mask is not None:
            key_padding_mask = ~ctx_mask.bool()    # True = 무시

        r_star, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask,
                              need_weights=False)  # (B, Nt, r_dim)

        dec_in = [target, r_star]
        if self.use_global:
            if ctx_mask is None:
                r_mean = v.mean(dim=1)
            else:
                w = ctx_mask.unsqueeze(-1).to(v.dtype)
                r_mean = (v * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
            dec_in.append(r_mean.unsqueeze(1).expand(-1, target.shape[1], -1))

        out = self.decoder(torch.cat(dec_in, dim=-1))   # (B, Nt, 2)
        mu = out[..., 0]
        sigma = F.softplus(out[..., 1]) + self.min_sigma
        return mu, sigma
