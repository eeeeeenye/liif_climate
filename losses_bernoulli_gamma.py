""" Bernoulli-Gamma 조건부 분포 손실 (Cannon 2008, J. Hydrometeorology 9:1284-1300).

    분포 (식 6):  y = 0          확률 1 - p
                  y > 0          확률 p * Gamma(y; alpha, beta)   (beta = scale)
    평균 (2011 정오표):  mu  = p * alpha * beta
    분산              :  var = p * alpha * beta^2 * (1 + (1 - p) * alpha)
    손실 (식 12, 지점 1개):  NLPD = -log f(y | x)
        dry :  -log(1 - p)
        wet :  -log p + alpha*log(beta) + lgamma(alpha) - (alpha - 1)*log(y) + y/beta
    (공분산/분산 제약, AR(1) 은 지점별 보정에서는 쓰지 않으므로 구현하지 않음)

    단위: y, beta 는 mm/h (물리 단위). 학습 텐서는 [0,1] minmax 이므로 호출 전에
    denorm(gt * (max - min) + min) 해서 넘긴다.

    이산화 주의: ASOS 는 0.1 mm 단위라 y>0 의 최솟값이 0.1. 연속 gamma 밀도를 그대로 쓰되
    wet 은 y >= wet_thr(기본 0.05) 로 판정하고 log(y) 는 y_floor 로 바닥을 둔다.
"""

import math

import torch

LOGIT_MIN, LOGIT_MAX = -8.0, 8.0     # alpha, beta 의 exp 폭주 방지 (alpha, beta in ~[3e-4, 3e3])


def bg_params(raw):
    """ raw [..., 3] 로짓 -> (p, alpha, beta) 각 [...] """
    p = torch.sigmoid(raw[..., 0])
    alpha = torch.exp(raw[..., 1].clamp(LOGIT_MIN, LOGIT_MAX))
    beta = torch.exp(raw[..., 2].clamp(LOGIT_MIN, LOGIT_MAX))
    return p, alpha, beta


def bg_mean(raw):
    p, a, b = bg_params(raw)
    return p * a * b


def bg_var(raw):
    p, a, b = bg_params(raw)
    return p * a * b * b * (1.0 + (1.0 - p) * a)


def bg_nll_elem(p, alpha, beta, y, wet_thr=0.05, y_floor=0.05, eps=1e-6):
    """ 원소별 NLPD. y (mm/h) 와 같은 shape 반환. (wet_mask 도 함께 반환) """
    wet = y >= wet_thr
    ys = y.clamp(min=y_floor)
    nll_dry = -torch.log((1.0 - p).clamp(min=eps))
    nll_wet = (-torch.log(p.clamp(min=eps))
               + alpha * torch.log(beta) + torch.lgamma(alpha)
               - (alpha - 1.0) * torch.log(ys) + ys / beta)
    return torch.where(wet, nll_wet, nll_dry), wet


def bg_nll(raw, y, mask=None, wet_thr=0.05, y_floor=0.05):
    """ raw [B,q,3], y [B,q,1] 또는 [B,q] (mm/h), mask [B,q] (1=유효). 유효 지점 평균 NLPD. """
    if y.dim() == raw.dim():
        y = y.squeeze(-1)
    p, a, b = bg_params(raw)
    nll, _ = bg_nll_elem(p, a, b, y, wet_thr, y_floor)
    if mask is None:
        return nll.mean()
    return (nll * mask).sum() / mask.sum().clamp(min=1.0)


def bg_sample(raw, generator=None):
    """ 조건부 분포에서 표본 추출 (mm/h). Bernoulli 로 wet 결정 -> Gamma 표본. """
    p, a, b = bg_params(raw)
    wet = torch.rand(p.shape, device=p.device, generator=generator) < p
    g = torch._standard_gamma(a.expand_as(a))            # Gamma(alpha, scale=1)
    return torch.where(wet, g * b, torch.zeros_like(g))


def fit_climatology(n, n_wet, sum_y_wet, sum_logy_wet, y_floor=0.05):
    """ 무조건부 Bernoulli-Gamma MLE (Cannon 의 기준선). 충분통계량으로부터 (p, alpha, beta).
        alpha: log(alpha) - digamma(alpha) = log(mean) - mean(log y)  를 Newton 으로 푼다. """
    p = n_wet / n
    mean = sum_y_wet / n_wet
    s = math.log(mean) - sum_logy_wet / n_wet
    a = (3 - s + math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)   # Minka 초기값
    for _ in range(20):
        at = torch.tensor(a, dtype=torch.float64)
        f = math.log(a) - torch.digamma(at).item() - s
        fp = 1.0 / a - torch.polygamma(1, at).item()
        a = max(a - f / fp, 1e-3)
    return p, a, mean / a
