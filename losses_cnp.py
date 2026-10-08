""" CNP adapter 학습용 확률 손실 (Gaussian Negative Log-Likelihood).

    p(dy_t | x_t, C) = N(mu_t, sigma_t^2) 을 학습시키므로 단순 MSE 가 아니라 NLL 을 쓴다.

        L_NLL = mean_t [ 0.5 * ( (dy_t - mu_t)^2 / sigma_t^2 + log sigma_t^2 ) ]

    (상수항 0.5*log(2*pi) 는 최적화에 영향이 없어 생략.)

    beta-NLL (Seitzer et al., 2022, "On the Pitfalls of Heteroscedastic
    Uncertainty Estimation with Probabilistic Neural Networks") 옵션도 제공한다.
    각 항에 sigma_t^(2*beta) (stop-grad) 를 곱해, NLL 이 저잔차 지점에만 몰려
    분산 추정이 무너지는 것을 완화한다. beta=0 이면 순수 NLL, beta=1 이면 사실상
    가중 MSE. 기본값 0.0.
"""

import torch
import torch.nn as nn


def gaussian_nll(mu, sigma, target, beta=0.0, eps=1e-8, reduction='mean'):
    """ mu, sigma, target : 같은 shape. sigma > 0 (이미 softplus+min_sigma 통과). """
    var = (sigma ** 2).clamp_min(eps)
    nll = 0.5 * ((target - mu) ** 2 / var + torch.log(var))
    if beta and beta > 0.0:
        nll = nll * var.detach() ** beta
    if reduction == 'mean':
        return nll.mean()
    if reduction == 'sum':
        return nll.sum()
    return nll


class GaussianNLL(nn.Module):
    """ forward(mu, sigma, target) -> scalar. """

    def __init__(self, beta=0.0, eps=1e-8):
        super().__init__()
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, mu, sigma, target):
        return gaussian_nll(mu, sigma, target, beta=self.beta, eps=self.eps)

    def extra_repr(self):
        return f'beta={self.beta}, eps={self.eps}'


_LOSSES = {
    'gaussian_nll': GaussianNLL,
}


def make_loss(loss_spec):
    """ yaml 의 loss 블록 -> loss 객체.

        loss:
          name: gaussian_nll
          args: {beta: 0.0}
    """
    name = loss_spec['name']
    if name not in _LOSSES:
        raise NotImplementedError(
            f"losses_cnp.make_loss: 알 수 없는 loss '{name}'. 사용 가능: {list(_LOSSES)}")
    return _LOSSES[name](**(loss_spec.get('args', {}) or {}))
