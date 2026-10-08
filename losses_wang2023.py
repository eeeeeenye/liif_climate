""" Customized loss functions from Wang, Tian & Carroll (2023).

    Wang, F., Tian, D., & Carroll, M. (2023).
    Customized deep learning for precipitation bias correction and downscaling.
    Geoscientific Model Development, 16, 535-556.
    https://doi.org/10.5194/gmd-16-535-2023

    구현 범위:
        - Section 3.1.3 (Customized loss functions), Eq. (1): Weighted MAE

    핵심 아이디어
    -------------
    강수는 값의 분포가 심하게 왼쪽으로 치우쳐(0 근처에 집중) 있어서, 일반 MAE/MSE로
    학습하면 모델이 "약한 비 / 무강수"만 잘 맞추고 호우(tail)를 과소모의한다.
    Wang et al. (2023)은 손실의 각 항에 "정답 강수값 자체를 [MIN, MAX]로 클리핑한 값"을
    가중치로 곱해, 강수량이 클수록 그 픽셀의 오차에 더 큰 페널티를 준다.

        L_MAE_weighted = (1 / n) * sum_i ( w1_i * | y_pred_i - y_true_i | )

        w1_i = clip(y_true_i, MIN, MAX)
             = MIN                if y_true_i <= MIN
             = y_true_i           if MIN < y_true_i < MAX
             = MAX                if y_true_i >= MAX

    이 파일은 이 프로젝트의 다른 학습 스크립트가 nn.L1Loss() 를 쓰던 자리에
    그대로 바꿔 끼울 수 있는 형태(nn.Module, reduction='mean' 기본)로 제공한다.

    단위(space)에 대한 주의
    ----------------------
    논문에서 y_true, MIN, MAX 는 모두 물리 단위(mm/day)다. 따라서 가장 논문에
    충실한 사용법은 pred/gt 를 mm/day 로 되돌린 뒤 이 손실을 적용하고 MIN/MAX 도
    mm/day 로 주는 것이다.  다만 w1 이 y_true 를 클리핑한 값이므로 손실 전체가
    (MIN ~ MAX) 배 정도 커진다 -> learning rate 를 그만큼 낮추거나
    normalize_by_weight=True (아래) 로 sum(w1) 나눗셈을 쓰면 스케일이 완화된다.

    정규화([0,1]) 공간에서 쓰고 싶으면 MIN/MAX 도 [0,1] 스케일로 주면 된다.
    이 모듈 자체는 단위에 무관하며, 넘겨받은 y_true 를 w_min/w_max 로 클리핑할 뿐이다.
"""

import torch
import torch.nn as nn


def weighted_mae(pred, target, w_min, w_max,
                 normalize_by_weight=False, eps=1e-8):
    """ Wang et al. (2023) Eq. (1) 의 함수형 구현.

    Args:
        pred, target : 같은 shape 의 tensor. (단위는 호출자가 맞춰서 넘김)
        w_min, w_max : Eq. (1) 의 MIN, MAX. target 과 같은 단위.
        normalize_by_weight :
            False -> 논문 그대로 (1/n) * sum(w * |e|)  (n = 원소 개수)
            True  -> sum(w * |e|) / sum(w)  (가중 평균, 손실 스케일 안정화용)
        eps : normalize_by_weight=True 일 때 0 나눗셈 방지.
    """
    # w1 = clip(y_true, MIN, MAX)  -- 정답 강수값 자체를 가중치로 사용
    w = torch.clamp(target.detach(), min=w_min, max=w_max)

    abs_err = torch.abs(pred - target)
    weighted = w * abs_err

    if normalize_by_weight:
        return weighted.sum() / (w.sum() + eps)
    return weighted.mean()


class WeightedMAELoss(nn.Module):
    """ Wang et al. (2023), Section 3.1.3, Eq. (1) : Weighted MAE.

    nn.L1Loss() 자리에 그대로 대체 가능. 기본 reduction 은 논문과 동일한 'mean'.

    Args:
        w_min (float) : Eq. (1) 의 MIN (가중치 하한, target 단위).
        w_max (float) : Eq. (1) 의 MAX (가중치 상한, target 단위).
        normalize_by_weight (bool) : True 면 (1/n) 대신 (1/sum(w)) 로 나눔.
        clamp_pred (tuple|None) : 필요 시 pred 를 이 범위로 clamp 후 손실 계산.
                                  (예: 정규화 공간이면 (0.0, 1.0))  기본 None.
    """

    def __init__(self, w_min, w_max,
                 normalize_by_weight=False, clamp_pred=None):
        super().__init__()
        assert w_max > w_min, f'w_max({w_max}) 는 w_min({w_min}) 보다 커야 함'
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.normalize_by_weight = bool(normalize_by_weight)
        self.clamp_pred = clamp_pred

    def forward(self, pred, target):
        if self.clamp_pred is not None:
            pred = pred.clamp(self.clamp_pred[0], self.clamp_pred[1])
        return weighted_mae(
            pred, target,
            w_min=self.w_min, w_max=self.w_max,
            normalize_by_weight=self.normalize_by_weight,
        )

    def extra_repr(self):
        return (f'w_min={self.w_min}, w_max={self.w_max}, '
                f'normalize_by_weight={self.normalize_by_weight}, '
                f'clamp_pred={self.clamp_pred}')


# ---------------------------------------------------------------------------
# config -> loss 객체 헬퍼
# ---------------------------------------------------------------------------
_LOSSES = {
    'weighted_mae': WeightedMAELoss,
}


def make_loss(loss_spec):
    """ yaml config 의 loss 블록을 받아 loss 객체를 생성.

        loss:
          name: weighted_mae
          args:
            w_min: 0.1
            w_max: 30.0
            normalize_by_weight: false
    """
    name = loss_spec['name']
    if name not in _LOSSES:
        raise NotImplementedError(
            f"losses_wang2023.make_loss: 알 수 없는 loss '{name}'. "
            f"사용 가능: {list(_LOSSES)}")
    args = loss_spec.get('args', {}) or {}
    return _LOSSES[name](**args)
