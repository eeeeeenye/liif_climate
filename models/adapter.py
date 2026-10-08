import torch.nn as nn                    # PyTorch 신경망 모듈 사용
from models import register              # 모델 registry에 adapter를 등록하기 위한 함수
import torch

@register("adapter")                     # config에서 name: adapter로 호출할 수 있도록 등록
class ASOSAdapter(nn.Module):            # ASOS 보정을 위한 adapter layer 정의
    def __init__(self, in_dim, out_dim, hidden_dims=None):  # 입력/출력 차원 + (선택) hidden layer 크기 목록
        # in_dim : 1
        # out_dim : 1
        # hidden_dims: None/[]이면 예전 체크포인트와 호환되는 단일 Linear(bias=False).
        #              예: [32,16,32]를 주면 in_dim-32-16-32-out_dim MLP(ReLU)로 생성.
        #              체크포인트마다 학습 당시 구조가 다르므로, 로드할 체크포인트에 맞는
        #              hidden_dims를 config(layer_spec.args)나 build_and_load_model 호출부에서 지정할 것.
        super().__init__()               # nn.Module 초기화

        if not hidden_dims:
            self.asos_adapter = nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=False),
            )
        else:
            dims = [in_dim] + list(hidden_dims)
            layers = []
            for d_in, d_out in zip(dims[:-1], dims[1:]):
                layers.append(nn.Linear(d_in, d_out))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(dims[-1], out_dim))
            self.asos_adapter = nn.Sequential(*layers)

    def forward(self, x):                # forward 연산 정의
        shape = x.shape[:-1]             # 마지막 차원을 제외한 기존 shape 저장
        x = self.asos_adapter(
            x.view(-1, x.shape[-1])      # Linear layer 입력을 위해 2D 형태로 변환
        )
        return x.view(*shape, -1)        # 원래 batch/query 구조로 다시 reshape하여 반환

