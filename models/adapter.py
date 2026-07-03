import torch.nn as nn                    # PyTorch 신경망 모듈 사용
from models import register              # 모델 registry에 adapter를 등록하기 위한 함수

@register("adapter")                     # config에서 name: adapter로 호출할 수 있도록 등록
class ASOSAdapter(nn.Module):            # ASOS 보정을 위한 adapter layer 정의
    def __init__(self, in_dim, out_dim):  # 입력 차원과 출력 차원을 받아 초기화
        # in_dim : 1
        # out_dim : 1
        super().__init__()               # nn.Module 초기화
        print(f"add_lay dim ::: in_dim : {in_dim}, out_dim : {out_dim}")

        self.asos_adapter = nn.Sequential(
            nn.Linear(in_dim, out_dim)   # 선형 변환 layer: y = Wx + b
        )

    def forward(self, x):                # forward 연산 정의
        shape = x.shape[:-1]             # 마지막 차원을 제외한 기존 shape 저장
        x = self.asos_adapter(
            x.view(-1, x.shape[-1])      # Linear layer 입력을 위해 2D 형태로 변환
        )
        return x.view(*shape, -1)        # 원래 batch/query 구조로 다시 reshape하여 반환