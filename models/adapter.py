import torch.nn as nn
from models import register

@register("adapter")
class ASOSAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.asos_adapter = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.asos_adapter(x.view(-1, x.shape[-1]))
        # print(f"adapter shape : {shape}")
        # print(f"x shape : {x.shape}")
        return x.view(*shape, -1)