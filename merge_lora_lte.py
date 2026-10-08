""" LoRA-LTE 체크포인트 -> 일반 LTE 체크포인트 (W <- W + (1/r) B A 로 합침).

    합친 결과는 기존 LTE 추론 코드/스크립트를 그대로 쓸 수 있는 {'model': {name,args,sd}} 형식.
      python merge_lora_lte.py save/<run>/epoch-best-loss.pth [out.pth]
"""
import sys

import torch

from models.lora import build_lora_lte, merge_lora

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else src.replace('.pth', '_merged.pth')

ck = torch.load(src, map_location='cpu')
model, base_spec = build_lora_lte(ck['base_ckpt'], ck['rank'], ck['layers'], ck['lora'], device='cpu')
merge_lora(model)
spec = dict(base_spec)
spec['sd'] = model.state_dict()
torch.save({'model': spec, 'epoch': ck['epoch']}, dst)
print('saved', dst)
