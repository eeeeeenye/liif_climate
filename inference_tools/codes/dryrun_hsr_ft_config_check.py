"""
HSR adapter finetune config 드라이런 (GPU/학습 없이 데이터 경로만):

  config 의 train/val dataset + wrapper 를 학습과 똑같이 만들어(load_into_memory 만 False 로 바꿔 빠르게)
  wrapper 가 내주는 샘플 (inp = LR patch, coord = patch 기준 관측소 좌표, gt = 정규화된 ASOS) 을 N개 뽑고,
    · LR patch 를 관측소 좌표에서 샘플링한 값(mm/h) 과 ASOS(mm/h) 의 상관/평균비
    · dataset 길이, 시각 수, 관측소 수, 결측 비율, 값 범위(정규화 [0,1.7] 클램프 여부)
  를 출력한다.  wrapper 의 좌표 뒤집기/anchor crop 재정규화까지 포함한 최종 점검이라,
  정렬이 맞으면 상관 ~0.7 (LR=1km 를 관측소에 바로 대응하므로 500m nearest 보다 약간 낮을 수 있음),
  1시간 어긋나 있으면 ~0.4.

사용:  python dryrun_hsr_ft_config_check.py <config.yaml> [N]
env: liif   (CUDA 안 씀)
"""
import os, sys, random
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
os.chdir("/home/inhye_yoo/deeplearning/inr/liif")
import numpy as np, torch, yaml
import torch.nn.functional as F
import datasets

cfg_path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
V = cfg["physical_norm"]["max"]
print(f"config: {cfg_path}\nphysical_norm max = {V}")

for tag in ["train_dataset", "val_dataset"]:
    spec = cfg[tag]
    spec["dataset"]["args"]["load_into_memory"] = False
    for k in ("obs_path_3", "meta_path", "root_path_1", "root_path_2"):
        assert os.path.exists(spec["dataset"]["args"][k]), f"없음: {spec['dataset']['args'][k]}"
    ds = datasets.make(spec["dataset"])
    ds = datasets.make(spec["wrapper"], args={"dataset": ds})
    print(f"\n[{tag}] len={len(ds)}")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    idxs = random.sample(range(len(ds)), min(N, len(ds)))
    P, O, nst, nemp, nout = [], [], [], 0, 0
    gmin, gmax = 1e9, -1e9
    for i in idxs:
        s = ds[i]
        inp, coord, gt = s["inp"], s["coord"], s["gt"]            # (1,h,w), (n,2)=(y,x), (n,1)
        n = coord.shape[0]
        nst.append(n)
        if n == 0:
            nemp += 1
            continue
        assert gt.shape[0] == n and inp.ndim == 3
        assert torch.isfinite(gt).all() and (coord.abs() <= 1.06).all(), "좌표/값 이상"
        nout += int((coord.abs() > 1.0 + 1e-4).any())      # patch 마지막 픽셀 바깥 반쪽 (<=1.04). 모델이 clamp(liif_baseline.py:154)
        gmin, gmax = min(gmin, float(gt.min())), max(gmax, float(gt.max()))
        g = coord.flip(-1).view(1, n, 1, 2)                       # (x,y) 순서로 grid_sample
        v = F.grid_sample(inp[None], g, mode="bilinear", padding_mode="border", align_corners=False)
        P.append(v.view(-1).numpy() * V); O.append(gt.view(-1).numpy() * V)
    P, O = np.concatenate(P), np.concatenate(O)
    ok = np.isfinite(P) & np.isfinite(O)
    print(f"  샘플 {len(idxs)}개: 관측소/패치 평균 {np.mean(nst):.2f} (0개 패치 {nemp}), 관측점 {ok.sum()}")
    print(f"  좌표가 [-1,1] 을 살짝 넘은 패치: {nout}/{len(idxs)}  (모델이 clamp 하므로 무해)")
    print(f"  gt(정규화) 범위 [{gmin:.3f}, {gmax:.3f}]  (0~1.7 클램프 안)")
    print(f"  HSR(LR)@관측소 vs ASOS:  corr={np.corrcoef(P[ok], O[ok])[0,1]:.3f}   "
          f"mean HSR={P[ok].mean():.3f}  mean ASOS={O[ok].mean():.3f}  ratio={P[ok].mean()/O[ok].mean():.2f}")
