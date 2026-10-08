"""
IMERG 0.2->0.1deg LTE ft(어댑터 128/64/128 wmae) val 2020-2022 일강수: IMERG(0.1deg) vs ASOS / 예측 vs ASOS hexbin (x=IMERG/예측, y=ASOS).
  - hexbin_hsr_preds_vs_asos_2021jja_8x_2019_2020.py 와 같은 형식. 단위 mm/day.
  - 지점별 값: val 데이터셋(obs_state=true) 전체 이미지를 모델에 통과시켜 지점 좌표에서 예측 pf 를 얻고,
    같은 좌표에서 0.1deg IMERG(HR) 를 nearest 로 뽑는다 (obs 가 있는 (지점,일) 만).
사용: cd /home/inhye_yoo/deeplearning/inr/liif
      ~/anaconda3/envs/liif/bin/python inference_tools/codes/hexbin_imerg_preds_vs_asos_val2020_2022.py   # CKPT=epoch-best-loss, THR=0 (필터 없음)
출력: inference_tools/station_eval/figures/hexbin_imerg_preds_vs_asos/hexbin_imerg_preds_vs_asos_val2020_2022_<CKPT>[_nofilter].png
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = "/home/inhye_yoo/deeplearning/inr/liif"
sys.path.insert(0, ROOT)
import datasets, models

RUN = os.environ.get("RUN", "_climate_edsr_topo_80x100_topo_minmax_lte_ft_mlp_1e-3_wmae_128_64_128")
CKPT = os.environ.get("CKPT", "epoch-best-loss")
THR = float(os.environ.get("THR", "0"))
LIM, GRID = float(os.environ.get("LIM", "250")), 60
SUF = "_nofilter" if THR <= 0 else ""
OUT = f"{ROOT}/inference_tools/station_eval/figures/hexbin_imerg_preds_vs_asos/hexbin_imerg_preds_vs_asos_val2020_2022_{CKPT}{SUF}{'_lim'+str(int(LIM)) if LIM!=250 else ''}.png"
NPZ = f"{ROOT}/inference_tools/station_eval/preds_npz/station_preds_imerg_val2020_2022_{CKPT}.npz"

config = yaml.load(open(f"{ROOT}/configs/train_climate/climate_edsr_topo_80x100_topo_minmax_lte_ft_mlp_1e-3_wmae_128_64_128.yaml"), Loader=yaml.FullLoader)
pn = config["physical_norm"]; scale, shift = float(pn["max"]) - float(pn["min"]), float(pn["min"])


def pad_collate(batch):
    lens = [b["coord"].shape[0] for b in batch]; L = max(lens)
    if L == 0:
        return None
    pad = lambda x, n: F.pad(x, (0, 0, 0, L - n))
    m = torch.zeros(len(batch), L)
    for i, n in enumerate(lens):
        m[i, :n] = 1
    return {"inp": torch.stack([b["inp"] for b in batch]), "gt": torch.stack([pad(b["gt"], n) for b, n in zip(batch, lens)]),
            "coord": torch.stack([pad(b["coord"], n) for b, n in zip(batch, lens)]),
            "cell": torch.stack([pad(b["cell"], n) for b, n in zip(batch, lens)]),
            "level": torch.stack([b["level"] for b in batch]), "mask": m}


if not os.path.exists(NPZ):
    spec = config["val_dataset"]
    base = datasets.make(spec["dataset"])
    ds = datasets.make(spec["wrapper"], args={"dataset": base})
    loader = DataLoader(ds, batch_size=spec["batch_size"], shuffle=False, num_workers=4, collate_fn=pad_collate)
    sd = torch.load(f"{ROOT}/save/{RUN}/{CKPT}.pth", map_location="cpu")
    model = models.make(sd["model"], load_sd=True).cuda().eval()
    P, G, H, D = [], [], [], []
    i0 = 0
    with torch.no_grad():
        for b in loader:
            n = len(b["inp"]) if b is not None else spec["batch_size"]
            if b is None:
                i0 += n; continue
            bc = {k: v.cuda() for k, v in b.items()}
            pred = model(bc["inp"], bc["level"], bc["coord"], bc["cell"])
            hr = torch.stack([base[i0 + k][1] for k in range(n)]).cuda()           # (B,1,H,W) 0.1deg IMERG(정규화)
            hv = F.grid_sample(hr, bc["coord"].flip(-1).unsqueeze(1), mode="nearest", align_corners=False)[:, :, 0, :].transpose(1, 2)  # (B,Q,1)
            v = bc["mask"].bool()
            P.append(pred[v].flatten().cpu()); G.append(bc["gt"][v].flatten().cpu()); H.append(hv[v].flatten().cpu())
            D.append(torch.arange(i0, i0 + n).unsqueeze(1).expand(-1, v.shape[1]).cuda()[v].cpu())
            i0 += n
    f = lambda t: np.clip(torch.cat(t).numpy() * scale + shift, 0, None)
    np.savez(NPZ, pf=f(P), obs=f(G), imerg_hr=f(H), day_idx=torch.cat(D).numpy())
d = np.load(NPZ)
m = (d["obs"] >= THR) & np.isfinite(d["pf"])
asos, imerg, pred = d["obs"][m], d["imerg_hr"][m], d["pf"][m]
print(f"n={m.sum()}")


def stats(x, y):
    return dict(r=np.corrcoef(x, y)[0, 1], bias=np.mean(x - y), rmse=np.sqrt(np.mean((x - y) ** 2)), ratio=x.mean() / y.mean())


def counts_max(x, y):
    fig, ax = plt.subplots(); hb = ax.hexbin(x, y, gridsize=GRID, extent=(0, LIM, 0, LIM), mincnt=1); c = hb.get_array().max(); plt.close(fig); return c


vmax = max(counts_max(imerg, asos), counts_max(pred, asos))
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=130)
for ax, x, title, xl in [(axes[0], imerg, "IMERG(0.1deg) vs ASOS  (daily)", "IMERG daily precip (mm/day)"),
                         (axes[1], pred, "preds vs ASOS  (daily)", "LTE ft preds daily precip (mm/day)")]:
    s = stats(x, asos)
    hb = ax.hexbin(x, asos, gridsize=GRID, extent=(0, LIM, 0, LIM), mincnt=1, norm=LogNorm(1, vmax), cmap="viridis")
    ax.plot([0, LIM], [0, LIM], color="0.5", ls="--", lw=1, label="1:1")
    b, a = np.polyfit(x, asos, 1); xx = np.array([0, LIM]); ax.plot(xx, a + b * xx, color="red", lw=1.6, label=f"OLS: y = {a:.2f} + {b:.2f}x")
    ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel(xl); ax.set_ylabel("ASOS daily precip (mm/day)")
    ax.set_title(f"{title}\nn={len(x)}  r={s['r']:.2f}  bias={s['bias']:.2f}  RMSE={s['rmse']:.2f}  ratio={s['ratio']:.2f}", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    print(title, {k: round(float(v), 3) for k, v in s.items()}, f"OLS a={a:.3f} b={b:.3f}")
fig.suptitle(f"2020-2022 daily precip  (IMERG ft {CKPT}, {'all days, no filter' if THR <= 0 else f'ASOS >= {THR} mm/day'})", fontsize=13)
fig.tight_layout(rect=(0, 0, 0.92, 1))
cb = fig.colorbar(hb, cax=fig.add_axes([0.94, 0.15, 0.015, 0.7])); cb.set_label("count (log)")
fig.savefig(OUT); print("saved", OUT)
