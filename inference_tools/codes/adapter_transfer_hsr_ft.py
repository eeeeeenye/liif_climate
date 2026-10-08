"""
HSR finetune MLP adapter(1-32-16-32-1, ReLU) 가 어떻게 학습됐는지 그린다.
(adapter_fold_phenomenon.py 와 같은 방식: adapter 입력 p_k -> 출력 transfer function)

- 위쪽: adapter 의 transfer function. 에폭별 곡선(색=epoch), best-loss 체크포인트 강조,
        입력 p_k 구간별 실제 ASOS 평균(train/val)을 겹침.
        오른쪽은 강수 대부분이 몰려 있는 0~8 mm/hr 확대.
- 가운데: 실제 adapter 입력 p_k(= frozen imnet 의 per-direction 출력) 분포 (train 모집단).
- 아래: 고정 입력에서의 보정량(출력-입력)이 에폭에 따라 어떻게 변했는지 / train·val loss.

p_k 는 adapter 를 뗀 base 체크포인트의 imnet 출력에 forward hook 을 걸어 얻는다
(local ensemble 4방향 x 관측소 위치). frozen imnet 이라 finetune 모델의 adapter 입력과 같다.
p_k 와 ASOS 는 station 하나의 관측값을 그 4개 p_k 에 똑같이 짝지은 근사.

env: liif, GPU 2 (CUDA_VISIBLE_DEVICES=2, 프로세스 안에서는 cuda:0)
"""
import glob
import os
import re
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec

plt.rcParams["axes.unicode_minus"] = False

V = 74.8784  # physical_norm.max (mm/hr)
S = "/home/inhye_yoo/deeplearning/inr/liif/save"
FT_DIR = f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3"
BASE_CKPT = f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth"
CFG = ("/home/inhye_yoo/deeplearning/inr/liif/configs/train_climate/"
       "climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3.yaml")
CACHE = "/tmp/claude-1018/-home-inhye-yoo/fa366d51-be66-48da-b7e8-102eabca7ed9/scratchpad/adapter_pk_cache_hsr.npz"
OUT = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/adapter_diagnostics/adapter_transfer_hsr_ft.png"
CROP = 48
STRIDE = {"train": 3, "val": 2}  # 시간 간격(속도용)


# ---------------------------------------------------------------- p_k 수집 (GPU)
def collect_all():
    import yaml
    import copy
    import datasets
    import models

    with open(CFG) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sv = torch.load(BASE_CKPT, map_location="cuda:0", weights_only=False)
    spec = copy.deepcopy(sv["model"])
    model = models.make({"name": spec["name"], "args": spec["args"]}).to("cuda:0")
    model.load_state_dict(spec["sd"], strict=True)
    model.eval()
    model.topo = None

    store = []
    model.imnet.register_forward_hook(lambda m, i, o: store.append(o.detach().float().cpu()))

    out = {}
    for key, name in [("train_dataset", "train"), ("val_dataset", "val")]:
        ds_spec = config[key]["dataset"]
        ds_spec["args"]["load_into_memory"] = False  # LR 만 필요해서 통째로 올리지 않음
        ds = datasets.make(ds_spec)

        crops, coords, obs = [], [], []
        for idx in range(0, len(ds), STRIDE[name]):
            img_lr, _, obs_, _ = ds[idx]
            H_lr, W_lr = img_lr.shape[-2:]
            v = obs_["gt"]
            if v.ndim == 1:
                v = v.unsqueeze(-1)
            mask = v.squeeze(-1) != -2.0
            if mask.sum() == 0:
                continue
            c = obs_["coord"][mask].clone()
            v = v[mask]
            c[:, 0] = -c[:, 0]  # wrappers_ft.py 와 같은 남/북 부호 보정
            for k in range(c.shape[0]):
                ia = (float(c[k, 0]) + 1) / 2 * (H_lr - 1)
                ja = (float(c[k, 1]) + 1) / 2 * (W_lr - 1)
                x0 = min(max(int(round(ia - CROP / 2)), 0), H_lr - CROP)
                y0 = min(max(int(round(ja - CROP / 2)), 0), W_lr - CROP)
                crops.append(img_lr[:, x0:x0 + CROP, y0:y0 + CROP])
                coords.append(torch.tensor([((ia - x0) / (CROP - 1)) * 2 - 1,
                                            ((ja - y0) / (CROP - 1)) * 2 - 1], dtype=torch.float32))
                obs.append(float(v[k, 0]))
        crops_t = torch.stack(crops)
        coords_t = torch.stack(coords).unsqueeze(1)
        n = len(crops)
        print(f"[{name}] (시각,관측소) 쌍: {n}", flush=True)

        P = []
        with torch.no_grad():
            for s in range(0, n, 512):
                inp = crops_t[s:s + 512].to("cuda:0")
                co = coords_t[s:s + 512].to("cuda:0")
                cell = torch.ones_like(co)
                cell[:, :, 0] *= 2 / 1000
                cell[:, :, 1] *= 2 / 1000
                store.clear()
                model(inp, None, co, cell)
                assert len(store) == 4, f"local ensemble 4방향이 아님: {len(store)}"
                P.append(torch.stack([t.reshape(-1) for t in store], dim=1))  # (B, 4)
        out[f"P_{name}"] = torch.cat(P).numpy() * V
        out[f"O_{name}"] = np.array(obs) * V
    np.savez(CACHE, **out)
    return out


if os.path.exists(CACHE):
    d = dict(np.load(CACHE))
    print("cache 사용:", CACHE)
else:
    d = collect_all()

P_tr, O_tr, P_va, O_va = d["P_train"], d["O_train"], d["P_val"], d["O_val"]
pk_tr = P_tr.ravel()
print(f"train p_k: n={pk_tr.size}  <0: {(pk_tr < 0).mean()*100:.1f}%  <0.5: {(pk_tr < 0.5).mean()*100:.1f}%  "
      f"<1: {(pk_tr < 1).mean()*100:.1f}%  >=10: {(pk_tr >= 10).sum()}  >=30: {(pk_tr >= 30).sum()}  max: {pk_tr.max():.1f}")


# ---------------------------------------------------------------- adapter (CPU)
def load_adapter(path):
    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]["sd"]
    ad = {k.replace("add_layer.asos_adapter.", ""): v.clone()
          for k, v in sd.items() if "add_layer.asos_adapter" in k}
    dims, L = [1, 32, 16, 32], []
    for a, b in zip(dims[:-1], dims[1:]):
        L += [nn.Linear(a, b), nn.ReLU()]
    L += [nn.Linear(dims[-1], 1)]
    m = nn.Sequential(*L)
    m.load_state_dict(ad)
    return m.eval(), ck.get("epoch")


def transfer(m, x_mm):
    with torch.no_grad():
        return m(torch.tensor(x_mm / V, dtype=torch.float32).view(-1, 1)).view(-1).numpy() * V


avail = sorted(int(re.search(r"epoch-(\d+)\.pth", p).group(1))
               for p in glob.glob(f"{FT_DIR}/epoch-*.pth") if re.search(r"epoch-(\d+)\.pth", p))
print(f"사용 가능한 epoch 체크포인트: {avail[0]}~{avail[-1]} ({len(avail)}개)")
curve_eps = [e for e in [301, 302, 305, 310, 320, 350, 400, 450, 500, 550, 600] if e in avail]
row3_eps = [e for e in avail if e <= 305 or (e - 301) % 5 == 0]

adapters = {}
for e in sorted(set(curve_eps) | set(row3_eps)):
    adapters[e], _ = load_adapter(f"{FT_DIR}/epoch-{e}.pth")
m_best, ep_best = load_adapter(f"{FT_DIR}/epoch-best-loss.pth")
print("best-loss 체크포인트 epoch:", ep_best)

x_full = np.linspace(0, 90, 1800)
x_zoom = np.linspace(-0.5, 8, 850)

# ---------------------------------------------------------------- 요약 출력
print("\n[best-loss adapter] 입력 -> 출력 (mm/hr)")
for xv in [0, 0.5, 1, 2, 5, 10, 20, 40, 60, 74.9]:
    y = float(transfer(m_best, np.array([xv]))[0])
    print(f"  in={xv:6.1f}  out={y:7.2f}  correction={y - xv:+7.2f}")
for name, m in [("best-loss", m_best), ("last(600)", adapters.get(600))]:
    if m is None:
        continue
    y = transfer(m, x_full)
    print(f"{name}: monotone={bool((np.diff(y) >= -1e-3).all())}  min diff={np.diff(y).min():.4f}")


def dry_point(P, O):
    # imnet 이 '비 없음'일 때 내는 거의 상수(-0.011 mm/hr) 출력 (train p_k 의 ~27%)
    p = P.ravel()
    o = np.repeat(O, 4)
    m = p < 0
    return p[m].mean(), o[m].mean(), o[m].std(ddof=1) / np.sqrt(m.sum()), int(m.sum())


def binned(P, O, edges, min_n=30):
    p = P.ravel()
    o = np.repeat(O, 4)
    idx = np.digitize(p, edges) - 1
    rows = []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() < min_n:
            continue
        rows.append((p[m].mean(), o[m].mean(), o[m].std(ddof=1) / np.sqrt(m.sum()), int(m.sum())))
    return np.array(rows)


e_full = np.array([0, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 60, 100], dtype=float)
e_zoom = np.arange(0, 8.01, 0.5)
b_full = {"train": binned(P_tr, O_tr, e_full), "val": binned(P_va, O_va, e_full)}
b_zoom = {"train": binned(P_tr, O_tr, e_zoom), "val": binned(P_va, O_va, e_zoom)}
print("\n[p_k 구간별 실제 ASOS 평균 (0.5mm 구간, 0~8)]  p_k평균 | train ASOS평균(n) | val ASOS평균(n)")
for r_tr, r_va in zip(b_zoom["train"], b_zoom["val"]):
    print(f"  {r_tr[0]:5.2f} | {r_tr[1]:5.2f} ({int(r_tr[3]):5d}) | {r_va[1]:5.2f} ({int(r_va[3]):5d})")

# ---------------------------------------------------------------- loss (log.txt)
pat = re.compile(r"epoch (\d+)/\d+, train: loss=([\d.]+), val: psnr=([\d.]+), val loss: ([\d.]+)")
loss = {}
with open(f"{FT_DIR}/log.txt") as f:
    for line in f:
        mm = pat.search(line)
        if mm:
            loss[int(mm.group(1))] = (float(mm.group(2)), float(mm.group(4)))
le = sorted(loss)

# ---------------------------------------------------------------- 그림
fig = plt.figure(figsize=(15, 14))
gs = GridSpec(3, 2, height_ratios=[3.2, 1.1, 2.2], hspace=0.32, wspace=0.18)
ax_f, ax_z = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_hf, ax_hz = fig.add_subplot(gs[1, 0], sharex=ax_f), fig.add_subplot(gs[1, 1], sharex=ax_z)
ax_c, ax_l = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

cmap = cm.viridis
norm = colors.Normalize(vmin=min(curve_eps), vmax=max(curve_eps))
for ax, xs, b in [(ax_f, x_full, b_full), (ax_z, x_zoom, b_zoom)]:
    lim = xs.max()
    ax.plot([0, lim], [0, lim], "k--", alpha=0.35, lw=1, label="y = x (no correction)")
    for e in curve_eps:
        ax.plot(xs, transfer(adapters[e], xs), color=cmap(norm(e)), lw=1.0, alpha=0.75)
    ax.plot(xs, transfer(m_best, xs), color="darkorange", lw=2.8, label=f"best-loss (ep {ep_best})")
    for split, c, mk in [("train", "crimson", "o"), ("val", "royalblue", "s")]:
        r = b[split]
        ax.errorbar(r[:, 0], r[:, 1], yerr=r[:, 2], fmt=mk + "-", color=c, ms=5, lw=1.4, capsize=2,
                    zorder=5, label=f"observed ASOS mean per p_k bin ({split}, ±1 SE)")
    for split, c, mk in [("train", "crimson", "o"), ("val", "royalblue", "s")]:
        dp = dry_point(P_tr if split == "train" else P_va, O_tr if split == "train" else O_va)
        ax.errorbar([dp[0]], [dp[1]], yerr=[dp[2]], fmt=mk, mfc="none", color=c, ms=8, mew=1.8, capsize=2,
                    zorder=6, label=f"'dry' cluster p_k<0 ({split}): ASOS mean {dp[1]:.2f}, n={dp[3]}")
    ax.set_xlim(xs.min(), lim)
    ax.set_ylabel("adapter output / ASOS (mm/hr)")
ax_f.axvspan(V, 90, color="0.9", zorder=0)
ax_f.text(V + 0.8, 3, "above normalization\nmax (74.9)", fontsize=8, color="0.4")
ax_f.set_ylim(-4, 90)
ax_z.set_ylim(-0.5, 5.5)
ax_z.legend(loc='upper left', fontsize=7)
ax_f.set_title("Adapter transfer function (full range)")
ax_z.set_title("Zoom 0-8 mm/hr (where most p_k live)")
ax_f.legend(loc="upper left", fontsize=7)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=[ax_f, ax_z], location="right", fraction=0.02, pad=0.02).set_label("epoch (thin lines)")

ax_hf.hist(pk_tr, bins=np.arange(-2, 91, 1), color="navy", alpha=0.6)
ax_hf.set_yscale("log")
ax_hf.set_ylabel("# p_k (log)")
ax_hf.set_xlabel("adapter input = per-direction imnet output p_k (mm/hr)")
ax_hf.text(0.98, 0.9, f"train p_k n={pk_tr.size}  >=10: {(pk_tr>=10).sum()}  >=30: {(pk_tr>=30).sum()}  max: {pk_tr.max():.0f}",
           transform=ax_hf.transAxes, ha="right", va="top", fontsize=8.5, color="navy")
ax_hz.hist(pk_tr, bins=np.arange(-0.5, 8.01, 0.1), color="navy", alpha=0.6)
ax_hz.set_yscale("log")
ax_hz.set_xlabel("p_k (mm/hr)")
ax_hz.text(0.98, 0.9, f"p_k<0: {(pk_tr<0).mean()*100:.1f}%   p_k<1: {(pk_tr<1).mean()*100:.1f}%",
           transform=ax_hz.transAxes, ha="right", va="top", fontsize=8.5, color="navy")

fixed = [0.0, 1.0, 3.0, 10.0, 30.0]
fc = plt.get_cmap("tab10")
for i, xv in enumerate(fixed):
    ys = [float(transfer(adapters[e], np.array([xv]))[0]) - xv for e in row3_eps]
    ax_c.plot(row3_eps, ys, color=fc(i), lw=1.4, label=f"input {xv:g} mm/hr")
ax_c.axhline(0, color="k", lw=0.8, alpha=0.5)
ax_c.set_xlabel("epoch")
ax_c.set_ylabel("correction = adapter(x) - x  (mm/hr)")
ax_c.set_title("How the correction at fixed inputs moved during training")
ax_c.legend(fontsize=8, ncol=2)

ax_l.plot(le, [loss[e][0] for e in le], color="crimson", lw=1.0, label="train loss")
ax_l.plot(le, [loss[e][1] for e in le], color="royalblue", lw=1.0, label="val loss")
ax_l.axvline(ep_best, color="darkorange", lw=1.2, ls="--", label=f"best-loss ep {ep_best}")
ax_l.set_xlabel("epoch")
ax_l.set_ylabel("MSE (normalized 0-1)")
ax_l.set_title("Loss per epoch (noisy)")
ax_l.legend(fontsize=8)

fig.suptitle("HSR finetune MLP adapter (1-32-16-32-1, ReLU): what it learned", fontsize=14)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved", OUT)
