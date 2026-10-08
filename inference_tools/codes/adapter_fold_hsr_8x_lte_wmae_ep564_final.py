"""
8x LTE (wmae, 2019+2020) ep564 : adapter 적용 전/후 '최종 픽셀값' 비교 (LR bilinear 포함, mm/h).
  - x : adapter 를 항등으로 바꾼 최종 출력 (= Σ w·imnet + LR bilinear)
  - y : adapter 적용 최종 출력 (실제 모델 출력)
  - 주황 점 : E[ASOS | x] (학습 짝 2019~2020, 2mm/h bin, n>=5),  녹색 점 : E[ASOS | y] 대신 y 의 bin 평균(=transfer) 은 파란 선
  - 파란 점/선 : E[y | x] (bin 평균)  ← 실제 adapter 의 효과
첫 실행만 모델을 돌려 CACHE 저장 (물리 GPU 2).
출력: /home/inhye_yoo/adapter_fold_phenomenon_hsr_8x_lte_wmae_ep564_final.png
env: liif
"""
import os, sys, copy
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

V = 74.8784
S = "/home/inhye_yoo/deeplearning/inr/liif"
RUN = f"{S}/save/_climate_edsr_radar_hsr_8x_minmax_lte_ft_mlp_1e-3_lag1h_2019_2020_wmae"
CKPT = f"{RUN}/epoch-564.pth"
CFG = f"{RUN}/config.yaml"
SP = "/tmp/claude-1018/-home-inhye-yoo/3596360f-c0ca-4591-8713-92bf5c9692c0/scratchpad"
CACHE = f"{SP}/final_pair_cache_hsr_8x_lte_wmae_ep564.npz"
OUT = "/home/inhye_yoo/adapter_fold_phenomenon_hsr_8x_lte_wmae_ep564_final.png"
CROP = 48


def collect():
    import yaml, datasets, models
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    config = yaml.load(open(CFG), Loader=yaml.FullLoader)
    spec = copy.deepcopy(ck["model"])
    model = models.make({"name": spec["name"], "args": spec["args"]}).to("cuda:0")
    model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
    adapter = model.add_layer
    ds_spec = config["train_dataset"]["dataset"]; ds_spec["args"]["load_into_memory"] = False
    ds = datasets.make(ds_spec)
    crops, coords, obs = [], [], []
    for idx in range(len(ds)):
        img_lr, _, obs_, _ = ds[idx]
        H_lr, W_lr = img_lr.shape[-2:]
        v = obs_["gt"]
        if v.ndim == 1: v = v.unsqueeze(-1)
        mask = v.squeeze(-1) != -2.0
        if mask.sum() == 0: continue
        c = obs_["coord"][mask].clone(); v = v[mask]
        c[:, 0] = -c[:, 0]                                   # wrappers_ft.py 와 같은 남/북 부호 보정
        for k in range(c.shape[0]):
            ia = (float(c[k, 0]) + 1) / 2 * (H_lr - 1); ja = (float(c[k, 1]) + 1) / 2 * (W_lr - 1)
            x0 = min(max(int(round(ia - CROP / 2)), 0), H_lr - CROP); y0 = min(max(int(round(ja - CROP / 2)), 0), W_lr - CROP)
            crops.append(img_lr[:, x0:x0 + CROP, y0:y0 + CROP])
            coords.append(torch.tensor([((ia - x0) / (CROP - 1)) * 2 - 1, ((ja - y0) / (CROP - 1)) * 2 - 1], dtype=torch.float32))
            obs.append(float(v[k, 0]))
    crops_t = torch.stack(crops); coords_t = torch.stack(coords).unsqueeze(1); n = len(crops)
    print("(시각,관측소) 쌍:", n, flush=True)
    Y, X = [], []
    with torch.no_grad():
        for s in range(0, n, 512):
            inp = crops_t[s:s + 512].to("cuda:0"); co = coords_t[s:s + 512].to("cuda:0")
            cell = torch.ones_like(co); cell[:, :, 0] *= 2 / 1000; cell[:, :, 1] *= 2 / 1000
            model.add_layer = adapter;     Y.append(model(inp, None, co, cell).reshape(-1).cpu())
            model.add_layer = nn.Identity(); X.append(model(inp, None, co, cell).reshape(-1).cpu())
    os.makedirs(SP, exist_ok=True)
    np.savez(CACHE, X=torch.cat(X).numpy() * V, Y=torch.cat(Y).numpy() * V, O=np.array(obs) * V)


if not os.path.exists(CACHE):
    collect()
c = np.load(CACHE); X, Y, O = c["X"], c["Y"], c["O"]
print(f"n={len(O)}  mean: no-adapter={X.mean():.3f} adapter={Y.mean():.3f} ASOS={O.mean():.3f}")
print(f"corr(ASOS): no-adapter={np.corrcoef(X, O)[0,1]:.3f} adapter={np.corrcoef(Y, O)[0,1]:.3f}")

edges = np.arange(np.floor(X.min() / 2) * 2, X.max() + 2, 2.0)
ix = np.digitize(X, edges) - 1
bins = [i for i in range(len(edges) - 1) if (ix == i).sum() >= 5]
bx = np.array([X[ix == i].mean() for i in bins])
by = np.array([Y[ix == i].mean() for i in bins])          # E[adapter 출력 | adapter 전 출력]
bo = np.array([O[ix == i].mean() for i in bins])          # E[ASOS | adapter 전 출력]
for t in (5, 10, 20, 30, 40):
    j = np.argmin(abs(bx - t)); print(f"x≈{bx[j]:.1f}: adapter후 {by[j]:.1f}  E[ASOS|x] {bo[j]:.1f}  n={(ix == bins[j]).sum()}")

lim = max(X.max(), Y.max(), 10) * 1.05
fig, (ax, axh) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[4, 1], sharex=True)
ax.plot([-10, lim], [-10, lim], color="red", ls="--", lw=1.2, label="y = x (no correction)")
ax.plot(bx, bo, "o", color="tab:orange", ms=3.5, label="E[ASOS | before adapter]")
ax.plot(bx, by, "-", color="tab:blue", lw=2.6, label="E[after adapter | before adapter]")
ax.axvline(0, color="0.7", lw=0.8); ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlim(-10, lim); ax.set_ylim(-10, lim)
ax.set_ylabel("after adapter / ASOS (mm/hr)")
ax.set_title("8x LTE adapter effect on FINAL pixel value (incl. LR bilinear), WMAE, ep564, 2019-2020 train pairs")
ax.legend(loc="upper left", fontsize=8)
axh.hist(X, bins=np.arange(-10, lim, 1), color="navy", alpha=.6)
axh.set_yscale("log"); axh.set_xlabel("final output before adapter (mm/hr)"); axh.set_ylabel("# pairs (log)")
plt.tight_layout(); plt.savefig(OUT, dpi=135); print("saved", OUT)
