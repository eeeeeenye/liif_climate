"""
8x LTE (wmae, 2019+2020 학습) ep564 (epoch-best) MLP adapter transfer function.  adapter_fold_hsr_8x_lag1h.py 와 같은 형식/축.
  - 파란 실선 : ep564 adapter 변환 (입력 p_k = 방향별 imnet 출력, 출력 = adapter 출력; mm/h)
  - 주황 점   : E[ASOS | p_k] (학습 짝 2019~2020, 4방향 각각에 같은 ASOS 를 짝지음, 2mm/h bin n>=5)
  - 아래 히스토그램 : adapter 입력 p_k 분포
  p_k 는 ep564 체크포인트의 모델을 학습 데이터에 돌려 add_layer 입력을 hook 으로 수집 (첫 실행만, CACHE 저장). 물리 GPU 2.
출력: /home/inhye_yoo/adapter_fold_phenomenon_hsr_8x_lte_wmae_ep564_2019_2020.png
env:  liif
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
CACHE = "/tmp/claude-1018/-home-inhye-yoo/3596360f-c0ca-4591-8713-92bf5c9692c0/scratchpad/adapter_pk_cache_hsr_8x_lte_wmae_ep564.npz"
OUT = "/home/inhye_yoo/adapter_fold_phenomenon_hsr_8x_lte_wmae_ep564_2019_2020_negx.png"
CROP = 48

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
ad = {k.replace("add_layer.asos_adapter.", ""): v for k, v in ck["model"]["sd"].items() if "add_layer.asos_adapter" in k}
dims, L = [1, 32, 16, 32], []
for a, b in zip(dims[:-1], dims[1:]):
    L += [nn.Linear(a, b), nn.ReLU()]
m = nn.Sequential(*L, nn.Linear(dims[-1], 1)); m.load_state_dict(ad); m.eval()


def collect_pk():
    import yaml, datasets, models
    config = yaml.load(open(CFG), Loader=yaml.FullLoader)
    spec = copy.deepcopy(ck["model"])
    model = models.make({"name": spec["name"], "args": spec["args"]}).to("cuda:0")
    model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
    store = []
    model.add_layer.register_forward_hook(lambda mod, i, o: store.append(i[0].detach().float().cpu()))
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
    P = []
    with torch.no_grad():
        for s in range(0, n, 512):
            inp = crops_t[s:s + 512].to("cuda:0"); co = coords_t[s:s + 512].to("cuda:0")
            cell = torch.ones_like(co); cell[:, :, 0] *= 2 / 1000; cell[:, :, 1] *= 2 / 1000
            store.clear(); model(inp, None, co, cell)
            assert len(store) == 4, f"local ensemble 4방향이 아님: {len(store)}"
            P.append(torch.stack([t.reshape(-1) for t in store], dim=1))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, P_train=torch.cat(P).numpy() * V, O_train=np.array(obs) * V)


if not os.path.exists(CACHE):
    collect_pk()
_c = np.load(CACHE); Pk, Ok = _c["P_train"], _c["O_train"]
pk = Pk.ravel(); ok_ = np.repeat(Ok, Pk.shape[1])
x = torch.linspace(-0.7, 1.3, 4000).view(-1, 1); xin = x.view(-1).numpy() * V
with torch.no_grad():
    y = m(x).view(-1).numpy() * V
yc = np.clip(y, 0, None)

edges = np.arange(np.floor(pk.min()), pk.max() + 2, 2.0)
pix = np.digitize(pk, edges) - 1
px, py = zip(*[(pk[pix == i].mean(), ok_[pix == i].mean()) for i in range(len(edges) - 1) if (pix == i).sum() >= 5])

fig, (ax, axh) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[4, 1], sharex=True)
ax.plot([-50, V], [-50, V], color="red", ls="--", lw=1.2, label="y = x (no correction)")
ax.axvspan(V, 1.3 * V, color="0.9")
ax.plot(px, py, "o", color="tab:orange", ms=3.5, label="E[ASOS | p_k] (train pairs 2019-2020, x=p_k, per-direction)")
ax.plot(xin, yc, color="tab:blue", lw=2.6, label=f"8x LTE wmae best-r (ep {ck.get('epoch')})")
ax.axvline(pk.max(), color="0.5", lw=0.8, ls=":"); ax.axvline(pk.min(), color="0.5", lw=0.8, ls=":")
ax.text(V + 1, 5, f"extrapolation\n(p_k range {pk.min():.0f}~{pk.max():.0f}mm/hr,\n2019-2020 train)", fontsize=8, color="0.4")
ax.set_xlim(-50, 1.3 * V); ax.set_ylim(-10, V * 1.05); ax.axvline(0, color="0.7", lw=0.8); ax.axhline(0, color="0.7", lw=0.8)
ax.set_ylabel("adapter output / ASOS (mm/hr)")
ax.set_title("8x LTE MLP adapter transfer function (WMAE, lag1h pairs, 2019-2020 train, ep564)")
ax.legend(loc="upper left", fontsize=8)
axh.hist(pk, bins=np.arange(-50, 1.3 * V, 1), color="navy", alpha=.6)
axh.set_yscale("log"); axh.set_xlabel("adapter input = per-direction imnet output p_k (mm/hr)"); axh.set_ylabel("# p_k (log)")
plt.tight_layout(); plt.savefig(OUT, dpi=135)
f = lambda t: yc[np.argmin(abs(xin - t))]
print(f"ep {ck.get('epoch')}  f(5)={f(5):.1f} f(10)={f(10):.1f} f(20)={f(20):.1f} f(30)={f(30):.1f} f(40)={f(40):.1f} mono={bool((np.diff(y) >= -0.01).all())}")
print(f"saved {OUT}")
