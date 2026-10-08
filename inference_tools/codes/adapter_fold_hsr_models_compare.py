"""
[모델별 비교판] 8x MSE / 8x wmae / 2x wmae 128-64-128 어댑터 transfer function.
(원본 adapter_fold_hsr_8x_2019_2020.py 복사본; RUNS/build/출력만 변경. 히스토그램 p_k 는 8x base 기준)
8x LIIF ASOS-finetune MLP adapter transfer function (adapter_fold_hsr_lag1h_compare.py 와 같은 형식/축).
  - 파란 실선 : 2019+2020 학습 run (..._ft_mlp_1e-3_lag1h_2019_2020) best-loss
  - 회색 점선 : 기존 2019 만 학습한 run (..._ft_mlp_1e-3_lag1h) best-loss   (비교용)
  - 점/빨간 점선 : 정렬된 HSR-ASOS 시간별 짝 (2019~2020 = 학습 기간) 의 E[ASOS|HSR] 와 OLS  (x 는 HSR 500m 관측; 모델 p_k 아님)
  - 아래 히스토그램 : adapter 입력 p_k (frozen imnet 의 per-direction 출력; 8x base, 학습 (지점,시각) 전부)
p_k 는 두 run 모두 같은 base(imnet frozen) 라 공통. 첫 실행 때 GPU 2 로 계산해 CACHE 에 저장.

출력: /home/inhye_yoo/adapter_fold_phenomenon_hsr_8x_2019_2020.png
env:  liif
"""
import copy
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")

import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

V = 74.8784
S = "/home/inhye_yoo/deeplearning/inr/liif/save"
NEW = f"{S}/_climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020"
OLD = f"{S}/_climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h"
BASE_CKPT = f"{S}/_climate_edsr_radar_hsr_8x_minmax_liif/epoch-last.pth"
CFG = ("/home/inhye_yoo/deeplearning/inr/liif/configs/train_climate/"
       "climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020.yaml")
CACHE = "/tmp/claude-1018/-home-inhye-yoo/7ba25758-455d-42df-8513-9f96417dc540/scratchpad/adapter_pk_cache_hsr_8x_2019_2020.npz"
PAIRS = "/home/inhye_yoo/data/radar_HSR/analysis/results/pairs_hourly_jja_2019_2021.csv"
OUT = "/home/inhye_yoo/adapter_fold_phenomenon_hsr_models_compare.png"
CROP = 48

W8 = f"{S}/_climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020_wmae"
W2 = f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h_vzero_wmae_plainadapter_128_64_128"
RUNS = [("8x MSE 32/16/32", f"{NEW}/epoch-best-loss.pth", "0.45", "--", 2.2),
        ("8x wmae 32/16/32", f"{W8}/epoch-best-loss.pth", "tab:blue", "-", 2.6),
        ("2x wmae 128/64/128 (vzero)", f"{W2}/epoch-best-loss.pth", "tab:green", "-", 2.6)]


def collect_pk():
    import yaml, datasets, models
    config = yaml.load(open(CFG), Loader=yaml.FullLoader)
    sv = torch.load(BASE_CKPT, map_location="cuda:0", weights_only=False)
    spec = copy.deepcopy(sv["model"])
    model = models.make({"name": spec["name"], "args": spec["args"]}).to("cuda:0")
    model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
    store = []
    model.imnet.register_forward_hook(lambda m, i, o: store.append(o.detach().float().cpu()))

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


def load_ckpt(path):
    for _ in range(5):                       # 학습 중 덮어쓰는 파일은 잠깐 깨져 있을 수 있어 재시도
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            time.sleep(3)
    raise RuntimeError(path)


def build(ck):
    sd = ck["model"]["sd"]
    ad = {k.replace("add_layer.asos_adapter.", ""): v for k, v in sd.items() if "add_layer.asos_adapter" in k}
    ws = [v for k, v in ad.items() if k.endswith("weight")]
    dims, L = [1] + [w.shape[0] for w in ws[:-1]], []
    for a, b in zip(dims[:-1], dims[1:]):
        L += [nn.Linear(a, b), nn.ReLU()]
    m = nn.Sequential(*L, nn.Linear(dims[-1], 1)); m.load_state_dict(ad); m.eval()
    return m


if not os.path.exists(CACHE):
    collect_pk()
_c = np.load(CACHE); pk = _c["P_train"].ravel(); ok_ = np.repeat(_c["O_train"], _c["P_train"].shape[1])   # p_k 4방향 각각에 같은 ASOS 를 짝지음
x = torch.linspace(0, 1.3, 4000).view(-1, 1); xin = x.view(-1).numpy() * V

oh = pd.read_csv(PAIRS)
oh = oh[pd.to_datetime(oh["t"]).dt.year.isin([2019, 2020])]   # 학습 기간(2019~2020)
hx, hy = oh["radar_mm_h"].to_numpy(float), oh["asos_mm_h"].to_numpy(float)
edges = np.arange(0, hx.max() + 2, 2.0); ix = np.digitize(hx, edges) - 1
bx, by = zip(*[(hx[ix == i].mean(), hy[ix == i].mean()) for i in range(len(edges) - 1) if (ix == i).sum() >= 5])

fig, (ax, axh) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[4, 1], sharex=True)
ax.plot([0, V], [0, V], color="red", ls="--", lw=1.2, label="y = x (no correction)")
ax.axvspan(V, 1.3 * V, color="0.9")
ax.plot(bx, by, "o", color="#31548f", ms=3.5, label="E[ASOS | HSR] (aligned pairs 2019-2020, x=HSR obs)")
pix = np.digitize(pk, edges) - 1                       # p_k 기준 E[ASOS | p_k]: 같은 2 mm/h bin, n>=5
px, py = zip(*[(pk[pix == i].mean(), ok_[pix == i].mean()) for i in range(len(edges) - 1) if (pix == i).sum() >= 5])
ax.plot(px, py, "o", color="tab:orange", ms=3.5, label="E[ASOS | p_k] (train pairs 2019-2020, x=p_k, per-direction)")
slope, icpt = np.polyfit(hx, hy, 1); r = np.corrcoef(hx, hy)[0, 1]
xr = np.array([0, hx.max()])
ax.plot(xr, slope * xr + icpt, color="k", lw=1.8, ls="-",
        label=f"OLS ASOS~HSR 2019-2020 (y={slope:.2f}x{icpt:+.2f}, r={r:.2f}, n={len(hx)})")
print(f"OLS: slope={slope:.3f} icpt={icpt:.3f} r={r:.3f} n={len(hx)}")
print(f"{'run':28s} ep   f(5)   f(10)  f(20)  f(30)  f(40)  mono")
for name, path, c, ls, lw in RUNS:
    ck = load_ckpt(path); m = build(ck)
    with torch.no_grad():
        y = m(x).view(-1).numpy() * V
    yc = np.clip(y, 0, None)
    ax.plot(xin, yc, color=c, ls=ls, lw=lw, label=f"{name} (ep {ck.get('epoch')})")
    f = lambda t: yc[np.argmin(abs(xin - t))]
    print(f"{name:28s} {ck.get('epoch')}  {f(5):5.1f}  {f(10):5.1f}  {f(20):5.1f}  {f(30):5.1f}  {f(40):5.1f}  {bool((np.diff(y) >= -0.01).all())}")
ax.axvline(pk.max(), color="0.5", lw=0.8, ls=":")
ax.text(V + 1, 5, f"extrapolation\n(train max ~{pk.max():.0f}mm/hr)", fontsize=8, color="0.4")
ax.set_xlim(0, 1.3 * V); ax.set_ylim(-3, V * 1.05)
ax.set_ylabel("adapter output / ASOS (mm/hr)")
ax.set_title("HSR MLP adapter transfer function by model (lag1h pairs)")
ax.legend(loc="upper left", fontsize=8)
axh.hist(pk[pk > 0.1], bins=np.arange(0, 1.3 * V, 1), color="navy", alpha=.6)
axh.set_yscale("log"); axh.set_xlabel("adapter input = per-direction imnet output p_k (mm/hr)"); axh.set_ylabel("# p_k (log)")
plt.tight_layout(); plt.savefig(OUT, dpi=135); print("saved", OUT)
