"""
MLP adapter transfer function — flip bug fix removes the fold.

각 run 의 add_layer(asos_adapter) 만 떼어내서 per-direction imnet 출력 p_k (mm/day)
스윕에 대한 transfer function 을 그린다. 아래 histogram 은 실제 학습 p_k 분포.

출력: /home/inhye_yoo/adapter_fold_phenomenon.png
env:  /home/inhye_yoo/anaconda3/envs/liif/bin/python
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

V = 506.01995849609375
CSV = "/home/inhye_yoo/data/observation/IMERG_ASOS_Paired/codes/inference_results/station_base_pred_2000_2002_ftmlp_epochbest.csv"
S = "/home/inhye_yoo/deeplearning/inr/liif/save"

# name, ckpt, activation (None = 단일 Linear(bias=False) 어댑터), color
RUNS = [
    ("old  _ft_mlp (folded, ep3136)", f"{S}/_climate_edsr_topo_80x100_topo_precip_minmax_ft_mlp/epoch-best.pth", nn.ReLU, "gray"),
    ("new  ReLU flipped (best)",       f"{S}/_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-3_flipped/epoch-best.pth", nn.ReLU, "seagreen"),
    ("new  tanh flipped (best)",       f"{S}/_climate_edsr_topo_80x100_topo_minmax_ft_mlp_1e-3_flipped_tanh/epoch-best.pth", nn.Tanh, "royalblue"),
    ("new  linear flipped (best)",     f"{S}/_climate_edsr_imerg_asos_ft_adapter_80x100_linear_flipped/epoch-best.pth", None, "darkorange"),
]

d = pd.read_csv(CSV)
pk = d[["p1", "p2", "p3", "p4"]].values.ravel()


def load_adapter(path, act):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]["sd"]
    ad = {k.replace("add_layer.asos_adapter.", ""): v
          for k, v in sd.items() if "add_layer.asos_adapter" in k}
    if act is None:                              # 단일 Linear(1,1, bias=False)
        m = nn.Sequential(nn.Linear(1, 1, bias=False))
    else:                                        # 1-32-16-32-1 MLP
        dims = [1, 32, 16, 32]; L = []
        for a, b in zip(dims[:-1], dims[1:]):
            L += [nn.Linear(a, b), act()]
        L += [nn.Linear(dims[-1], 1)]
        m = nn.Sequential(*L)
    m.load_state_dict(ad)
    m.eval()
    return m, ck.get("epoch")


x = torch.linspace(0, 1.3, 4000).view(-1, 1)
xin = x.view(-1).numpy() * V

fig, (ax, axh) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[4, 1], sharex=True)
ax.plot([0, 520], [0, 520], color="k", alpha=.3, lw=1, label="y = x (no correction)")
ax.axvspan(506, 650, color="0.9")
for name, path, act, c in RUNS:
    try:
        m, ep = load_adapter(path, act)
    except Exception as e:
        print(name, "skip:", e); continue
    with torch.no_grad():
        y = m(x).view(-1).numpy() * V
    yc = np.clip(y, 0, None)
    lw = 2.0 if name.startswith("old") else 2.6
    ax.plot(xin, yc, color=c, lw=lw, label=f"{name} (ep {ep})")
    mono = bool((np.diff(y) >= -0.1).all())
    print(f"{name} ep{ep}: monotone={mono}  200->{yc[np.argmin(abs(xin-200))]:.0f}  "
          f"326->{yc[np.argmin(abs(xin-326))]:.0f}  506->{yc[np.argmin(abs(xin-506))]:.0f}  650->{yc[-1]:.0f}")

ax.text(508, 30, "extrapolation\n(train max ~382mm)", fontsize=8, color="0.4")
ax.set_xlim(0, 650); ax.set_ylim(-20, 520)
ax.set_ylabel("adapter output (mm/day)")
ax.set_title("MLP adapter transfer function — flip bug fix removes the fold (ReLU & tanh both monotone)")
ax.legend(loc="upper left", fontsize=9)

axh.hist(pk[pk > 1], bins=np.arange(0, 650, 5), color="navy", alpha=.6)
axh.set_yscale("log")
axh.set_xlabel("adapter input = per-direction imnet output p_k (mm/day)")
axh.set_ylabel("# p_k (log)")
axh.text(360, axh.get_ylim()[1] * .2,
         f"training p_k:  >=150:{(pk>=150).sum()}  >=200:{(pk>=200).sum()}  >=326:{(pk>=326).sum()}  max:{pk.max():.0f}",
         fontsize=9, color="navy")

plt.tight_layout()
out = "/home/inhye_yoo/adapter_fold_phenomenon.png"
plt.savefig(out, dpi=135)
print("saved", out)
