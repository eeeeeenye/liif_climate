"""
HSR finetune(model B) MLP adapter transfer function — adapter_fold_phenomenon.py 와 같은 형식.
각 체크포인트의 add_layer(asos_adapter) 만 떼어내서 per-direction imnet 출력 p_k (mm/hr) 스윕에 대한
transfer function 을 그린다. 아래 histogram 은 실제 학습 p_k 분포 (adapter_transfer_hsr_ft.py 캐시 사용).

출력: /home/inhye_yoo/adapter_fold_phenomenon_hsr.png
env:  liif
"""
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

V = 74.8784
CACHE = "/tmp/claude-1018/-home-inhye-yoo/fa366d51-be66-48da-b7e8-102eabca7ed9/scratchpad/adapter_pk_cache_hsr.npz"
S = "/home/inhye_yoo/deeplearning/inr/liif/save/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3"

RUNS = [
    ("best-loss", f"{S}/epoch-best-loss.pth", "darkorange"),
]

pk = np.load(CACHE)["P_train"].ravel()


def load_adapter(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]["sd"]
    ad = {k.replace("add_layer.asos_adapter.", ""): v
          for k, v in sd.items() if "add_layer.asos_adapter" in k}
    dims = [1, 32, 16, 32]; L = []
    for a, b in zip(dims[:-1], dims[1:]):
        L += [nn.Linear(a, b), nn.ReLU()]
    L += [nn.Linear(dims[-1], 1)]
    m = nn.Sequential(*L)
    m.load_state_dict(ad)
    m.eval()
    return m, ck.get("epoch")


x = torch.linspace(0, 1.3, 4000).view(-1, 1)
xin = x.view(-1).numpy() * V

fig, (ax, axh) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[4, 1], sharex=True)
ax.plot([0, V], [0, V], color="k", alpha=.3, lw=1, label="y = x (no correction)")
ax.axvspan(V, 1.3 * V, color="0.9")
for name, path, c in RUNS:
    m, ep = load_adapter(path)
    with torch.no_grad():
        y = m(x).view(-1).numpy() * V
    yc = np.clip(y, 0, None)
    lw = 2.6
    ax.plot(xin, yc, color=c, lw=lw, label=f"{name} (ep {ep})")
    mono = bool((np.diff(y) >= -0.01).all())
    print(f"{name} ep{ep}: monotone={mono}  10->{yc[np.argmin(abs(xin-10))]:.1f}  "
          f"30->{yc[np.argmin(abs(xin-30))]:.1f}  {V:.0f}->{yc[np.argmin(abs(xin-V))]:.1f}  {1.3*V:.0f}->{yc[-1]:.1f}")

ax.text(V + 1, 5, f"extrapolation\n(train max ~{pk.max():.0f}mm/hr)", fontsize=8, color="0.4")
ax.axvline(pk.max(), color="0.5", lw=0.8, ls=":")
ax.set_xlim(0, 1.3 * V); ax.set_ylim(-3, V * 1.05)
ax.set_ylabel("adapter output (mm/hr)")
ax.set_title("HSR finetune MLP adapter transfer function (model B, ASOS hourly)")
ax.legend(loc="upper left", fontsize=9)

axh.hist(pk[pk > 0.1], bins=np.arange(0, 1.3 * V, 1), color="navy", alpha=.6)
axh.set_yscale("log")
axh.set_xlabel("adapter input = per-direction imnet output p_k (mm/hr)")
axh.set_ylabel("# p_k (log)")
axh.text(45, axh.get_ylim()[1] * .2,
         f"training p_k:  >=10:{(pk>=10).sum()}  >=20:{(pk>=20).sum()}  >=30:{(pk>=30).sum()}  max:{pk.max():.0f}",
         fontsize=9, color="navy")

plt.tight_layout()
out = "/home/inhye_yoo/adapter_fold_phenomenon_hsr.png"
plt.savefig(out, dpi=135)
print("saved", out)
