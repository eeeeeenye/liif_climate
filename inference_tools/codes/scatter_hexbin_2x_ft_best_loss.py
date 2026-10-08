"""
2x LR 입력 + 8x 사전학습에서 어댑터만 파인튜닝한 모델(..._2x_minmax_liif_ft_mlp_1e-3_lag1h_from8x)의 epoch-best-loss 로
2021/2019 JJA 시간별 hexbin 2패널(HSR vs ASOS | preds vs ASOS, ASOS>=0.1) — scatter_hourly_hsr_pred_vs_asos_jja2021.py 와 같은 형식.
  p_k/가중치: 8x pretrain(동결 imnet) 에 2x LR 입력을 넣어 추출(지점 최근접 500m), 어댑터는 학습 중 체크포인트를 복사해서 사용(쓰는 도중 깨짐 방지).
  pred = sum_k w_k g(p_k) / sum_k w_k, g=어댑터(0 미만 0).  HSR/ASOS 는 station_preds_val2021_lag1h_mse_new.npz 의 hsr_hr/obs(같은 (지점,시각) 순서, obs 일치 검증).
env: liif (GPU 2)
출력: inference_tools/station_eval/figures/hexbin_hsr_preds_vs_asos/2x_ft/hexbin_hsr_preds_vs_asos_{YEAR}jja_{TAG}_best-loss_ep<epoch>.png
      inference_tools/station_eval/tables/pk_all_{YEAR}jja_{TAG}_best-loss_ep<epoch>.csv
"""
import os, sys, copy, shutil
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif"); os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np, pandas as pd, torch, torch.nn as nn, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import datasets, models

YEAR = int(os.environ.get("YEAR", "2021")); KEY = "val_dataset" if YEAR == 2021 else "train_dataset"
REF = "station_preds_val2021_lag1h_mse_new.npz" if YEAR == 2021 else "station_preds_train2019_8x_ft_epoch-best-loss.npz"
H = "/home/inhye_yoo/deeplearning/inr/liif"; S = f"{H}/save"; V = 74.8784; CROP = 48; LIM = 80; GRID = 60
PRE = os.environ.get("PRE", "8x")   # 사전학습 배율: 8x(from8x 모델) | 2x(2x 사전학습 -> 2x ft)
TAG = "2x_from8x" if PRE == "8x" else "2x_pre2x"
SFX = "_from8x" if PRE == "8x" else ""
RUN = f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h{SFX}"
TMP = f"/tmp/claude-1018/-home-inhye-yoo/4e9fbb29-fb41-4952-accd-5396550b8bc4/scratchpad/{TAG}_best_loss.pth"
shutil.copy(f"{RUN}/epoch-best-loss.pth", TMP)
ck = torch.load(TMP, map_location="cpu", weights_only=False); EP = ck["epoch"]; print("best-loss epoch", EP, flush=True)

cfg = yaml.load(open(f"{H}/configs/train_climate/climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h{SFX}.yaml"), Loader=yaml.FullLoader)
sv = torch.load(f"{S}/_climate_edsr_radar_hsr_{PRE}_minmax_liif/epoch-last.pth", map_location="cuda:0", weights_only=False)
spec = copy.deepcopy(sv["model"]); model = models.make({"name": spec["name"], "args": spec["args"]}).to("cuda:0")
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
store = []; model.imnet.register_forward_hook(lambda m, i, o: store.append(o.detach().float().cpu()))
ds = datasets.make(cfg[KEY]["dataset"]); crops, coords, obs = [], [], []
for idx in range(len(ds)):
    img_lr, _, obs_, _ = ds[idx]; H_lr, W_lr = img_lr.shape[-2:]; v = obs_["gt"]
    if v.ndim == 1: v = v.unsqueeze(-1)
    mask = v.squeeze(-1) != -2.0
    if mask.sum() == 0: continue
    c = obs_["coord"][mask].clone(); v = v[mask]; c[:, 0] = -c[:, 0]
    for k in range(c.shape[0]):
        ia = (float(c[k, 0]) + 1) / 2 * (H_lr - 1); ja = (float(c[k, 1]) + 1) / 2 * (W_lr - 1)
        x0 = min(max(int(round(ia - CROP / 2)), 0), H_lr - CROP); y0 = min(max(int(round(ja - CROP / 2)), 0), W_lr - CROP)
        crops.append(img_lr[:, x0:x0 + CROP, y0:y0 + CROP])
        coords.append(torch.tensor([((ia - x0) / (CROP - 1)) * 2 - 1, ((ja - y0) / (CROP - 1)) * 2 - 1], dtype=torch.float32)); obs.append(float(v[k, 0]))
crops_t = torch.stack(crops); coords_t = torch.stack(coords).unsqueeze(1); n = len(crops); print(f"{YEAR} 쌍:", n, flush=True); P, W = [], []
with torch.no_grad():
    for s in range(0, n, 256):
        inp = crops_t[s:s + 256].to("cuda:0"); co = coords_t[s:s + 256].to("cuda:0")
        cell = torch.ones_like(co); cell[:, :, 0] *= 2 / 1000; cell[:, :, 1] *= 2 / 1000
        store.clear(); model(inp, None, co, cell); assert len(store) == 4
        P.append(torch.stack([t.reshape(-1) for t in store], dim=1)); W.append(torch.stack([w.reshape(-1).cpu() for w in model.le_weights], dim=1))
P = torch.cat(P).numpy() * V; W = torch.cat(W).numpy(); O = np.array(obs) * V

ref = np.load(f"{H}/inference_tools/station_eval/preds_npz/" + REF)
assert np.abs(O - ref["obs"]).max() < 1e-3, "(지점,시각) 순서/ASOS 불일치"
ad = {k.replace("add_layer.asos_adapter.", ""): v for k, v in ck["model"]["sd"].items() if "add_layer.asos_adapter" in k}
L = []
for i, o in zip([1, 32, 16], [32, 16, 32]): L += [nn.Linear(i, o), nn.ReLU()]
net = nn.Sequential(*L, nn.Linear(32, 1)); net.load_state_dict(ad); net.eval()
with torch.no_grad(): G = np.clip(net(torch.tensor(P / V, dtype=torch.float32).reshape(-1, 1)).numpy().reshape(P.shape) * V, 0, None)
pred = (G * W).sum(1) / W.sum(1)
df = pd.DataFrame({"tid": ref["tid"], "lat": ref["lat"], "lon": ref["lon"], "asos": ref["obs"], "hsr": ref["hsr_hr"], "pred": pred})
for i in range(4): df[f"pk{i+1}_in"] = P[:, i]
for i in range(4): df[f"g{i+1}_out"] = G[:, i]
df = df[df.asos >= 0.1]
T = f"{H}/inference_tools/station_eval/tables/pk_all_{YEAR}jja_{TAG}_best-loss_ep{EP}.csv"; df.to_csv(T, index=False)

def stats(x, y): return dict(r=np.corrcoef(x, y)[0, 1], bias=np.mean(x - y), rmse=np.sqrt(np.mean((x - y) ** 2)), ratio=x.mean() / y.mean())
asos = df.asos.to_numpy()
vmax = None
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), dpi=130)
for ax, x, title, xl in [(axes[0], df.hsr.to_numpy(), "HSR vs ASOS  (hourly)", "HSR radar hourly precip (mm/h)"),
                         (axes[1], df.pred.to_numpy(), "preds vs ASOS  (hourly)", "LIIF preds hourly precip (mm/h)")]:
    s = stats(x, asos); hb = ax.hexbin(x, asos, gridsize=GRID, extent=(0, LIM, 0, LIM), mincnt=1, norm=LogNorm(1, 8000), cmap="viridis")
    ax.plot([0, LIM], [0, LIM], "r--", lw=1, label="1:1"); b, a = np.polyfit(x, asos, 1); xx = np.array([0, LIM])
    ax.plot(xx, a + b * xx, "k-", lw=1.6, label=f"OLS: y = {a:.2f} + {b:.2f}x")
    ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlabel(xl); ax.set_ylabel("ASOS hourly precip (mm/h)")
    ax.set_title(f"{title}\nn={len(x)}  r={s['r']:.2f}  bias={s['bias']:.2f}  RMSE={s['rmse']:.2f}  ratio={s['ratio']:.2f}", fontsize=9); ax.legend(loc="upper left", fontsize=9)
    print(title, {k: round(float(v), 3) for k, v in s.items()}, f"OLS a={a:.3f} b={b:.3f}")
fig.suptitle(f"{YEAR} JJA  hourly precip  (nearest 500m grid, ASOS rain hours >= 0.1 mm/h)  |  2x LR, {PRE}-pretrain + adapter ft (2019), best-loss ep {EP}", fontsize=12)
fig.subplots_adjust(right=0.9); cax = fig.add_axes([0.93, 0.15, 0.013, 0.7]); fig.colorbar(hb, cax=cax, label="count (log)")
out = f"{H}/inference_tools/station_eval/figures/hexbin_hsr_preds_vs_asos/2x_ft/hexbin_hsr_preds_vs_asos_{YEAR}jja_{TAG}_best-loss_ep{EP}.png"
fig.savefig(out, bbox_inches="tight"); print("saved", out); print("csv", T)
