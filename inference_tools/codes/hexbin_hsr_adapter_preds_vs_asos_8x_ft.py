"""
hexbin_hsr_preds_vs_asos_2021jja_8x_2019_2020.py 와 같은 형식의 hexbin 4패널:
  1: HSR 500m                      vs ASOS
  2: 어댑터 입력 p_k (방향별 4개)        vs ASOS
  3: 어댑터에서 바로 나온 값 g(p_k) vs ASOS   (방향별 4개 = 지점·시각마다 4점, 같은 ASOS 를 4번 짝지음; 0 미만은 0 으로)
  4: 최종 pred (g(p_k) 면적가중평균 후 clamp) vs ASOS
p_k = 동결된 사전학습 imnet(_climate_edsr_radar_hsr_8x_minmax_liif/epoch-last) 의 방향별(local ensemble 4) 출력. 모델 간 공통, 어댑터만 다름.
  - 2019 : adapter_pk_cache_hsr_8x_2019_2020.npz 앞 28468행 (= 2019 학습 짝, obs 일치 확인) 을 pk_cache_8x_train2019.npz 로 복사해 재사용
  - 2021 : 처음 실행 때 GPU 2 로 추출해 pk_cache_8x_val2021.npz 에 저장 (val_dataset, 지점·시각 순서는 station_preds_dump_val2021_* 와 같음; obs 로 일치 검증)
사용:  python hexbin_hsr_adapter_preds_vs_asos_8x_ft.py        # THR=0 (필터 없음, 기본) / THR=0.1
출력:  station_eval/figures/hexbin_hsr_preds_vs_asos/hexbin_hsr_adapter_preds_vs_asos_<연도>jja_8x_<모델>_epoch-best-loss{_nofilter}.png
env: liif
"""
import os, sys, copy
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif")
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools"
S = "/home/inhye_yoo/deeplearning/inr/liif/save"
V = 74.8784
THR = float(os.environ.get("THR", "0"))
SUF = "_nofilter" if THR <= 0 else ""
LIM, GRID = 80, 60
CACHE19_SRC = "/tmp/claude-1018/-home-inhye-yoo/7ba25758-455d-42df-8513-9f96417dc540/scratchpad/adapter_pk_cache_hsr_8x_2019_2020.npz"
PK = {2019: f"{HERE}/station_eval/preds_npz/pk_cache_8x_train2019.npz", 2021: f"{HERE}/station_eval/preds_npz/pk_cache_8x_val2021.npz"}
NPZ = lambda tag, y: f"{HERE}/station_eval/preds_npz/" + {("19", 2019): "station_preds_train2019_8x_ft_epoch-best-loss.npz", ("19", 2021): "station_preds_val2021_8x_ft_epoch-best-loss.npz",
                                                           ("1920", 2021): "station_preds_val2021_8x_ft_2019_2020_epoch-best-loss.npz"}[(tag, y)]
ADAPTER = {"19": f"{S}/_climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h/epoch-best-loss.pth",
           "1920": f"{S}/_climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h_2019_2020/epoch-best-loss.pth"}
JOBS = [("19", 2019), ("19", 2021), ("1920", 2021)]       # (어댑터 학습기간, 평가연도)
LABEL = {"19": "8x_ft_2019only", "1920": "8x_ft_2019_2020"}
OUTD = f"{HERE}/station_eval/figures/hexbin_hsr_preds_vs_asos/"


def collect_pk_val2021():
    import yaml, datasets, models
    CROP = 48
    cfg = yaml.load(open("/home/inhye_yoo/deeplearning/inr/liif/configs/train_climate/climate_edsr_radar_hsr_8x_minmax_liif_ft_mlp_1e-3_lag1h.yaml"), Loader=yaml.FullLoader)
    sv = torch.load(f"{S}/_climate_edsr_radar_hsr_8x_minmax_liif/epoch-last.pth", map_location="cuda:0", weights_only=False)
    spec = copy.deepcopy(sv["model"]); model = models.make({"name": spec["name"], "args": spec["args"]}).to("cuda:0")
    model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
    store = []; model.imnet.register_forward_hook(lambda m, i, o: store.append(o.detach().float().cpu()))
    ds = datasets.make(cfg["val_dataset"]["dataset"]); crops, coords, obs = [], [], []
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
    crops_t = torch.stack(crops); coords_t = torch.stack(coords).unsqueeze(1); n = len(crops); print("val2021 쌍:", n, flush=True); P = []
    with torch.no_grad():
        for s in range(0, n, 512):
            inp = crops_t[s:s + 512].to("cuda:0"); co = coords_t[s:s + 512].to("cuda:0")
            cell = torch.ones_like(co); cell[:, :, 0] *= 2 / 1000; cell[:, :, 1] *= 2 / 1000
            store.clear(); model(inp, None, co, cell); assert len(store) == 4
            P.append(torch.stack([t.reshape(-1) for t in store], dim=1))
    np.savez(PK[2021], P=torch.cat(P).numpy() * V, O=np.array(obs) * V)


if not os.path.exists(PK[2019]):
    c = np.load(CACHE19_SRC); np.savez(PK[2019], P=c["P_train"][:28468], O=c["O_train"][:28468])
if not os.path.exists(PK[2021]):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"; collect_pk_val2021()


def adapter(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ad = {k.replace("add_layer.asos_adapter.", ""): v for k, v in ck["model"]["sd"].items() if "add_layer.asos_adapter" in k}
    L = []
    for a, b in zip([1, 32, 16, 32][:-1], [1, 32, 16, 32][1:]): L += [nn.Linear(a, b), nn.ReLU()]
    m = nn.Sequential(*L, nn.Linear(32, 1)); m.load_state_dict(ad); m.eval(); return m, ck.get("epoch")


def stats(x, y): return dict(r=np.corrcoef(x, y)[0, 1], bias=np.mean(x - y), rmse=np.sqrt(np.mean((x - y) ** 2)), ratio=x.mean() / y.mean())


for tag, yr in JOBS:
    d = np.load(NPZ(tag, yr)); c = np.load(PK[yr]); P = c["P"]
    assert np.abs(c["O"] - d["obs"]).max() < 1e-3, f"{yr}: p_k 캐시와 npz 의 (지점,시각) 순서/ASOS 불일치"
    net, ep = adapter(ADAPTER[tag])
    with torch.no_grad(): G = np.clip(net(torch.tensor(P / V, dtype=torch.float32).reshape(-1, 1)).numpy().reshape(P.shape) * V, 0, None)   # 방향별 어댑터 출력
    m = (d["obs"] >= THR) & np.isfinite(d["pf"]); asos, hsr, pred = d["obs"][m], d["hsr_hr"][m], d["pf"][m].astype(float); g = G[m].reshape(-1); a4 = np.repeat(asos, G.shape[1])
    pk_ = P[m].reshape(-1)                                                                                                        # 어댑터 입력 p_k (방향별)
    panels = [(hsr, asos, "HSR vs ASOS  (hourly)", "HSR radar hourly precip (mm/h)"),
              (pk_, a4, f"adapter input p_k vs ASOS  (per direction x{G.shape[1]})", "adapter input p_k (mm/h)"),
              (g, a4, f"adapter output g(p_k) vs ASOS  (per direction x{G.shape[1]})", "adapter output g(p_k) (mm/h)"),
              (pred, asos, "final preds vs ASOS  (hourly)", "LIIF preds hourly precip (mm/h)")]
    def cmax(x, y):
        f, a = plt.subplots(); h = a.hexbin(x, y, gridsize=GRID, extent=(0, LIM, 0, LIM), mincnt=1); v = h.get_array().max(); plt.close(f); return v
    vmax = max(cmax(x, y) for x, y, _, _ in panels)
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.2), dpi=130)
    for ax, (x, y, title, xl) in zip(axes, panels):
        s = stats(x, y); hb = ax.hexbin(x, y, gridsize=GRID, extent=(0, LIM, 0, LIM), mincnt=1, norm=LogNorm(1, vmax), cmap="viridis")
        ax.plot([0, LIM], [0, LIM], color="0.5", ls="--", lw=1, label="1:1"); b, a = np.polyfit(x, y, 1); xx = np.array([0, LIM])
        ax.plot(xx, a + b * xx, color="red", lw=1.6, label=f"OLS: y = {a:.2f} + {b:.2f}x")
        ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel(xl); ax.set_ylabel("ASOS hourly precip (mm/h)")
        ax.set_title(f"{title}\nn={len(x)}  r={s['r']:.2f}  bias={s['bias']:.2f}  RMSE={s['rmse']:.2f}  ratio={s['ratio']:.2f}", fontsize=9); ax.legend(loc="upper left", fontsize=9)
        print(yr, tag, title, {k: round(float(v), 3) for k, v in s.items()}, f"OLS a={a:.3f} b={b:.3f}")
    fig.suptitle(f"{yr} JJA  hourly precip  (nearest 500m grid, {'all hours, no filter' if THR <= 0 else f'ASOS rain hours >= {THR} mm/h'})  |  adapter: {LABEL[tag]} (ep {ep})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.955, 0.96)); cax = fig.add_axes([0.965, 0.15, 0.01, 0.7]); cb = fig.colorbar(hb, cax=cax); cb.set_label("count (log)")
    out = f"{OUTD}hexbin_hsr_adapter_preds_vs_asos_{yr}jja_{LABEL[tag]}_epoch-best-loss{SUF}_4panel.png"; fig.savefig(out); plt.close(fig); print("saved", out)
