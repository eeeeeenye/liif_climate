"""
2x LIIF 사전학습(동결 imnet) 의 방향별 p_k(4) 와 local-ensemble 가중치 w_k(4, model.le_weights = 면적가중치, pred 순서에 맞게 교환된 것) 를 추출해 저장.
  weighted p_k = sum_k w_k * p_k  (= 모델의 어댑터 전 blend)
사용: python collect_pk_weights_8x.py          # train2019, val2021 둘 다, GPU 2
출력: inference_tools/station_eval/preds_npz/pk_w_cache_2x_{train2019,val2021}.npz  (P[N,4], W[N,4], O[N] : mm/h, 무차원, mm/h)
env: liif
"""
import os, sys, copy
sys.path.insert(0, "/home/inhye_yoo/deeplearning/inr/liif"); os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np, torch, yaml, datasets, models
S = "/home/inhye_yoo/deeplearning/inr/liif/save"; V = 74.8784; CROP = 48
OUTD = "/home/inhye_yoo/deeplearning/inr/liif/inference_tools/station_eval/preds_npz/"
cfg = yaml.load(open("/home/inhye_yoo/deeplearning/inr/liif/configs/train_climate/climate_edsr_radar_hsr_2x_minmax_liif_ft_mlp_1e-3_lag1h.yaml"), Loader=yaml.FullLoader)
sv = torch.load(f"{S}/_climate_edsr_radar_hsr_2x_minmax_liif/epoch-last.pth", map_location="cuda:0", weights_only=False)
spec = copy.deepcopy(sv["model"]); model = models.make({"name": spec["name"], "args": spec["args"]}).to("cuda:0")
model.load_state_dict(spec["sd"], strict=True); model.eval(); model.topo = None
store = []; model.imnet.register_forward_hook(lambda m, i, o: store.append(o.detach().float().cpu()))
for tag, key in (("val2021", "val_dataset"),):
    ds = datasets.make(cfg[key]["dataset"]); crops, coords, obs = [], [], []
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
    crops_t = torch.stack(crops); coords_t = torch.stack(coords).unsqueeze(1); n = len(crops); print(tag, "쌍:", n, flush=True); P, W = [], []
    with torch.no_grad():
        for s in range(0, n, 512):
            inp = crops_t[s:s + 512].to("cuda:0"); co = coords_t[s:s + 512].to("cuda:0")
            cell = torch.ones_like(co); cell[:, :, 0] *= 2 / 1000; cell[:, :, 1] *= 2 / 1000
            store.clear(); model(inp, None, co, cell); assert len(store) == 4
            P.append(torch.stack([t.reshape(-1) for t in store], dim=1))
            W.append(torch.stack([w.reshape(-1).cpu() for w in model.le_weights], dim=1))
    np.savez(f"{OUTD}pk_w_cache_2x_{tag}.npz", P=torch.cat(P).numpy() * V, W=torch.cat(W).numpy(), O=np.array(obs) * V); print(tag, "saved", flush=True)
