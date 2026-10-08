"""[새 파일] 특정 (지점,시각) 점들에서 C(lte_ctx) 와 기준선(lte)의 예측을 항별로 분해.
   최종 = sum_k w_k g_k + LR bilinear(질의 위치).  p_k = imnet 원출력, g_k = 어댑터 출력.  (정규화 공간 -> mm/h 환산해서 표시)
   C 는 어댑터 문맥 특징을 일부 0 으로 만든 ablation 예측도 같이 출력.
사용: python exp_ctx_binloss/diag_points.py     (RUN_C, RUN_B 환경변수로 체크포인트 폴더 지정 가능)
"""
import sys, os, copy, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np, torch, yaml
import datasets, models, models_ctx  # noqa
PM, CROP = 74.8784, 48; dev = torch.device("cuda:0")
SC = "/tmp/claude-1018/-home-inhye-yoo/4e9fbb29-fb41-4952-accd-5396550b8bc4/scratchpad"
RUN_C = os.environ.get("RUN_C", f"{SC}/snap_C"); RUN_B = os.environ.get("RUN_B", f"{ROOT}/save/_climate_edsr_radar_hsr_8x_minmax_lte_ft_mlp_1e-3_lag1h_2019_2020_wmae")
d = np.load(f"{ROOT}/exp_ctx_binloss/results/station_preds_val2021_snap_C_epoch-best-loss.npz")
sel = np.where((d["pf"] >= 30) & (d["obs"] <= 10))[0]; print("대상 점", len(sel))
cfg = yaml.load(open(f"{RUN_C}/config.yaml"), Loader=yaml.FullLoader); val_ds = datasets.make(cfg['val_dataset']['dataset'])
meta = json.load(open("/home/inhye_yoo/data/observation/raw/ASOS/hourly/hsr_obs/meta_HSR_500m_2019JJA_obscrop_lag1h_val.json")); lat_max, lat_min = meta['lat_range']; lon_min, lon_max = meta['lon_range']
def load(run, ck):
    sv = torch.load(f"{run}/{ck}.pth", map_location='cuda:0', weights_only=False); sp = copy.deepcopy(sv['model']); m = models.make({"name": sp["name"], "args": sp["args"]}).to(dev)
    m.load_state_dict(sp["sd"], strict=True); m.eval(); m.topo = None; return m, sv['epoch']
mc, epc = load(RUN_C, "epoch-best-loss"); mb, epb = load(RUN_B, "epoch-best-loss"); print(f"C ep{epc}, 기준선 ep{epb}")
def point_inputs(i):
    t, la, lo = int(d["tid"][i]), float(d["lat"][i]), float(d["lon"][i]); img_lr, img_hr, obs_, _ = val_ds[t]; H, W = img_lr.shape[-2:]
    y_ = (la - lat_min) / (lat_max - lat_min) * 2 - 1; x_ = (lo - lon_min) / (lon_max - lon_min) * 2 - 1       # 지점 좌표(정규화), eval 과 같은 변환의 역
    ia = (y_ + 1) / 2 * (H - 1); ja = (x_ + 1) / 2 * (W - 1)
    x0 = min(max(int(round(ia - CROP / 2)), 0), H - CROP); y0 = min(max(int(round(ja - CROP / 2)), 0), W - CROP)
    crop = img_lr[:, x0:x0 + CROP, y0:y0 + CROP].unsqueeze(0).to(dev); co = torch.tensor([[[((ia - x0) / (CROP - 1)) * 2 - 1, ((ja - y0) / (CROP - 1)) * 2 - 1]]], dtype=torch.float32).to(dev)
    cell = torch.ones_like(co) * 2 / 1000; return crop, co, cell
rows = []
with torch.no_grad():
    for i in sel:
        crop, co, cell = point_inputs(i)
        # --- C
        mc.gen_feat(crop); mc.topo = None; out = mc.query_rgb(co, cell)
        preds = torch.cat(mc.le_preds, -1); w = torch.stack(mc.le_weights, -1); feats, lr = mc._context_feats(preds, co)
        g = mc.le_g; res = (g - preds)
        def pred_with(fe): return float(((preds + mc.add_layer(fe).squeeze(-1)) * w).sum(-1) + lr.squeeze()) * PM
        f_noctx = feats.clone(); f_noctx[..., 1:] = 0; f_nolr = feats.clone(); f_nolr[..., 4:] = 0
        # --- 기준선: 어댑터 없이 원 p_k 를 얻은 뒤 어댑터를 따로 적용
        mb.gen_feat(crop); mb.topo = None; mb.adapter_per_direction = False; _ = mb.query_rgb(co, cell); pb = torch.cat(mb.le_preds, -1); wb = torch.stack(mb.le_weights, -1)
        gb = mb.add_layer(pb.unsqueeze(-1)).squeeze(-1).clamp(min=0) if False else mb.add_layer(pb.unsqueeze(-1)).squeeze(-1); mb.adapter_per_direction = True
        pb_pred = float((gb * wb).sum(-1) + lr.squeeze()) * PM
        rows.append(dict(asos=d["obs"][i], hsr=d["hsr_hr"][i], lr=float(lr.squeeze()) * PM, lrmax=float(feats[0, 0, 0, 5]) / mc.in_scale * PM,
                         pk=(preds[0, 0] * PM).cpu().numpy(), g=(g[0, 0] * PM).cpu().numpy(), res=(res[0, 0] * PM).cpu().numpy(), predC=float(out.squeeze()) * PM,
                         ab_noctx=pred_with(f_noctx), ab_nolr=pred_with(f_nolr), pk_b=(pb[0, 0] * PM).cpu().numpy(), g_b=(gb[0, 0] * PM).cpu().numpy(), predB=pb_pred))
print("\n(모두 mm/h)  최종 = Σ w·g + LR보간값 (LR=저해상도 입력의 질의 위치 bilinear, 8x 입력이라 약 4km 해상도)")
for k, r in enumerate(rows):
    print(f"\n[{k}] lat {d['lat'][sel[k]]:.2f} lon {d['lon'][sel[k]]:.2f}  ASOS {r['asos']:.1f}  HSR(500m) {r['hsr']:.1f}  | LR 보간값 {r['lr']:.1f}  LR 3x3 최대 {r['lrmax']:.1f}")
    print(f"   C   : p_k {np.round(r['pk'],1)} → 어댑터 보정량(g−p) {np.round(r['res'],1)} → g_k {np.round(r['g'],1)} → 최종 {r['predC']:.1f}   (재현검증: eval 값 {d['pf'][sel[k]]:.1f})")
    print(f"        ablation: 문맥 특징을 0 으로 {r['ab_noctx']:.1f} | LR 특징(lr, lrmax)만 0 으로 {r['ab_nolr']:.1f}")
    print(f"   기준선: p_k {np.round(r['pk_b'],1)} → g_k {np.round(r['g_b'],1)} → 최종 {r['predB']:.1f}")
