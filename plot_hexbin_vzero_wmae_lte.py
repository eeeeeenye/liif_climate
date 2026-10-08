"""vzero+WMAE LTE adapter 검증(2023 JJA) 예측 vs ASOS 라벨 hexbin. 사용: python plot_hexbin_vzero_wmae_lte.py [--ckpt epoch-best-loss.pth]"""
import argparse, os, yaml
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import models
import train_liif_finetune_vzero_wmae_lte as m

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='save/_climate_edsr_radar_hsr_2x_minmax_lte_topo_cap63clip_ft_vzero_wmae_2019_2022')
ap.add_argument('--ckpt', default='epoch-best-loss.pth')
ap.add_argument('--out', default=None)
a = ap.parse_args()

m.config = yaml.load(open(os.path.join(a.dir, 'config.yaml')), Loader=yaml.FullLoader)
m.log = print
pm = m.config['physical_norm']['max'] - m.config['physical_norm'].get('min', 0.0)
spec = m.config['val_dataset']
loader = m.make_data_loader(spec, tag='val')
sv = torch.load(os.path.join(a.dir, a.ckpt))
model = models.make(sv['model'], load_sd=True).cuda().eval()

P, G = [], []
with torch.no_grad():
    for b in loader:
        if b is None: continue
        for k, v in b.items():
            if torch.is_tensor(v): b[k] = v.cuda()
        pr = model(b['inp'], b.get('level'), b['coord'], b['cell'])
        mb = b['mask'].unsqueeze(-1).bool().expand_as(b['gt'])
        P.append(pr.clamp(0, 1)[mb].cpu()); G.append(b['gt'][mb].cpu())
p = torch.cat(P).numpy() * pm; g = torch.cat(G).numpy() * pm

lim = 60
fig, ax = plt.subplots(figsize=(6.4, 5.6))
hb = ax.hexbin(g, p, gridsize=70, extent=(0, lim, 0, lim), bins='log', mincnt=1, cmap='viridis', norm=LogNorm())
ax.plot([0, lim], [0, lim], 'r--', lw=1, label='1:1')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('ASOS-based label (mm/h)'); ax.set_ylabel('Prediction (mm/h)')
r = np.corrcoef(p, g)[0, 1]; e = p - g
ax.set_title(f'{os.path.basename(a.ckpt)}  n={len(p):,}\nr={r:.3f} MAE={np.abs(e).mean():.3f} bias={e.mean():+.3f} RMSE={np.sqrt((e**2).mean()):.3f}', fontsize=10)
fig.colorbar(hb, label='count'); ax.legend(loc='upper left')
out = a.out or os.path.expanduser('~/hexbin_hsr_2x_lte_cap63clip_ft_vzero_wmae.png')
fig.tight_layout(); fig.savefig(out, dpi=150); print('saved', out, f'r={r:.3f}')
