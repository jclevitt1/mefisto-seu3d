"""Build aligned covariates (ap, lr, z) per embryo for the common-frame MEFISTO refit.
ap: the anterior-posterior axis from the anterior/posterior cell-type centroids (analysis.ap_coords), sign flipped per embryo so
anterior types sit at positive ap. lr: in-plane perpendicular. z: section axis as published (starts at 0). ap and lr are
centred at each embryo's own cell centroid."""
import json, numpy as np, pandas as pd, anndata as ad, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from analysis import ANTERIOR, POSTERIOR, BLUE, ORANGE
FLIP_TYPES = ['Cardiomyocytes', 'Pharyngeal mesoderm', 'Rostral neurectoderm']
a = ad.read_h5ad('data/e8_joint_2500per.h5ad')
rot = {}
for g in a.obs.embryo.cat.categories if hasattr(a.obs.embryo, 'cat') else a.obs.embryo.unique():
    o = a.obs[a.obs.embryo == g]
    ca = o.loc[o.celltype.isin(ANTERIOR), ['x', 'y']].mean().to_numpy(); cp = o.loc[o.celltype.isin(POSTERIOR), ['x', 'y']].mean().to_numpy()
    u = ca - cp; u /= np.linalg.norm(u); v = np.array([-u[1], u[0]])
    xy = o[['x', 'y']].to_numpy(); c = xy.mean(0)
    ap, lr = (xy - c) @ u, (xy - c) @ v
    flip = 1.0 if ap[o.celltype.isin(FLIP_TYPES).to_numpy()].mean() > 0 else -1.0
    ap = ap * flip
    a.obs.loc[o.index, 'ap'] = ap; a.obs.loc[o.index, 'lr'] = lr
    rot[g] = dict(angle_deg=float(np.degrees(np.arctan2(u[1], u[0]))), unit_ap=[float(u[0]), float(u[1])], flip=flip, centroid_xy=[float(c[0]), float(c[1])],
                  anterior_mean_ap=float(ap[o.celltype.isin(FLIP_TYPES).to_numpy()].mean()), posterior_mean_ap=float(ap[o.celltype.isin(POSTERIOR).to_numpy()].mean()),
                  ap_range=[float(ap.min()), float(ap.max())], lr_range=[float(lr.min()), float(lr.max())], z_range=[float(o.z.min()), float(o.z.max())])
a.obs['ap'] = a.obs['ap'].astype(float); a.obs['lr'] = a.obs['lr'].astype(float)
a.obsm['aligned'] = a.obs[['ap', 'lr', 'z']].to_numpy()
a.write_h5ad('data/e8_joint_2500per_aligned.h5ad')
json.dump(rot, open('data/aligned_rotation.json', 'w'), indent=1); print(json.dumps(rot, indent=1))
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
for ax, (g, col) in zip(axes, zip(rot, [BLUE, ORANGE])):
    o = a.obs[a.obs.embryo == g]
    ax.scatter(o.ap, o.lr, s=3, c='#c9c8c4', lw=0)
    ant, post = o.celltype.isin(FLIP_TYPES), o.celltype.isin(POSTERIOR)
    ax.scatter(o.ap[ant], o.lr[ant], s=6, c=col, lw=0, label='anterior types (heart, pharyngeal, rostral neurect.)')
    ax.scatter(o.ap[post], o.lr[post], s=6, c='#52514e', lw=0, label='posterior types')
    ax.axvline(0, color='#b0afab', lw=0.8); ax.set_title(f'{g}: rotation {rot[g]["angle_deg"]:.1f} deg, flip {rot[g]["flip"]:+.0f}')
    ax.set_xlabel('ap (um, anterior positive)'); ax.set_aspect('equal')
axes[0].set_ylabel('lr (um)'); axes[0].legend(loc='lower left', fontsize=7, frameon=False)
fig.suptitle('Aligned frame check: anterior cell types at positive ap in both embryos', fontsize=10)
fig.tight_layout(); fig.savefig('figures/aligned_frame_check.png', dpi=150)
