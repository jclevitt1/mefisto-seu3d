"""Figures: factors in space for a list of models (one row per model), in the paper's orientation (tissue rotated as in
plot_factors_vs_cov(rotate_y=TRUE)), smoothness score under each panel; plus a side-by-side w ED Fig. 9."""
import os, sys, json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c, anndata as ad
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('p', ['#d8d7d3', '#f3b7a6', '#f4f3f0', '#9fb3dd', '#2b3e8c'])   # grey-red / white / blue like the paper
def factors_in_space(paths, names, out, max_k=12, official=None):
    rows = []
    for p, n in zip(paths, names):
        m = c.load(p); act = c.active(m); rows.append((n, m, act[:max_k]))
    K = max(len(a) for _, _, a in rows)
    fig, axes = plt.subplots(len(rows), K, figsize=(1.9 * K, 2.3 * len(rows)), squeeze=False)
    for i, (n, m, act) in enumerate(rows):
        cov = m['cov']; x, y = cov[:, 1], -cov[:, 0]                                   # imagecol right, imagerow down
        for j in range(K):
            ax = axes[i, j]
            if j >= len(act): ax.axis('off'); continue
            k = act[j]; z = m['Z'][k].to_numpy(); lim = np.percentile(np.abs(z), 99)
            ax.scatter(x, y, c=z, s=3, cmap='RdBu', norm=TwoSlopeNorm(0, -lim, lim), lw=0, rasterized=True); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            sm = f"  s={m['smoothness'][k]:.2f}" if 'smoothness' in m else ''
            tg = m['W'][k].abs().sort_values(ascending=False).index[:2]
            ax.set_title(f"{k}  R2 {m['r2'][k]:.2f}%{sm}\n{', '.join(tg)}", fontsize=7)
            if j == 0: ax.set_ylabel(f"{n}\ntotal R2 {m['r2_total']:.2f}%", fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
if __name__ == '__main__':
    import glob
    paths = sys.argv[2:] if len(sys.argv) > 2 else sorted(glob.glob(os.path.join(c.M, '*.hdf5')))
    factors_in_space(paths, [os.path.basename(p).replace('.hdf5', '') for p in paths], sys.argv[1] if len(sys.argv) > 1 else os.path.join(c.F, 'factors_all.png'))
