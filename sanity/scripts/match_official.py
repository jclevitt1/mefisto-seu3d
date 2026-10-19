"""Recover gene names and barcodes for the paper's anonymised ST_model.hdf5 (figshare 13233860, file 25659233).
Spots: the official covariates are Seurat GetTissueCoordinates (imagerow, imagecol in low-res pixels), so match to 10x spots by
low-res pixel position. Genes: the official matrix is Seurat NormalizeData (log1p(1e4 * count / total)), so match each official
column to the 10x gene whose log-normalised profile over the matched spots is identical. Writes data/official_input.h5ad
(the paper's exact 2,696 x 2,000 input matrix w real names) plus data/official_match.json w the match statistics."""
import h5py, numpy as np, pandas as pd, anndata as ad, scanpy as sc, json
from scipy.spatial import cKDTree
D = 'data/'
f = h5py.File(D + 'ST_model_official.hdf5', 'r')
cov = f['cov_samples/group0'][:]                      # (2696, 2): covariate0, covariate1
Y = f['data/view0/group0'][:] + f['intercepts/view0/group0'][:]   # centred data + per-gene intercept = input matrix
a = ad.read_h5ad(D + 'visium_anterior1_raw.h5ad')
sf = json.load(open(D + 'visium/V1_Mouse_Brain_Sagittal_Anterior/spatial/scalefactors_json.json'))['tissue_lowres_scalef']
px = a.obsm['spatial'].astype(float)                  # 10x: (x = column, y = row) in full-res pixels
low = np.c_[px[:, 0] * sf, px[:, 1] * sf]             # official covariate0 = imagecol? test both orders
best = None
for order, L in [('col,row', low), ('row,col', low[:, ::-1])]:
    d, idx = cKDTree(L).query(cov)
    if best is None or np.median(d) < best[1]: best = (order, np.median(d), d, idx)
order, med, d, idx = best
print('covariate order', order, 'median px distance', med, 'max', d.max(), 'spots within 0.5 px', (d < 0.5).sum(), 'unique', len(set(idx)))
ok = d < 0.5
# Seurat NormalizeData on the 10x counts
b = a.copy(); sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
Xb = b.X.toarray() if hasattr(b.X, 'toarray') else b.X
Xb = Xb[idx]                                          # rows aligned to official spots (where ok)
# match genes: for each official column, find the 10x gene w the same values on matched spots (hash of rounded values)
key = {}
for j in range(Xb.shape[1]):
    v = Xb[ok, j]
    if v.any(): key.setdefault(np.round(v, 4).tobytes(), []).append(j)
genes, n_multi, n_miss = [], 0, 0
for j in range(Y.shape[1]):
    hits = key.get(np.round(Y[ok, j], 4).tobytes(), [])
    if len(hits) == 0:
        # fall back to best correlation
        c = np.array([np.corrcoef(Y[ok, j], Xb[ok, g])[0, 1] if Xb[ok, g].std() > 0 else 0 for g in range(Xb.shape[1])])
        g = int(np.nanargmax(c)); genes.append(b.var_names[g]); n_miss += 1
        print('  no exact match for official feature', j, '-> best corr', b.var_names[g], round(float(c[g]), 4))
    else:
        if len(hits) > 1: n_multi += 1
        genes.append(b.var_names[hits[0]])
genes = pd.Index(genes)
print('genes matched exactly:', Y.shape[1] - n_miss, 'ambiguous (identical profiles):', n_multi, 'unmatched:', n_miss)
print('first 10 official features (Seurat vst rank order):', list(genes[:10]))
print('Ttr position:', genes.get_loc('Ttr') if 'Ttr' in genes else None)
# the spots: 2,696 official vs 2,695 in the 10x filtered matrix
spot_names = np.where(ok, a.obs_names.values[idx], [f'official_spot{i}' for i in range(len(idx))])
off = ad.AnnData(X=Y.astype(np.float32), obs=pd.DataFrame({'matched': ok, 'imagerow': cov[:, 0] if order == 'row,col' else cov[:, 1],
                 'imagecol': cov[:, 1] if order == 'row,col' else cov[:, 0]}, index=spot_names), var=pd.DataFrame(index=genes))
off.obsm['spatial_lowres'] = off.obs[['imagerow', 'imagecol']].to_numpy()
off.var_names_make_unique()
off.write_h5ad(D + 'official_input.h5ad')
# per-spot sanity check: does the official matrix equal Seurat NormalizeData of the 10x counts on the matched genes?
gi = [b.var_names.get_loc(g) for g in genes[:Y.shape[1]] if g in b.var_names]
diff = np.abs(Y[ok] - Xb[ok][:, gi]).max()
json.dump(dict(cov_order=order, median_px=float(med), n_spots_matched=int(ok.sum()), n_spots_official=int(len(idx)), n_genes_exact=int(Y.shape[1] - n_miss),
               n_genes_ambiguous=n_multi, max_abs_diff_matrix=float(diff), ttr_rank=int(genes.get_loc('Ttr')) if 'Ttr' in genes else None,
               first_features=list(genes[:10])), open(D + 'official_match.json', 'w'), indent=1)
print('max |official - my Seurat-style lognorm| on matched spots x genes:', diff)
print(off)
