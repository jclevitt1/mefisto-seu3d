"""The MEFISTO paper's Visium recipe, applied to the E8.0 embryos.
Usage: paper_recipe.py prep            -> data/e8_joint_2500per_vst.h5ad and data/pca_ceiling.csv
       paper_recipe.py fit K [nogp]    -> models/paper_e8_vst_K{K}[_mofa].hdf5
Differences from prep.py / fit.py: HVGs by Seurat vst on counts (prep.py:19 uses dispersion on log data), and the fit settings
saved in the paper's ST_model.hdf5 (1,000 inducing points, n_grid 20, GP step from iteration 20 every 10, fast convergence,
ELBO every iteration, seed 2021, float64). Same cells (2,500 per embryo, seed 0), same counts, same covariates (x, y, z)."""
import os, sys, time, json, numpy as np, pandas as pd, anndata as ad, scanpy as sc, scipy.sparse as sp

def prep():
    r1 = ad.read_h5ad('data/GSM9046247_Embryo_E8.0_stereo_rep1.h5ad')
    r2 = ad.read_h5ad('data/GSM9046248_Embryo_E8.0_stereo_rep2.h5ad')
    cols = ['x', 'y', 'z', 'chip', 'germ_layer', 'celltype', 'nCount_RNA', 'nFeature_RNA']
    a1 = ad.AnnData(X=sp.csr_matrix(r1.layers['counts']), obs=r1.obs[cols].copy(), var=pd.DataFrame(index=r1.var_names))
    a2 = ad.AnnData(X=sp.csr_matrix(r2.raw.X), obs=r2.obs[cols].copy(), var=pd.DataFrame(index=r2.var_names))
    a1.obs['embryo'] = 'E8.0_rep1'; a2.obs['embryo'] = 'E8.0_rep2'
    genes = a1.var_names.intersection(a2.var_names)
    a = ad.concat([a1[:, genes], a2[:, genes]], index_unique='-')
    a.obs['celltype'] = a.obs['celltype'].astype(str).replace({'pre-somitic': 'Pre-somitic'})
    a.obs['embryo'] = a.obs['embryo'].astype('category')
    rng = np.random.default_rng(0)       # same subsample as prep.py 2500
    keep = np.concatenate([rng.choice(np.where(a.obs.embryo == e)[0], size=min(2500, (a.obs.embryo == e).sum()), replace=False)
                           for e in a.obs.embryo.cat.categories])
    a = a[np.sort(keep)].copy()
    a.layers['sct_counts'] = a.X.copy()
    full = a.copy(); sc.pp.normalize_total(full, target_sum=1e4); sc.pp.log1p(full)
    # presented gene set (prep.py:19) and the paper's (vst on counts), both on the same normalised matrix
    disp = full.copy(); sc.pp.highly_variable_genes(disp, n_top_genes=2000, batch_key='embryo', flavor='seurat')
    sc.pp.highly_variable_genes(a, n_top_genes=2000, batch_key='embryo', flavor='seurat_v3', layer='sct_counts')
    vst = full[:, a.var.highly_variable].copy(); disp = disp[:, disp.var.highly_variable].copy()
    vst.obsm['spatial3d'] = vst.obs[['x', 'y', 'z']].to_numpy(dtype=float)
    vst.write_h5ad('data/e8_joint_2500per_vst.h5ad')
    # PCA ceiling: per embryo, genes centred within the embryo (as mofapy2 does), percent of total sum of squares
    rows = []
    for name, m in [('dispersion (presented)', disp), ('vst (paper)', vst)]:
        for e in m.obs.embryo.cat.categories:
            X = m[m.obs.embryo == e].X; X = X.toarray() if sp.issparse(X) else np.asarray(X)
            zeros = (X == 0).mean()
            s2 = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
            rows.append(dict(genes=name, embryo=e, zeros=round(100 * zeros, 1),
                             pc4=round(100 * s2[:4].sum() / s2.sum(), 2), pc12=round(100 * s2[:12].sum() / s2.sum(), 2)))
    print('genes shared by the two sets:', len(set(disp.var_names) & set(vst.var_names)))
    pd.DataFrame(rows).to_csv('pca_ceiling.csv', index=False); print(pd.DataFrame(rows).to_string())

def fit(K, use_gp):
    from mofapy2.run.entry_point import entry_point
    a = ad.read_h5ad('data/e8_joint_2500per_vst.h5ad'); a.obs['sample'] = a.obs_names
    a.X = np.asarray(a.X.toarray() if sp.issparse(a.X) else a.X, dtype=np.float64)
    out = f'models/paper_e8_vst_K{K}' + ('' if use_gp else '_mofa') + ('_' + os.environ['CONV'] if 'CONV' in os.environ else '') + '.hdf5'
    t0 = time.time(); ent = entry_point()
    ent.set_data_options(scale_views=False, center_groups=True, use_float32=False)
    ent.set_data_from_anndata(a, groups_label='embryo', use_raw=False, save_metadata=True)
    ent.set_model_options(factors=K, spikeslab_weights=True, ard_weights=True, ard_factors=not use_gp)
    ent.set_train_options(iter=1000, convergence_mode=os.environ.get('CONV', 'fast'), startELBO=1, freqELBO=1, seed=2021, verbose=False, quiet=False, gpu_mode=False)
    if use_gp:
        ent.set_covariates(['x', 'y', 'z'], covariates_names=['x', 'y', 'z'])
        ent.set_smooth_options(scale_cov=False, model_groups=True, sparseGP=True, frac_inducing=1000 / a.n_obs, n_grid=20, start_opt=20, opt_freq=10)
    ent.build(); t1 = time.time(); ent.run(); t2 = time.time()
    ent.save(out, save_data=False)
    meta = dict(output=out, K=K, mefisto=use_gp, n_cells=int(a.n_obs), n_genes=int(a.n_vars), build_s=round(t1 - t0, 1), train_s=round(t2 - t1, 1))
    json.dump(meta, open(out.replace('.hdf5', '.json'), 'w'), indent=1); print(meta)

if __name__ == '__main__':
    if sys.argv[1] == 'prep': prep()
    else: fit(int(sys.argv[2]), not (len(sys.argv) > 3 and sys.argv[3] == 'nogp'))
