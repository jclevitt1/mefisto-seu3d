"""Shared code for the sanity test: preprocessing variants, one parameterised mofapy2 fit, and a summary of a saved model.
Does not touch the SEU-3D data or the presented scripts- prep.py / fit.py logic is copied verbatim where marked."""
import os, sys, time, json, numpy as np, pandas as pd, anndata as ad, scanpy as sc, h5py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # seu3d/sanity
D = os.path.join(ROOT, 'data'); M = os.path.join(ROOT, 'models'); L = os.path.join(ROOT, 'logs'); F = os.path.join(ROOT, 'figures')
for p in (D, M, L, F): os.makedirs(p, exist_ok=True)
RAW = os.path.join(D, 'visium_anterior1_raw.h5ad')          # 10x V1_Mouse_Brain_Sagittal_Anterior, 2,695 spots x 32,285 genes, raw UMI
OFFICIAL = os.path.join(D, 'official_input.h5ad')           # the paper's exact input matrix (figshare model, names recovered)
OFFICIAL_MODEL = os.path.join(D, 'ST_model_official.hdf5')

def lowres_coords(a):
    sf = json.load(open(os.path.join(D, 'visium/V1_Mouse_Brain_Sagittal_Anterior/spatial/scalefactors_json.json')))['tissue_lowres_scalef']
    px = a.obsm['spatial'].astype(float)
    return np.c_[px[:, 1] * sf, px[:, 0] * sf]            # (imagerow, imagecol), Seurat GetTissueCoordinates convention

def preprocess(variant, n_hvg=2000):
    """Return an AnnData (spots x HVGs, log-normalised X) and the covariate matrix (imagerow, imagecol, low-res pixels).
    'official'   : the paper's matrix exactly as saved in ST_model.hdf5 (Seurat NormalizeData + FindVariableFeatures vst).
    'seurat'     : my mirror of the R tutorial on the 10x counts: normalize_total(1e4) + log1p, 2,000 HVGs by Seurat vst (scanpy seurat_v3 on counts).
    'ours'       : prep.py lines 17-20 on the 10x raw counts: normalize_total(1e4), log1p, 2,000 HVGs flavor='seurat' (dispersion on log data).
    'ours_vst'   : prep.py normalisation but the official HVG method (vst on counts).            [ablation: HVG filter]
    'ours_hvgall': prep.py path, HVGs restricted to genes detected in >= 10 spots first.        [ablation: near-zero genes]
    'seurat_disp': official-style but prep.py's HVG flavour (isolates the HVG method).
    'ours_thin<N>' / 'seurat_thin<N>': counts binomially thinned to N UMIs per spot (seed 0) before the 'ours' / 'seurat' path:
                   the paper's tissue at Stereo-seq-bin depth (SEU-3D E8.0 rep1 median nCount_SCT = 1,181), to size the depth effect.
    
    """
    thin = None
    if '_thin' in variant:
        variant, thin = variant.split('_thin'); thin = int(thin)
    if variant == 'official':
        a = ad.read_h5ad(OFFICIAL); a.X = np.asarray(a.X, dtype=np.float64)
        return a, a.obsm['spatial_lowres'].astype(float)
    a = ad.read_h5ad(RAW); a.var_names_make_unique()
    cov = lowres_coords(a); a.obsm['lowres'] = cov
    if thin is not None:
        X = a.X.toarray().astype(np.int64); rng = np.random.default_rng(0); tot = X.sum(1)
        p = np.minimum(1.0, thin / np.maximum(tot, 1))
        a.X = sc._utils.get_literal_vals if False else np.asarray(rng.binomial(X, p[:, None]), dtype=np.float32)
    a.layers['counts'] = a.X.copy()
    if variant == 'ours_hvgall':
        sc.pp.filter_genes(a, min_cells=10)
    if variant in ('seurat', 'ours_vst'):
        sc.pp.highly_variable_genes(a, flavor='seurat_v3', n_top_genes=n_hvg, layer='counts')   # Seurat FindVariableFeatures(selection.method='vst')
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)                                   # prep.py line 18 == Seurat NormalizeData
    if variant in ('ours', 'ours_hvgall', 'seurat_disp'):
        sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor='seurat')                     # prep.py line 19 (batch_key dropped: one section)
    a = a[:, a.var.highly_variable].copy()                                                       # prep.py line 20
    return a, a.obsm['lowres'].astype(float)

def fit(a, cov, out, K=12, frac_inducing=0.1, n_grid=10, opt_freq=25, start_opt=None, convergence='medium', iters=300, seed=0,
        scale_views=False, center_groups=True, use_float32=True, freqELBO=5, ard_factors=True, spikeslab_weights=True, ard_weights=True,
        use_gp=True, sparseGP=True, scale_cov=False, log=None):
    """One mofapy2 fit. Defaults == fit.py lines 10-16 with OPTFREQ=25 (the presented fit). Returns metadata dict."""
    from mofapy2.run.entry_point import entry_point
    start_opt = opt_freq if start_opt is None else start_opt
    a = a.copy(); a.obs['sample'] = a.obs_names; a.obs['group'] = 'g1'
    a.obs['imagerow'] = cov[:, 0]; a.obs['imagecol'] = cov[:, 1]
    if hasattr(a.X, 'toarray'): a.X = a.X.toarray()
    a.X = np.asarray(a.X, dtype=np.float32 if use_float32 else np.float64)
    t0 = time.time()
    ent = entry_point()
    ent.set_data_options(scale_views=scale_views, center_groups=center_groups, use_float32=use_float32)
    ent.set_data_from_anndata(a, groups_label='group', use_raw=False, save_metadata=True)
    ent.set_model_options(factors=K, spikeslab_weights=spikeslab_weights, ard_weights=ard_weights, ard_factors=ard_factors)
    ent.set_train_options(iter=iters, convergence_mode=convergence, startELBO=1, freqELBO=freqELBO, seed=seed, verbose=False, quiet=False, gpu_mode=False)
    if use_gp:
        ent.set_covariates(['imagerow', 'imagecol'], covariates_names=['imagerow', 'imagecol'])
        ent.set_smooth_options(scale_cov=scale_cov, model_groups=True, sparseGP=sparseGP, frac_inducing=frac_inducing if sparseGP else None,
                               n_grid=n_grid, start_opt=start_opt, opt_freq=opt_freq)
    ent.build()
    t1 = time.time(); ent.run(); t2 = time.time()
    ent.save(out, save_data=True)
    meta = dict(output=out, K=K, frac_inducing=frac_inducing if (use_gp and sparseGP) else None,
                n_inducing=int(ent.smooth_opts.get('n_inducing', 0)) if use_gp else 0, n_grid=n_grid, opt_freq=opt_freq, start_opt=start_opt,
                convergence=convergence, iters_cap=iters, seed=seed, scale_views=scale_views, center_groups=center_groups, use_float32=use_float32,
                freqELBO=freqELBO, ard_factors_requested=ard_factors, ard_factors_effective=bool(ent.model_opts['ard_factors']),
                spikeslab_weights=spikeslab_weights, ard_weights=ard_weights, mefisto=use_gp, sparseGP=sparseGP,
                n_spots=int(a.n_obs), n_genes=int(a.n_vars), build_seconds=round(t1 - t0, 1), train_seconds=round(t2 - t1, 1))
    json.dump(meta, open(out.replace('.hdf5', '.json'), 'w'), indent=1)
    return meta

def load(path):
    """Read a saved mofapy2 model: W (genes x K), Z (spots x K), R2 per factor (%), total R2 (%), smoothness, lengthscale, ELBO."""
    f = h5py.File(path, 'r'); dec = lambda x: [v.decode() if isinstance(v, bytes) else v for v in x]
    g, v = dec(f['groups/groups'][:])[0], dec(f['views/views'][:])[0]
    W = pd.DataFrame(f[f'expectations/W/{v}'][:].T, index=dec(f[f'features/{v}'][:])); K = W.shape[1]; W.columns = [f'F{k+1}' for k in range(K)]
    Z = pd.DataFrame(f[f'expectations/Z/{g}'][:].T, index=dec(f[f'samples/{g}'][:]), columns=W.columns)
    out = dict(W=W, Z=Z, K=K, r2=pd.Series(f[f'variance_explained/r2_per_factor/{g}'][0], index=W.columns),
               r2_total=float(f[f'variance_explained/r2_total/{g}'][0]), elbo=f['training_stats/elbo'][:], cov=f[f'cov_samples/{g}'][:] if 'cov_samples' in f else None)
    if 'training_stats/scales' in f:
        out['smoothness'] = pd.Series(f['training_stats/scales'][:], index=W.columns); out['lengthscale'] = pd.Series(f['training_stats/length_scales'][:], index=W.columns)
    out['n_iter'] = int(len(f['training_stats/time'][:]) - 1)   # iterations run (time[0] is the init)
    if os.path.abspath(path) == os.path.abspath(OFFICIAL_MODEL):     # figshare model: names are feature<i>_view0 / sample<i>_group0, so recover from official_input.h5ad- covariates are (imagecol, imagerow) here
        import anndata as ad; oi = ad.read_h5ad(OFFICIAL, backed='r'); out['W'].index = oi.var_names; out['Z'].index = oi.obs_names; oi.file.close()
        out['cov'] = out['cov'][:, ::-1]
    fin = out['elbo'][np.isfinite(out['elbo'])]; out['elbo_final'] = float(fin[-1]) if len(fin) else float('nan')
    if out['cov'] is None and 'samples_metadata' in f and 'imagerow' in f[f'samples_metadata/{g}']:   # plain-MOFA fits: coordinates from the saved metadata
        out['cov'] = np.c_[f[f'samples_metadata/{g}/imagerow'][:], f[f'samples_metadata/{g}/imagecol'][:]].astype(float)
    f.close(); return out

def active(m, min_r2=0.01):
    """analysis.active_factors rule: factors with R2 >= 0.01 percent."""
    return [k for k in m['W'].columns if m['r2'][k] >= min_r2]

def summarise(path, official=None):
    """Replication-table row for a saved model. `official` = load(OFFICIAL_MODEL) with recovered names (for factor matching)."""
    m = load(path); act = active(m)
    top = {k: list(m['W'][k].abs().sort_values(ascending=False).index[:5]) for k in act}
    ttr = [k for k in act if 'Ttr' in top[k][:3]]
    row = dict(model=os.path.basename(path), n_factors=m['K'], n_active=len(act), r2_total=round(m['r2_total'], 2),
               r2_per_factor=[round(float(m['r2'][k]), 2) for k in act], n_iter=m['n_iter'], elbo_final=round(m['elbo_final'], 1) if np.isfinite(m['elbo_final']) else None)
    if 'smoothness' in m:
        row['smoothness'] = [round(float(m['smoothness'][k]), 2) for k in act]; row['lengthscale'] = [round(float(m['lengthscale'][k]), 1) for k in act]
    row['ttr_factor'] = ttr; row['top_genes'] = {k: top[k][:4] for k in act}
    if official is not None:
        common = m['Z'].index.intersection(official['Z'].index)
        if len(common) > 100:
            C = pd.DataFrame({ko: [abs(np.corrcoef(m['Z'].loc[common, k], official['Z'].loc[common, ko])[0, 1]) if m['Z'][k].std() > 0 else 0 for k in act]
                              for ko in official['W'].columns}, index=act)
            row['match_official'] = {ko: (C[ko].idxmax(), round(float(C[ko].max()), 2)) for ko in official['W'].columns}
            row['n_common_spots'] = int(len(common))
    return row, m
