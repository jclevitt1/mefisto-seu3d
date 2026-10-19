"""Build the joint E8.0 AnnData: harmonise labels, SCT counts, normalise+log1p, 2000 HVGs, then subsample."""
import anndata as ad, scanpy as sc, numpy as np, pandas as pd, scipy.sparse as sp, sys
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SEED = 0
r1 = ad.read_h5ad('data/GSM9046247_Embryo_E8.0_stereo_rep1.h5ad')
r2 = ad.read_h5ad('data/GSM9046248_Embryo_E8.0_stereo_rep2.h5ad')
a1 = ad.AnnData(X=sp.csr_matrix(r1.layers['counts']), obs=r1.obs[['x','y','z','chip','germ_layer','celltype','nCount_RNA','nFeature_RNA']].copy(), var=pd.DataFrame(index=r1.var_names))
a2 = ad.AnnData(X=sp.csr_matrix(r2.raw.X), obs=r2.obs[['x','y','z','chip','germ_layer','celltype','nCount_RNA','nFeature_RNA']].copy(), var=pd.DataFrame(index=r2.var_names))
a1.obs['embryo'] = 'E8.0_rep1'; a2.obs['embryo'] = 'E8.0_rep2'
genes = a1.var_names.intersection(a2.var_names)
a = ad.concat([a1[:, genes], a2[:, genes]], index_unique='-')
a.obs['celltype'] = a.obs['celltype'].astype(str).replace({'pre-somitic': 'Pre-somitic'})
a.obs['embryo'] = a.obs['embryo'].astype('category')
rng = np.random.default_rng(SEED)
keep = np.concatenate([rng.choice(np.where(a.obs.embryo == e)[0], size=min(N_PER, (a.obs.embryo == e).sum()), replace=False) for e in a.obs.embryo.cat.categories])
a = a[np.sort(keep)].copy()
a.layers['sct_counts'] = a.X.copy()
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=2000, batch_key='embryo', flavor='seurat')
a = a[:, a.var.highly_variable].copy()
a.obsm['spatial3d'] = a.obs[['x','y','z']].to_numpy(dtype=float)
a.write_h5ad(f'data/e8_joint_{N_PER}per.h5ad')
print(a, '\nper embryo', a.obs.embryo.value_counts().to_dict())
