"""Common-frame MEFISTO refit- same as fit.py / the first pass
(K=12, 300 iters, sparse GP, 10% inducing, n_grid 10, start_opt=opt_freq=25, seed 0), only the covariates are the
aligned (ap, lr, z) built by build_aligned.py instead of the published (x, y, z)."""
import sys, time, json, anndata as ad
from mofapy2.run.entry_point import entry_point
inp, out = 'data/e8_joint_2500per_aligned.h5ad', 'models/mefisto_e8_2500per_K12_aligned.hdf5'
K, iters, frac, ngrid, optfreq = 12, 300, 0.1, 10, 25
a = ad.read_h5ad(inp); a.obs['sample'] = a.obs_names
t0 = time.time()
ent = entry_point()
ent.set_data_options(scale_views=False, center_groups=True, use_float32=True)
ent.set_data_from_anndata(a, groups_label='embryo', use_raw=False, save_metadata=True)
ent.set_model_options(factors=K, spikeslab_weights=True, ard_weights=True, ard_factors=True)
ent.set_train_options(iter=iters, convergence_mode='medium', startELBO=1, freqELBO=5, seed=0, verbose=False, quiet=False, gpu_mode=False)
ent.set_covariates(['ap', 'lr', 'z'], covariates_names=['ap', 'lr', 'z'])
ent.set_smooth_options(scale_cov=False, model_groups=True, sparseGP=True, frac_inducing=frac, n_grid=ngrid, start_opt=optfreq, opt_freq=optfreq)
ent.build()
t1 = time.time(); ent.run(); t2 = time.time()
ent.save(out, save_data=True)
meta = dict(input=inp, output=out, factors=K, iters_requested=iters, frac_inducing=frac, n_grid=ngrid, opt_freq=optfreq, mefisto=True, covariates=['ap', 'lr', 'z'],
            n_cells=int(a.n_obs), n_genes=int(a.n_vars), cells_per_group=a.obs.embryo.value_counts().to_dict(),
            build_seconds=round(t1 - t0, 1), train_seconds=round(t2 - t1, 1))
json.dump(meta, open(out.replace('.hdf5', '.json'), 'w'), indent=1); print(meta)
