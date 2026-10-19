"""Fit MEFISTO (mofapy2)- embryos as groups, (x,y,z) as covariates. Usage: fit.py in.h5ad out.hdf5 K iters frac_inducing [nogp]"""
import sys, time, json, numpy as np, anndata as ad
from mofapy2.run.entry_point import entry_point
inp, out, K, iters, frac = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
use_gp = not (len(sys.argv) > 6 and sys.argv[6] == 'nogp')
a = ad.read_h5ad(inp)
a.obs['sample'] = a.obs_names
t0 = time.time()
ent = entry_point()
ent.set_data_options(scale_views=False, center_groups=True, use_float32=True)
ent.set_data_from_anndata(a, groups_label='embryo', use_raw=False, save_metadata=True)
ent.set_model_options(factors=K, spikeslab_weights=True, ard_weights=True, ard_factors=False)
ent.set_train_options(iter=iters, convergence_mode='medium', startELBO=1, freqELBO=5, seed=0, verbose=False, quiet=False, gpu_mode=False)
if use_gp:
    ent.set_covariates(['x', 'y', 'z'], covariates_names=['x', 'y', 'z'])
    ent.set_smooth_options(scale_cov=False, model_groups=True, sparseGP=True, frac_inducing=frac, n_grid=int(__import__("os").environ.get("NGRID", 10)), start_opt=int(__import__("os").environ.get("OPTFREQ", 20)), opt_freq=int(__import__("os").environ.get("OPTFREQ", 20)))
ent.build()
t1 = time.time(); ent.run(); t2 = time.time()
ent.save(out, save_data=True)
meta = dict(input=inp, output=out, factors=K, iters_requested=iters, frac_inducing=frac if use_gp else None, mefisto=use_gp,
            n_cells=int(a.n_obs), n_genes=int(a.n_vars), cells_per_group=a.obs.embryo.value_counts().to_dict(),
            build_seconds=round(t1 - t0, 1), train_seconds=round(t2 - t1, 1))
json.dump(meta, open(out.replace('.hdf5', '.json'), 'w'), indent=1); print(meta)
