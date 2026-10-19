"""Runs one named fit. Usage: run_fit.py <name>. Names/settings are in SPECS below; output is models/<name>.hdf5 + .json, log in logs/<name>.log."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c

OFFICIAL = dict(K=4, frac_inducing=1000 / 2696, n_grid=20, opt_freq=10, start_opt=20, convergence='fast', iters=1000, seed=2021,
                scale_views=False, center_groups=True, use_float32=False, freqELBO=1, ard_factors=False)   # R tutorial + mofapy2 defaults; figshare model's saved opts
OURS = dict(K=12, frac_inducing=0.1, n_grid=10, opt_freq=25, start_opt=25, convergence='medium', iters=300, seed=0,
            scale_views=False, center_groups=True, use_float32=True, freqELBO=5, ard_factors=True)         # fit.py lines 10-16, OPTFREQ=25
SPECS = {
    # replicate-official
    'official_exact':   ('official', OFFICIAL),                                   # paper's matrix + paper's settings: does my mofapy2 reproduce the figshare model?
    'official_mine':    ('seurat',   OFFICIAL),                                   # my preprocessing of the 10x counts + paper's settings
    # replicate-ours
    'ours':             ('ours',     OURS),                                       # prep.py + fit.py on the 10x counts
    'ours_on_official': ('official', OURS),                                       # paper's matrix + fit.py settings (isolates fit settings from preprocessing)
    # ablations, one flip from 'ours' toward 'official'
    'abl_scale_views':  ('ours',     {**OURS, 'scale_views': True}),
    'abl_inducing1000': ('ours',     {**OURS, 'frac_inducing': 1000 / 2695}),
    'abl_fullgp':       ('ours',     {**OURS, 'sparseGP': False}),
    'abl_K4':           ('ours',     {**OURS, 'K': 4}),
    'abl_ngrid20':      ('ours',     {**OURS, 'n_grid': 20}),
    'abl_optfreq10':    ('ours',     {**OURS, 'opt_freq': 10, 'start_opt': 20}),
    'abl_conv_fast':    ('ours',     {**OURS, 'convergence': 'fast', 'freqELBO': 1}),
    'abl_conv_slow':    ('ours',     {**OURS, 'convergence': 'slow', 'iters': 450}),
    'abl_float64':      ('ours',     {**OURS, 'use_float32': False}),
    'abl_hvg_vst':      ('ours_vst', OURS),                                       # HVG: Seurat vst on counts instead of dispersion on log data
    'abl_hvg_min10':    ('ours_hvgall', OURS),                                    # genes in >= 10 spots before HVG selection
    'abl_seed2021':     ('ours',     {**OURS, 'seed': 2021}),
    'abl_ardF_false':   ('ours',     {**OURS, 'ard_factors': False}),             # should be identical- mofapy2 forces it off under MEFISTO
    'abl_all_official': ('ours',     OFFICIAL),                                   # our preprocessing, every fit setting official
    # depth test: the paper's tissue thinned to Stereo-seq-bin depth, then our pipeline / the official pipeline
    'ours_thin1200':    ('ours_thin1200',   OURS),
    'official_thin1200':('seurat_thin1200', OFFICIAL),
    'mofa_thin1200':    ('ours_thin1200',   {**OURS, 'use_gp': False}),
    # plain MOFA
    'mofa_ours':        ('ours',     {**OURS, 'use_gp': False}),
    'mofa_official':    ('official', {**OFFICIAL, 'use_gp': False}),
    'mofa_ours_K4':     ('ours',     {**OURS, 'use_gp': False, 'K': 4}),
}
name = sys.argv[1]; variant, opts = SPECS[name]
out = os.path.join(c.M, name + '.hdf5')
a, cov = c.preprocess(variant)
print(name, variant, a.shape, 'zeros', round(float((a.X == 0).mean()) if not hasattr(a.X, 'nnz') else 1 - a.X.nnz / a.X.size, 4), opts, flush=True)
meta = c.fit(a, cov, out, **opts); meta.update(name=name, variant=variant)
json.dump(meta, open(out.replace('.hdf5', '.json'), 'w'), indent=1)
print('DONE', name, meta, flush=True)
