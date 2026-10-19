#!/usr/bin/env python3
"""Builds stuck_here.ipynb: the "where I am stuck" report, with the evidence for each hypothesis directly under it.
Loads the two saved presented fits and the saved tables; imports the repo's own modules. Run from the repo root:
python build_stuck_here.py && jupyter nbconvert --to notebook --execute --inplace stuck_here.ipynb"""
import nbformat as nbf

OUT = 'stuck_here.ipynb'
nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md("""
# MEFISTO on the SEU-3D E8.0 embryos: where I am stuck

MEFISTO and plain MOFA (mofapy2 0.7.5) fitted to the two E8.0 mouse embryos of the SEU-3D digital embryo (Xie et al., Cell 2025).
One matrix entry is the log-normalised SCT count of one gene in one Stereo-seq cell bin; one row is one bin, median about 1,200
counts over 16,000 genes. 2,500 bins per embryo, 2,000 variable genes, 12 factors, the embryos as groups, (x, y, z) as covariates.
MEFISTO is MOFA plus a Gaussian-process prior over the coordinates on each factor; everything else is identical.

**Settled:** the low per-entry variance explained is a depth effect, not the factors (report in my last email;
`variance_explained.ipynb` in this repo has the full argument).

**Stuck:** why MEFISTO performs worse than plain MOFA on data this spatial. The tables below are the evidence; the fits
themselves come from `fit.py` and are loaded here.
""")
code('''
import numpy as np, pandas as pd
from IPython.display import Image, display
from analysis import load_model, active_factors, pseudobulk_r2
JOINT = 'data/e8_joint_2500per.h5ad'
M = {'MEFISTO': load_model('models/mefisto_e8_2500per_K12.hdf5'), 'MOFA': load_model('models/mofa_e8_2500per_K12.hdf5')}
clf = pd.read_csv('classifier_summary.csv', index_col=[0, 1])
pb = pd.concat({k: pseudobulk_r2(m, JOINT) for k, m in M.items()}, names=['model']).reset_index(level=1, drop=True)
H = pd.read_csv('heldout_summary.csv')
''')

md("""
## Stuck here

The spatial model loses on every per-cell score. R2 is mofapy2's per-embryo variance explained; "cell-type means" scores the same
reconstruction after averaging the bins of each published cell type (types with at least 25 bins); germ-layer accuracy is a
cross-validated logistic regression from the factor scores to the three germ layers (balanced accuracy, chance 0.33).
""")
code('''
rows = {}
for k, m in M.items():
    p = pb.loc[k].set_index('embryo')
    rows[k] = {'R2 rep1 (%)': m['r2_total']['E8.0_rep1'], 'R2 rep2 (%)': m['r2_total']['E8.0_rep2'],
               'R2 on cell-type means rep1 (%)': p.loc['E8.0_rep1', 'r2_pseudobulk'], 'R2 on cell-type means rep2 (%)': p.loc['E8.0_rep2', 'r2_pseudobulk'],
               'germ-layer accuracy': clf.loc[(k, 'germ layer'), 'balanced accuracy'], 'active factors': len(active_factors(m))}
stuck = pd.DataFrame(rows).T.round(2)
stuck.loc['MEFISTO / MOFA'] = (stuck.loc['MEFISTO'] / stuck.loc['MOFA']).round(2)
stuck
''')
code("display(Image('figures/fig4_mefisto_vs_mofa_space.png'))")
md("""
Figure 4: each active MEFISTO factor next to its best-matched MOFA factor, both embryos. The MEFISTO-to-MOFA R2 ratio is 0.40
here; on the MEFISTO paper's own Visium data it is 0.86 at full depth and 0.72 after thinning to 1,200 counts per spot
(section "Ruled out"), so depth explains part of the gap and not the rest. I do not have the mechanism. Four hypotheses follow,
each with what has been measured.
""")

md("""
## A. Sampling noise

A MOFA factor has an independent standard-normal prior per cell, so it can absorb per-bin sampling noise; a smooth factor cannot.
If that were the whole gap, MOFA's lead would vanish on hidden entries and on cell-type means, where the per-bin noise is gone.

Held-out test A (`heldout.py`, 1,000 bins per embryo, three seeds): 10% of the entries of every bin are hidden and the reconstruction
is scored on them.
""")
code('''
H[H.test == 'A'].pivot_table(index='model', columns='embryo', values=['r', 'mse']).round(3)
''')
code('''
pb[['embryo', 'r2_cell', 'r2_pseudobulk', 'between_type_share']].round(2)
''')
md("""
**Partly true, not the whole story.** On hidden entries MOFA's 2.5x in-sample lead shrinks to 0.02 in correlation, so most of the
per-entry gap is noise-fitting. But the gap survives averaging within cell type (78 against 55%), where per-bin noise is divided
away, so what MEFISTO drops is structured.
""")

md("""
## B. The smooth prior discards real, non-smooth biology

MEFISTO's factor prior is a Gaussian process with a stationary squared-exponential kernel and one lengthscale per factor; variation
that is not smooth at that lengthscale is left unexplained by construction. Cell identity at E8.0 is patchy: sheets a few
cells thick, sharp boundaries between them, and scattered populations. If this is the mechanism, the labels that are sheets and boundaries should
be the ones MEFISTO's factors lose.
""")
code('''
clf.round(3)
''')
code("display(Image('figures/fig5_all_factors_vs_atlas.png'))")
md("""
**Consistent with everything measured, not tested directly.** Germ layer, the most spatial label there is, drops from 0.86 to 0.71;
cell type from 0.46 to 0.37. The test I would run: centre expression by cell type and refit both models on the residuals, where
only within-tissue gradients remain. If MEFISTO still loses there, the prior is misspecified for this tissue even where it is
smooth; if it wins, the GP's value is confined to within-type spatial structure and the loss on the full data is the boundaries.
""")

md("""
## C. Fitting path: five factors died

MEFISTO's prior per factor is scale x spatial kernel + (1 - scale) x identity, with the scale and lengthscale learned. At scale 0 it
is MOFA's prior. So a factor the GP cannot describe could in principle survive as a MOFA-like factor. Did it?
""")
code('''
m = M['MEFISTO']
fac = pd.DataFrame({'R2 rep1 (%)': m['r2'].iloc[:, 0], 'R2 rep2 (%)': m['r2'].iloc[:, 1], 'GP scale': m['smoothness'],
                    'lengthscale (um)': m['lengthscale'], 'sharedness': m['sharedness']}).round(2)
fac['active'] = fac.index.isin(active_factors(m))
fac
''')
code("display(Image('figures/fig1_sharedness.png'))")
md("""
**Tested in part.** The five inactive factors are exactly those whose GP scale went to about 0 (0.05, 0.10, 0.01, 0.00, 0.11 against
0.49 to 0.80 for the seven active). The model did fall back to an iid-like prior for them, and they then died instead of surviving
as MOFA-like factors. Factor-level ARD is not the reason: mofapy2 forces `ard_factors = False` whenever covariates are given, and a
refit with it explicitly off is bit-identical to this one, so the pruning comes from the weight-side ARD and spike-and-slab prior
on W. Why those factors die under the GP framework and not under MOFA is open.
""")

md("""
## D. Resolution and geometry

The sparse GP is summarised at 10% inducing points. If the inducing grid is coarser than a germ layer, the smooth part of the model
cannot resolve the sheets whatever the lengthscale.
""")
code('''
geo = {}
for g, o in m['obs'].items():
    ext = (o[['x', 'y', 'z']].max() - o[['x', 'y', 'z']].min())
    n_ind = round(0.1 * len(o))
    geo[g] = {'extent x (um)': ext.x, 'extent y (um)': ext.y, 'extent z (um)': ext.z, 'bins': len(o), 'inducing points': n_ind,
              'inducing spacing (um)': (ext.prod() / n_ind) ** (1 / 3), 'sections (distinct z)': o.z.nunique()}
geo = pd.DataFrame(geo).T.round(0)
ls = m['lengthscale'][active_factors(m)]
print(f'learned lengthscales of the active factors: {ls.min():.0f} to {ls.max():.0f} um')
geo
''')
md("""
**Not tested.** Inducing spacing of 90 to 120 um against germ layers a few cells thick, and learned lengthscales of 42 to 288 um, and the two embryos are not in a common
coordinate frame (an anterior-posterior aligned refit is in `extra/`). The denser-inducing-point refit with a tighter lengthscale
bound is the one experiment that speaks to this; on Visium it was killed at 40 minutes and never finished. If germ-layer accuracy
recovers with denser inducing points the gap was resolution (D); if not, it is the smoothness assumption itself (B).
""")

md("""
## Ruled out

Fit settings, gene selection, and depth alone. Every setting in `fit.py` was flipped one at a time toward the MEFISTO paper's on the
paper's own Visium example; all moved R2 by less than seed-to-seed noise except the number of factors and the inducing fraction
(`sanity/tables/ablation.md`). The paper's gene selection raises R2 on the embryos to 4.5 / 7.2% (MEFISTO) against 9.2 / 13.2%
(MOFA), same ratio. Depth: the Visium counts thinned binomially to the embryos' 1,200 per spot, same tissue, same genes, same code.
""")
code('''
S = pd.read_csv('sanity/tables/all_models.csv').set_index('name')
S.loc[['ours', 'ours_thin1200', 'mofa_ours', 'mofa_thin1200'], ['K', 'n_active', 'r2_total']].rename(
    columns={'n_active': 'active', 'r2_total': 'R2 (%)'}).rename(index={'ours': 'MEFISTO, Visium full depth', 'ours_thin1200': 'MEFISTO, Visium thinned to 1,200',
    'mofa_ours': 'MOFA, Visium full depth', 'mofa_thin1200': 'MOFA, Visium thinned to 1,200'})
''')
md("""
MEFISTO-to-MOFA ratio 0.86 at full depth, 0.72 thinned, 0.40 on the embryos.
""")

md("""
## To do

Work out what the ELBO optimises against what R2 measures. Both models share the likelihood and the priors on weights and noise;
the only term that differs is the KL between the factor posterior and the factor prior (independent normal for MOFA, Gaussian
process for MEFISTO), so if B is right the mechanism has to be in that term. The sparse GP makes MEFISTO's bound looser, so the
final ELBOs below are not a controlled comparison.
""")
code('''
pd.Series({k: m['elbo'][~np.isnan(m['elbo'])][-1] for k, m in M.items()}, name='final ELBO').round(0)
''')
md("""
Then the two refits: B's residual refit and D's denser inducing points.

## Questions

1. Is there a known MEFISTO setting for stacked sections, and does the lab have a view on the inducing-point density needed for
   sheet-like structure?
2. The one thing only MEFISTO can do is predict bins it never saw from their coordinates (held-out test B in
   `variance_explained.ipynb`): r = 0.13 / 0.16 against 0.08 / 0.11 for the mean of the 10 nearest observed bins. Is held-out
   prediction at new positions the right test of what spatial information adds, and is a gain this size meaningful at this depth?
3. For the AD data, is R2 on cell-type (or spatial-bin) averages plus held-out prediction an acceptable pair of headline metrics?

## Reproduce

Everything here loads saved fits and tables. The fits: `fit.py` (the two presented models), `heldout.py` (tests A and B),
`compare.py` (classifier, figures 4 to 6), `sanity/` (the Visium runs); commands in the README. `data/` and `models/` are not in
the repo.
""")

nb['cells'] = cells
nbf.write(nb, OUT)
print('wrote', OUT, len(cells), 'cells')
