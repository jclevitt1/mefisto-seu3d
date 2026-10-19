"""Compare the common-frame MEFISTO refit vs the first pass.
Writes factor_summary_aligned.csv (one row per aligned factor, w the matched first-pass factor and its numbers) and
celltype_means_aligned.csv; prints a markdown table for NOTES.md."""
import numpy as np, pandas as pd, h5py
from analysis import load_model, factor_summary, active_factors, celltype_agreement
A = load_model('models/mefisto_e8_2500per_K12_aligned.hdf5'); B = load_model('models/mefisto_e8_2500per_K12.hdf5')
with h5py.File('models/mefisto_e8_2500per_K12_aligned.hdf5') as f:
    print('covariates stored:', [c.decode() if isinstance(c, bytes) else c for c in f['covariates_names'][:]] if 'covariates_names' in f else 'n/a')
sa, sb = factor_summary(A), factor_summary(B)
# match aligned factors to first-pass factors by cell-by-cell correlation of Z (same cells, same order checked)
cells = pd.concat([A['Z'][g] for g in A['groups']]).index; cells_b = pd.concat([B['Z'][g] for g in B['groups']]).index
assert list(cells) == list(cells_b), 'cell order differs between models'
Za = pd.concat([A['Z'][g] for g in A['groups']]); Zb = pd.concat([B['Z'][g] for g in B['groups']])
act_a, act_b = active_factors(A), active_factors(B)
C = pd.DataFrame(np.corrcoef(Za[act_a].T.to_numpy(), Zb[act_b].T.to_numpy())[:len(act_a), len(act_a):], index=act_a, columns=act_b)
print('\ncorrelation matrix (aligned rows x first-pass cols):\n', C.round(2).to_string())
rows = []
for k in sa.index:
    r = {'factor_aligned': k, 'active_aligned': k in act_a}
    for c in ['r2_rep1', 'r2_rep2', 'sharedness', 'celltype_agreement', 'smoothness', 'lengthscale', 'corr_AP_rep1', 'corr_AP_rep2', 'corr_z_rep1', 'corr_z_rep2', 'top_pos', 'top_neg']:
        r[c] = sa.loc[k, c] if c in sa.columns else np.nan
    if k in act_a:
        j = C.loc[k].abs().idxmax(); r['match_firstpass'] = j; r['match_corr'] = C.loc[k, j]
        for c in ['r2_rep1', 'r2_rep2', 'sharedness', 'celltype_agreement', 'smoothness', 'lengthscale']: r[f'fp_{c}'] = sb.loc[j, c]
    rows.append(r)
out = pd.DataFrame(rows).set_index('factor_aligned'); out.to_csv('factor_summary_aligned.csv'); print('\nwrote factor_summary_aligned.csv')
# per-cell-type means for the aligned factors (same layout as celltype_means.csv)
cm = pd.concat({g: A['Z'][g][act_a].groupby(A['obs'][g].celltype).mean().round(2) for g in A['groups']}, axis=1); cm.to_csv('celltype_means_aligned.csv')
print('\n| aligned factor | matched first-pass factor (corr) | var. expl. rep1 / rep2 (%): first pass -> aligned | GP sharedness: first pass -> aligned | cell-type agreement: first pass -> aligned | smoothness: first pass -> aligned | length scale (um): first pass -> aligned |')
print('|---|---|---|---|---|---|---|')
for k, r in out[out.active_aligned].iterrows():
    print(f"| {k} | {r.match_firstpass} ({r.match_corr:+.2f}) | {r.fp_r2_rep1:.2f} / {r.fp_r2_rep2:.2f} -> {r.r2_rep1:.2f} / {r.r2_rep2:.2f} | {r.fp_sharedness:.2f} -> {r.sharedness:.2f} | {r.fp_celltype_agreement:.2f} -> {r.celltype_agreement:.2f} | {r.fp_smoothness:.2f} -> {r.smoothness:.2f} | {r.fp_lengthscale:.0f} -> {r.lengthscale:.0f} |")
print('\nfirst-pass factors not matched as best match by any aligned factor:', sorted(set(act_b) - set(out.match_firstpass.dropna())))
print('total r2:', A['r2_total'], 'vs first pass', B['r2_total'])
print('\nmean factor value per cell type (aligned, active factors):\n', cm.to_string())
