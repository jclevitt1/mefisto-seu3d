"""Builds the replication and ablation tables from every saved model in models/ plus the official figshare model. Writes
tables/replication.csv, tables/ablation.csv and prints both as markdown."""
import os, sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c
T = os.path.join(c.ROOT, 'tables'); os.makedirs(T, exist_ok=True)
# official model w recovered names- rebuild W/Z index from official_input.h5ad
off = c.load(c.OFFICIAL_MODEL)
import anndata as ad
oi = ad.read_h5ad(c.OFFICIAL, backed='r'); off['W'].index = oi.var_names; off['Z'].index = oi.obs_names
rows = []
r, m = c.summarise(c.OFFICIAL_MODEL, None)
r['W'] = None; r['name'] = 'paper (ST_model.hdf5, figshare)'; r['variant'] = 'official'
top = {k: list(off['W'][k].abs().sort_values(ascending=False).index[:4]) for k in off['W'].columns}
r['top_genes'] = top; r['ttr_factor'] = [k for k in top if 'Ttr' in top[k][:3]]; rows.append(r)
for p in sorted(glob.glob(os.path.join(c.M, '*.hdf5'))):
    j = json.load(open(p.replace('.hdf5', '.json'))) if os.path.exists(p.replace('.hdf5', '.json')) else {}
    r, m = c.summarise(p, off); r['name'] = j.get('name', os.path.basename(p)); r['variant'] = j.get('variant')
    r['minutes'] = round(j.get('train_seconds', np.nan) / 60, 1); r['n_inducing'] = j.get('n_inducing'); r['K'] = j.get('K'); rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(T, 'all_models.csv'), index=False)
def fmt(v): return ', '.join(f'{x:.2f}' for x in v) if isinstance(v, list) and len(v) and isinstance(v[0], float) else (', '.join(map(str, v)) if isinstance(v, list) else v)
cols = ['name', 'n_factors', 'n_active', 'r2_total', 'r2_per_factor', 'smoothness', 'lengthscale', 'ttr_factor', 'n_iter', 'minutes', 'n_inducing', 'match_official']
out = df[[c_ for c_ in cols if c_ in df]].copy()
for c_ in ['r2_per_factor', 'smoothness', 'lengthscale', 'ttr_factor']: out[c_] = out[c_].map(fmt)
out['match_official'] = out['match_official'].map(lambda d: ', '.join(f'{ko}:{v[0]} ({v[1]:.2f})' for ko, v in d.items()) if isinstance(d, dict) else '')
def md(df):   # markdown table without tabulate
    cols = list(df.columns); lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows(): lines.append('| ' + ' | '.join('' if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in r) + ' |')
    return '\n'.join(lines)
print(md(out))
out.to_csv(os.path.join(T, 'summary.csv'), index=False)
print('\ntop genes per active factor:')
for r in rows: print(' ', r['name'], {k: v[:4] for k, v in r['top_genes'].items()})
