"""Load a mofapy2/MEFISTO model and make the three figures plus summary tables."""
import h5py, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

BLUE, ORANGE = '#2a78d6', '#eb6834'          # embryo 1, embryo 2
DIVERGING = LinearSegmentedColormap.from_list('div', ['#0d366b', '#2a78d6', '#f0efec', '#e34948', '#7a1f1f'])
GERM = {'ectoderm': '#2a78d6', 'mesoderm': '#eb6834', 'endoderm': '#1baf7a'}
ANTERIOR = ['Rostral neurectoderm', 'Forebrain/Midbrain/Hindbrain', 'Cardiomyocytes', 'Pharyngeal mesoderm']
POSTERIOR = ['Caudal epiblast', 'Caudal Mesoderm', 'Allantois', 'Pre-somitic', 'NMP', 'Caudal neurectoderm']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#b0afab',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.titlesize': 10,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})

def dec(x): return [v.decode() if isinstance(v, bytes) else v for v in x]

def load_model(path):
    f = h5py.File(path, 'r')
    groups, views = dec(f['groups/groups'][:]), dec(f['views/views'][:])
    v = views[0]
    W = pd.DataFrame(f[f'expectations/W/{v}'][:].T, index=dec(f[f'features/{v}'][:]))
    K = W.shape[1]; W.columns = [f'F{k+1}' for k in range(K)]
    Z, obs = {}, {}
    for g in groups:
        Z[g] = pd.DataFrame(f[f'expectations/Z/{g}'][:].T, index=dec(f[f'samples/{g}'][:]), columns=W.columns)
        md = f[f'samples_metadata/{g}']
        obs[g] = pd.DataFrame({c: dec(md[c][:]) for c in md.keys()}, index=Z[g].index)
        for c in ['x', 'y', 'z']: obs[g][c] = obs[g][c].astype(float)
    r2 = pd.DataFrame({g: f[f'variance_explained/r2_per_factor/{g}'][0] for g in groups}, index=W.columns)
    r2tot = {g: float(f[f'variance_explained/r2_total/{g}'][0]) for g in groups}
    out = dict(groups=groups, W=W, Z=Z, obs=obs, r2=r2, r2_total=r2tot, K=K, elbo=f['training_stats/elbo'][:])
    if 'training_stats/Kg' in f:
        Kg = f['training_stats/Kg'][:]
        out['Kg'] = Kg
        out['sharedness'] = pd.Series([Kg[k, 0, 1] / np.sqrt(Kg[k, 0, 0] * Kg[k, 1, 1]) for k in range(K)], index=W.columns)
        out['smoothness'] = pd.Series(f['training_stats/scales'][:], index=W.columns)
        out['lengthscale'] = pd.Series(f['training_stats/length_scales'][:], index=W.columns)
    f.close()
    return out

def ap_coords(o):
    """Rotate in-plane (x, y) so the anterior-posterior axis (posterior to anterior cell-type centroids) is vertical."""
    ca = o.loc[o.celltype.isin(ANTERIOR), ['x', 'y']].mean().to_numpy(); cp = o.loc[o.celltype.isin(POSTERIOR), ['x', 'y']].mean().to_numpy()
    u = (ca - cp); u /= np.linalg.norm(u); v = np.array([-u[1], u[0]])
    xy = o[['x', 'y']].to_numpy() - (ca + cp) / 2
    return pd.DataFrame({'ap': xy @ u, 'lr': xy @ v, 'z': o.z - o.z.mean()}, index=o.index)

def eta2(values, labels):
    df = pd.DataFrame({'v': values, 'l': labels}); m = df.v.mean()
    ss_b = sum(len(g) * (g.v.mean() - m) ** 2 for _, g in df.groupby('l')); ss_t = ((df.v - m) ** 2).sum()
    return ss_b / ss_t

def active_factors(m, min_r2=0.01):
    """Factors explaining at least min_r2 percent of variance in at least one embryo."""
    return [k for k in m['W'].columns if (m['r2'].loc[k] >= min_r2).any()]

def factor_summary(m):
    rows = []
    for k in m['W'].columns:
        r = {'factor': k, 'r2_rep1': m['r2'].loc[k, m['groups'][0]], 'r2_rep2': m['r2'].loc[k, m['groups'][1]]}
        if 'sharedness' in m: r.update(sharedness=m['sharedness'][k], smoothness=m['smoothness'][k], lengthscale=m['lengthscale'][k])
        for i, g in enumerate(m['groups']):
            o, z = m['obs'][g], m['Z'][g][k]; apc = ap_coords(o)
            if z.std() < 1e-8: continue
            r[f'corr_AP_rep{i+1}'] = np.corrcoef(z, apc.ap)[0, 1]
            r[f'corr_z_rep{i+1}'] = np.corrcoef(z, o.z)[0, 1]
            r[f'eta2_germ_rep{i+1}'] = eta2(z.to_numpy(), o.germ_layer.to_numpy())
            r[f'eta2_celltype_rep{i+1}'] = eta2(z.to_numpy(), o.celltype.to_numpy())
            r[f'corr_depth_rep{i+1}'] = np.corrcoef(z, np.log(o.nCount_RNA.astype(float)))[0, 1]
        w = m['W'][k]; r['top_pos'] = ', '.join(w.sort_values(ascending=False).index[:6]); r['top_neg'] = ', '.join(w.sort_values().index[:6])
        r['celltype_agreement'] = celltype_agreement(m, k)
        rows.append(r)
    return pd.DataFrame(rows).set_index('factor')

def pseudobulk_r2(m, h5ad, min_cells=25):
    """Variance explained on cell-type means instead of single cells, per embryo.
    Y = the fitted matrix centred per gene within the embryo (as mofapy2 does), Yhat = Z W^T. For each cell type with
    >= min_cells cells in that embryo, average Y and Yhat over its cells; R2 = 1 - SS(mean Y - mean Yhat) / SS(mean Y),
    summed over types and genes, each type counted once (unweighted). Also returns the cell-level R2 recomputed the same
    way (sanity check- should equal mofapy2's r2_total) and the share of cell-level variance that lies between cell types."""
    import anndata as ad
    a = ad.read_h5ad(h5ad); rows = []
    for g in m['groups']:
        b = a[a.obs.embryo == g]; X = b.X.toarray() if hasattr(b.X, 'toarray') else np.asarray(b.X)
        Y = X - X.mean(0)
        Yhat = m['Z'][g].loc[b.obs_names].to_numpy() @ m['W'].loc[b.var_names].to_numpy().T
        counts = b.obs.celltype.value_counts(); types = counts[counts >= min_cells].index
        mY = np.array([Y[(b.obs.celltype == t).to_numpy()].mean(0) for t in types])
        mYhat = np.array([Yhat[(b.obs.celltype == t).to_numpy()].mean(0) for t in types])
        n = counts.loc[types].to_numpy()
        rows.append(dict(embryo=g, n_types=len(types), cells_in_types=int(n.sum()),
                         r2_cell=100 * (1 - ((Y - Yhat) ** 2).sum() / (Y ** 2).sum()),
                         r2_pseudobulk=100 * (1 - ((mY - mYhat) ** 2).sum() / (mY ** 2).sum()),
                         between_type_share=100 * (n @ (mY ** 2).sum(1)) / (Y ** 2).sum()))
    return pd.DataFrame(rows).round(2)

def celltype_agreement(m, k, min_cells=25):
    """Coordinate-free sharedness check: correlation between the two embryos of the mean factor value per published
    cell type (types with >= min_cells cells in both embryos). Unlike the GP group correlation, this doesn't depend on
    the two embryos sharing a coordinate frame."""
    g1, g2 = m['groups'][:2]
    if m['Z'][g1][k].std() < 1e-8 or m['Z'][g2][k].std() < 1e-8: return np.nan
    m1 = m['Z'][g1][k].groupby(m['obs'][g1].celltype).agg(['mean', 'size']); m2 = m['Z'][g2][k].groupby(m['obs'][g2].celltype).agg(['mean', 'size'])
    common = [c for c in m1.index.intersection(m2.index) if m1.loc[c, 'size'] >= min_cells and m2.loc[c, 'size'] >= min_cells]
    return float(np.corrcoef(m1.loc[common, 'mean'], m2.loc[common, 'mean'])[0, 1]) if len(common) >= 4 else np.nan

def fig_sharedness(m, path, title, mofa=None):
    S = factor_summary(m); K = m['K']; y = np.arange(K); h = 0.38
    on = S.index.isin(active_factors(m)); ya, yo = y[on], y[~on]  # panels b and c: GP sharedness / smoothness only for active factors
    INACTIVE = dict(marker='o', ls='none', color='#d8d7d3', ms=4, mew=0, clip_on=False, label='inactive factor (R2 = 0)')
    n = 4 if mofa is not None else 3
    fig, axes = plt.subplots(1, n, figsize=(9.6, 4.6), dpi=150, gridspec_kw={'width_ratios': [1.3, 1, 1, 1][:n]})
    ax = axes[0]
    ax.barh(y - h / 2, S.r2_rep1, h, color=BLUE, label=m['groups'][0]); ax.barh(y + h / 2, S.r2_rep2, h, color=ORANGE, label=m['groups'][1])
    ax.set_yticks(y); ax.set_yticklabels(S.index); ax.invert_yaxis(); ax.set_xlabel('variance explained (%)'); ax.set_title('a. MEFISTO: variance explained\nper factor and embryo', loc='left', fontsize=9); ax.legend(frameon=False, loc='lower right')
    ax = axes[1]
    if 'sharedness' in S:
        ax.hlines(ya, 0, S.sharedness[on], color='#b0afab', lw=1); ax.plot(S.sharedness[on], ya, 'o', color='#0b0b0b', ms=6, label='GP group')
        ax.plot(S.celltype_agreement[on], ya, 'D', color='#8a8985', mfc='none', ms=5, label='cell-type mean')
        if len(yo): ax.plot(np.zeros(len(yo)), yo, **INACTIVE)
        ax.set_xlim(-1.4, 1.05); ax.axvline(0, color='#b0afab', lw=0.8); ax.legend(frameon=False, fontsize=5.5, loc='lower left', handletextpad=0.2, borderaxespad=0.1)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_xlabel('sharedness\n(1 = same in both)'); ax.set_title('b. Sharedness across\nthe two embryos', loc='left', fontsize=9)
    ax = axes[2]
    if 'smoothness' in S:
        ax.hlines(ya, 0, S.smoothness[on], color='#b0afab', lw=1); ax.plot(S.smoothness[on], ya, 'o', color='#0b0b0b', ms=6); ax.set_xlim(0, 1.05)
        if len(yo): ax.plot(np.zeros(len(yo)), yo, **INACTIVE); ax.legend(frameon=False, fontsize=5.5, loc='lower right', handletextpad=0.2, borderaxespad=0.1)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_xlabel('smoothness\n(smooth share of variance)'); ax.set_title('c. How spatially smooth\neach factor is', loc='left', fontsize=9)
    if mofa is not None:
        S0 = factor_summary(mofa); ax = axes[3]
        ax.barh(y - h / 2, S0.r2_rep1, h, color=BLUE); ax.barh(y + h / 2, S0.r2_rep2, h, color=ORANGE)
        ax.set_yticks(y); ax.set_yticklabels(S0.index, fontsize=7); ax.invert_yaxis(); ax.set_xlabel('variance explained (%)'); ax.set_title('d. Plain MOFA, no spatial\nprior: variance explained', loc='left', fontsize=9)
        ax.set_xlim(0, max(S.r2_rep1.max(), S.r2_rep2.max(), S0.r2_rep1.max(), S0.r2_rep2.max()) * 1.1); axes[0].set_xlim(ax.get_xlim())
    fig.suptitle(title, x=0.01, ha='left', fontsize=10, color='#0b0b0b'); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return S

def fig_space(m, path, title):
    """Per active factor, one row of four panels: rep1 in-plane (x-y, A-P vertical), rep1 across sections (z vs A-P),
    rep2 in-plane, rep2 across sections. Row 0 is the reference- germ layers and anterior/posterior types."""
    act = active_factors(m); K = len(act); G = m['groups']; ncol = 4; nrow = 1 + K
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.6, 2.45 * nrow), dpi=150, constrained_layout=True)
    apc = {g: ap_coords(m['obs'][g]) for g in G}
    for j, g in enumerate(G):  # reference row
        o = m['obs'][g]; c = apc[g]; short = g.replace('E8.0_', '')
        ax = axes[0, 2 * j]
        for gl, col in GERM.items():
            s = o.germ_layer == gl; ax.scatter(c.lr[s], c.ap[s], s=1.2, c=col, lw=0, label=gl, rasterized=True)
        ax.set_title(f'{short}: germ layer, x-y (n={len(o)})', fontsize=8); ax.set_aspect('equal'); ax.set_xlabel('left-right (um)'); ax.set_ylabel('posterior -> anterior (um)')
        if j == 0: ax.legend(frameon=False, markerscale=6, fontsize=7, loc='lower left')
        ax = axes[0, 2 * j + 1]
        other = ~o.celltype.isin(ANTERIOR + POSTERIOR)
        ax.scatter(o.z[other], c.ap[other], s=1, c='#d8d7d3', lw=0, rasterized=True)
        s = o.celltype.isin(ANTERIOR); ax.scatter(o.z[s], c.ap[s], s=1.5, c='#4a3aa7', lw=0, label='anterior types', rasterized=True)
        s = o.celltype.isin(POSTERIOR); ax.scatter(o.z[s], c.ap[s], s=1.5, c='#eda100', lw=0, label='posterior types', rasterized=True)
        ax.set_title(f'{short}: anterior / posterior types, x-z', fontsize=8); ax.set_aspect('equal'); ax.set_xlabel('section position z (um)')
        if j == 0: ax.legend(frameon=False, markerscale=6, fontsize=7, loc='lower left')
    for i, fk in enumerate(act):
        k = int(fk[1:]) - 1; r = 1 + i
        lim = max(np.percentile(np.abs(np.concatenate([m['Z'][g].iloc[:, k].to_numpy() for g in G])), 99), 1e-6)
        for j, g in enumerate(G):
            c = apc[g]; z = m['Z'][g].iloc[:, k].to_numpy(); o = m['obs'][g]; short = g.replace('E8.0_', '')
            order = np.argsort(np.abs(z))
            for v, (xx, xl) in enumerate([(c.lr.to_numpy(), 'left-right (um)'), (o.z.to_numpy(), 'section position z (um)')]):
                ax = axes[r, 2 * j + v]
                sc = ax.scatter(xx[order], c.ap.to_numpy()[order], c=z[order], s=1.2, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), lw=0, rasterized=True)
                ax.set_aspect('equal'); ax.set_title(f'{fk} {short} {"x-y" if v == 0 else "x-z"}  R2 {m["r2"].loc[fk, g]:.2f}%', fontsize=8)
                if r == nrow - 1: ax.set_xlabel(xl)
            if j == 0: axes[r, 0].set_ylabel('posterior -> anterior (um)')
        cb = fig.colorbar(sc, ax=axes[r, 3], fraction=0.05, pad=0.02, shrink=0.8); cb.ax.tick_params(labelsize=6)
    for ax in axes.ravel():
        if not ax.has_data(): ax.axis('off')
    fig.suptitle(title, x=0.01, ha='left', fontsize=10, color='#0b0b0b'); fig.savefig(path); plt.close(fig)

def fig_genes_celltypes(m, path, title, min_cells=25):
    G = m['groups']; act = active_factors(m); K = len(act)
    Zall = pd.concat([m['Z'][g] for g in G])[act]; oall = pd.concat([m['obs'][g] for g in G])
    Zs = (Zall - Zall.mean()) / Zall.std()
    ct = oall.celltype.value_counts(); keep = ct[ct >= min_cells].index
    germ_of = oall.groupby('celltype').germ_layer.agg(lambda s: s.value_counts().index[0])
    M = Zs[oall.celltype.isin(keep)].groupby(oall.celltype[oall.celltype.isin(keep)]).mean()
    M = M.loc[sorted(M.index, key=lambda c: (['ectoderm', 'mesoderm', 'endoderm'].index(germ_of[c]) if germ_of[c] in GERM else 9, c))]
    fig = plt.figure(figsize=(9.6, max(0.3 * len(M) + 2.4, 0.42 * K + 2.4)), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0]); lim = np.nanmax(np.abs(M.to_numpy()))
    im = ax.imshow(M.to_numpy(), cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
    ax.set_xticks(range(K)); ax.set_xticklabels(M.columns, fontsize=8); ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f'{c}  ({ct[c]})' for c in M.index]); ax.tick_params(length=0)
    for i, c in enumerate(M.index): ax.get_yticklabels()[i].set_color(GERM.get(germ_of[c], '#0b0b0b'))
    ax.set_title('a. Mean factor value per published cell type\n(z-scored across all cells; label colour = germ layer;\nn cells in brackets)', loc='left', fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.5); cb.set_label('mean standardised\nfactor value', fontsize=7); cb.ax.tick_params(labelsize=7)
    ax2 = fig.add_subplot(gs[1]); ax2.axis('off'); ax2.set_title('b. Top genes per factor\n(+: highest weights, -: lowest weights)\n', loc='left', fontsize=9)
    lines = []
    for k in act:
        w = m['W'][k]; lines.append(f'{k}  +: ' + ', '.join(w.sort_values(ascending=False).index[:7])); lines.append('      -: ' + ', '.join(w.sort_values().index[:7]))
    ax2.text(0, 1, '\n'.join(lines), va='top', ha='left', fontsize=7.2, family='monospace', transform=ax2.transAxes, color='#0b0b0b', linespacing=1.35)
    fig.suptitle(title, x=0.01, ha='left', fontsize=11, color='#0b0b0b'); fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return M

def known_answer_check(S, act):
    """Known-answer rules on the summary table, active factors only. AP_factor: |corr_AP| > 0.4 in both embryos with the same
    sign; germ_factor: germ layer explains > 25% of the factor's variance in both; section_artefact: |corr_z| > 0.5 in either;
    shared_GP: sharedness > 0.5; shared_celltypes: cell-type agreement > 0.8."""
    chk = S.loc[act, ['r2_rep1', 'r2_rep2', 'sharedness', 'celltype_agreement', 'smoothness', 'corr_AP_rep1', 'corr_AP_rep2',
                      'eta2_germ_rep1', 'eta2_germ_rep2', 'corr_z_rep1', 'corr_z_rep2', 'corr_depth_rep1', 'corr_depth_rep2']].round(2)
    chk['AP_factor'] = (chk[['corr_AP_rep1', 'corr_AP_rep2']].abs().min(axis=1) > 0.4) & (np.sign(chk.corr_AP_rep1) == np.sign(chk.corr_AP_rep2))
    chk['germ_factor'] = chk[['eta2_germ_rep1', 'eta2_germ_rep2']].min(axis=1) > 0.25
    chk['section_artefact'] = chk[['corr_z_rep1', 'corr_z_rep2']].abs().max(axis=1) > 0.5
    chk['shared_GP'] = chk.sharedness > 0.5
    chk['shared_celltypes'] = chk.celltype_agreement > 0.8
    return chk

def aligned_comparison(m, mA):
    """Each active first-pass factor of m vs its best-matched (cell-by-cell correlation of Z) active factor of the
    common-frame refit mA: sharedness, cell-type agreement, smoothness, AP correlation and variance explained, before and after."""
    S1, SA = factor_summary(m), factor_summary(mA); act1, actA = active_factors(m), active_factors(mA)
    ZA = pd.concat([mA['Z'][g] for g in mA['groups']]); Z1 = pd.concat([m['Z'][g] for g in m['groups']]).loc[ZA.index]
    Cm = pd.DataFrame(np.corrcoef(Z1[act1].T, ZA[actA].T)[:len(act1), len(act1):], index=act1, columns=actA)
    rows = []
    for k in act1:
        j = Cm.abs().loc[k].idxmax()
        rows.append({'first_pass': k, 'aligned': j, 'corr': Cm.loc[k, j],
                     'sharedness': S1.loc[k, 'sharedness'], 'sharedness_aligned': SA.loc[j, 'sharedness'],
                     'celltype_agreement': S1.loc[k, 'celltype_agreement'], 'celltype_agreement_aligned': SA.loc[j, 'celltype_agreement'],
                     'smoothness': S1.loc[k, 'smoothness'], 'smoothness_aligned': SA.loc[j, 'smoothness'],
                     'corr_AP_rep1_aligned': SA.loc[j, 'corr_AP_rep1'], 'corr_AP_rep2_aligned': SA.loc[j, 'corr_AP_rep2'],
                     'r2_rep1': S1.loc[k, 'r2_rep1'], 'r2_rep1_aligned': SA.loc[j, 'r2_rep1'],
                     'r2_rep2': S1.loc[k, 'r2_rep2'], 'r2_rep2_aligned': SA.loc[j, 'r2_rep2']})
    return pd.DataFrame(rows).set_index('first_pass')

if __name__ == '__main__':
    import sys
    m = load_model(sys.argv[1]); tag = sys.argv[2]
    mofa = load_model(sys.argv[3]) if len(sys.argv) > 3 else None
    pre = f'figures/{tag}_' if tag else 'figures/'
    S = fig_sharedness(m, f'{pre}fig1_sharedness.png', 'Figure 1. MEFISTO factors on two E8.0 SEU-3D embryos: variance explained, sharedness, smoothness; plain MOFA as comparison', mofa=mofa)
    fig_space(m, f'{pre}fig2_factors_in_space.png', 'Figure 2. Each factor painted onto both embryos, x-y (in-plane, sections overlaid) and x-z (across sections); A-P vertical')
    M = fig_genes_celltypes(m, f'{pre}fig3_genes_celltypes.png', 'Figure 3. Factors against the published cell types, and their top genes')
    S.to_csv(f'{tag + "_" if tag else ""}factor_summary.csv'); M.to_csv(f'{tag + "_" if tag else ""}celltype_means.csv')
    print(S.round(2).to_string())
