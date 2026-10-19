"""MEFISTO vs plain MOFA on the saved models: Figure 4 (side by side in space), the cross-validated classifier and Leiden
comparison (Figure 5), and the held-out prediction table and Figure 6 from the results heldout.py wrote. No model is fitted here.
Usage as a script: python compare.py [mefisto.hdf5 mofa.hdf5]  (writes figures/fig4..6 and classifier_summary.csv, leiden_summary.csv)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from analysis import BLUE, ORANGE, DIVERGING, GERM, ANTERIOR, POSTERIOR, active_factors, factor_summary, ap_coords

MIN_CELLS, N_NEIGH, LEIDEN_RES, SEED = 25, 15, 1.0, 0     # classifier / Leiden settings
HO_N_PER, HO_KNN = 1000, 10                               # held-out experiment size (heldout_funcs.py) and its spatial kNN baseline

# ---------------------------------------------------------------- Figure 4: MEFISTO and MOFA side by side in space
def mofa_match(m, m0):
    """Best plain-MOFA factor per active MEFISTO factor by cell-by-cell correlation of Z (same cells, same order), the signed
    correlation, the sign that makes it positive, and the MOFA factor's variance explained and cell-type agreement."""
    act, act0 = active_factors(m), active_factors(m0); S0 = factor_summary(m0)
    Z1 = pd.concat([m['Z'][g] for g in m['groups']])[act]; Z0 = pd.concat([m0['Z'][g] for g in m0['groups']]).loc[Z1.index, act0]
    C = pd.DataFrame(np.corrcoef(Z1.T, Z0.T)[:len(act), len(act):], index=act, columns=act0)
    best = C.abs().idxmax(axis=1)
    out = pd.DataFrame({'mofa': best, 'corr': [C.loc[k, best[k]] for k in act], 'sign': [np.sign(C.loc[k, best[k]]) for k in act]}, index=act)
    out['MOFA_r2_rep1'] = S0.loc[out.mofa, 'r2_rep1'].to_numpy(); out['MOFA_r2_rep2'] = S0.loc[out.mofa, 'r2_rep2'].to_numpy()
    out['MOFA_celltype_agreement'] = S0.loc[out.mofa, 'celltype_agreement'].to_numpy()
    return out

def reference_row(axes_row, m, apc, legend=True):
    """Figure 2's top row: germ layers in x-y and anterior/posterior types in x-z, per embryo, into four axes."""
    for j, g in enumerate(m['groups']):
        o = m['obs'][g]; c = apc[g]; short = g.replace('E8.0_', '')
        ax = axes_row[2 * j]
        for gl, col in GERM.items():
            s = o.germ_layer == gl; ax.scatter(c.lr[s], c.ap[s], s=1.2, c=col, lw=0, label=gl, rasterized=True)
        ax.set_title(f'{short}: germ layer, x-y (n={len(o)})', fontsize=8); ax.set_aspect('equal'); ax.set_xlabel('left-right (um)'); ax.set_ylabel('posterior -> anterior (um)')
        if j == 0 and legend: ax.legend(frameon=False, markerscale=6, fontsize=7, loc='lower left')
        ax = axes_row[2 * j + 1]
        other = ~o.celltype.isin(ANTERIOR + POSTERIOR)
        ax.scatter(o.z[other], c.ap[other], s=1, c='#d8d7d3', lw=0, rasterized=True)
        s = o.celltype.isin(ANTERIOR); ax.scatter(o.z[s], c.ap[s], s=1.5, c='#4a3aa7', lw=0, label='anterior types', rasterized=True)
        s = o.celltype.isin(POSTERIOR); ax.scatter(o.z[s], c.ap[s], s=1.5, c='#eda100', lw=0, label='posterior types', rasterized=True)
        ax.set_title(f'{short}: anterior / posterior types, x-z', fontsize=8); ax.set_aspect('equal'); ax.set_xlabel('section position z (um)')
        if j == 0 and legend: ax.legend(frameon=False, markerscale=6, fontsize=7, loc='lower left')

def factor_row(axes_row, fig, mm, fk, apc, label, sign=1.0, xlabels=False):
    """One factor of model mm painted into four axes (rep1 x-y, rep1 x-z, rep2 x-y, rep2 x-z) with a numeric colour bar."""
    G = mm['groups']
    lim = max(np.percentile(np.abs(np.concatenate([mm['Z'][g][fk].to_numpy() for g in G])), 99), 1e-6)
    for j, g in enumerate(G):
        c = apc[g]; z = sign * mm['Z'][g][fk].to_numpy(); o = mm['obs'][g]; short = g.replace('E8.0_', '')
        order = np.argsort(np.abs(z))
        for v, (xx, xl) in enumerate([(c.lr.to_numpy(), 'left-right (um)'), (o.z.to_numpy(), 'section position z (um)')]):
            ax = axes_row[2 * j + v]
            sc = ax.scatter(xx[order], c.ap.to_numpy()[order], c=z[order], s=1.2, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), lw=0, rasterized=True)
            ax.set_aspect('equal'); ax.set_title(f'{label} {short} {"x-y" if v == 0 else "x-z"}  R2 {mm["r2"].loc[fk, g]:.2f}%', fontsize=8)
            if xlabels: ax.set_xlabel(xl)
    axes_row[0].set_ylabel('posterior -> anterior (um)')
    cb = fig.colorbar(sc, ax=axes_row[3], fraction=0.05, pad=0.02, shrink=0.8); cb.ax.tick_params(labelsize=6)

def fig_space_vs_mofa(m, m0, path, title):
    """Figure 4: Figure 2's layout twice, side by side. Left half is MEFISTO, right half the best-matched plain-MOFA factor
    (sign flipped if the correlation is negative, marked with a minus). Row 0 is the atlas reference on both halves."""
    act = active_factors(m); nrow = 1 + len(act); apc = {g: ap_coords(m['obs'][g]) for g in m['groups']}
    mt = mofa_match(m, m0)
    fig, axes = plt.subplots(nrow, 9, figsize=(19.8, 2.45 * nrow), dpi=150, constrained_layout=True, width_ratios=[1, 1, 1, 1, 0.3, 1, 1, 1, 1])
    for ax in axes[:, 4]: ax.axis('off')                      # spacer column between the halves
    axes = np.concatenate([axes[:, :4], axes[:, 5:]], axis=1)
    reference_row(axes[0, :4], m, apc); reference_row(axes[0, 4:], m, apc, legend=False)
    for i, fk in enumerate(act):
        r = 1 + i; last = r == nrow - 1
        factor_row(axes[r, :4], fig, m, fk, apc, f'MEFISTO {fk}', xlabels=last)
        f0, s0 = mt.loc[fk, 'mofa'], mt.loc[fk, 'sign']
        factor_row(axes[r, 4:], fig, m0, f0, apc, f'MOFA {"-" if s0 < 0 else ""}{f0}', sign=s0, xlabels=last)
    for ax in axes.ravel():
        if not ax.has_data(): ax.axis('off')
    fig.canvas.draw()                                          # fix the layout, then centre a heading over each half
    for cols, text in [((0, 3), 'MEFISTO (spatial prior)'), ((4, 7), 'plain MOFA, best-matched factor (minus = sign flipped)')]:
        x = (axes[0, cols[0]].get_position().x0 + axes[0, cols[1]].get_position().x1) / 2
        fig.text(x, 1.0, text, ha='center', va='bottom', fontsize=11, color='#0b0b0b')
    fig.suptitle(title, x=0.01, y=1.025, ha='left', fontsize=10, color='#0b0b0b'); fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return mt

# ---------------------------------------------------------------- all factors together against the atlas labels
def standardised_scores(mm, index):
    """Active-factor scores of all cells (rows ordered as index), each column standardised so the strong factors don't dominate."""
    act = active_factors(mm); Z = pd.concat([mm['Z'][g] for g in mm['groups']])[act].loc[index]
    return (Z - Z.mean()) / Z.std()

def cv_predict(X, y):
    """Out-of-fold predictions of a multinomial logistic regression (5-fold stratified) and of the majority-class baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(LogisticRegression(max_iter=5000), X, y, cv=cv)
    base = cross_val_predict(DummyClassifier(strategy='most_frequent'), X, y, cv=cv)
    return pd.Series(pred, index=y.index), pd.Series(base, index=y.index)

def leiden_factors(X):
    """Leiden (RB configuration) on an unweighted N_NEIGH-nearest-neighbour graph in factor space."""
    from sklearn.neighbors import NearestNeighbors
    import igraph as ig, leidenalg as la
    idx = NearestNeighbors(n_neighbors=N_NEIGH + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    edges = {(min(i, j), max(i, j)) for i in range(len(X)) for j in idx[i]}
    part = la.find_partition(ig.Graph(n=len(X), edges=sorted(edges)), la.RBConfigurationVertexPartition, resolution_parameter=LEIDEN_RES, seed=SEED)
    return pd.Series(part.membership, index=X.index)

def classify_and_cluster(models):
    """For each model in {name: loaded model}: cross-validated logistic regression from the standardised active-factor scores to
    germ layer and to cell type (types with >= MIN_CELLS cells pooled over embryos, the rest 'other'), balanced accuracy and macro F1
    vs the majority-class baseline; and Leiden clusters scored by ARI and NMI against the full atlas labels.
    Returns dict(classifier=table, leiden=table, preds={(model, target): Series}, clusters={model: Series}, obs=obs_all,
    targets={target: Series}, big_types=list, type_counts=Series)."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score
    m = next(iter(models.values()))
    obs_all = pd.concat([m['obs'][g] for g in m['groups']])
    type_counts = obs_all.celltype.value_counts(); big_types = list(type_counts.index[type_counts >= MIN_CELLS])
    ct_pooled = obs_all.celltype.where(obs_all.celltype.isin(big_types), 'other')
    targets = {'germ layer': obs_all.germ_layer, 'cell type': ct_pooled}
    preds, clusters, rows_c, rows_l = {}, {}, [], []
    for name, mm in models.items():
        X = standardised_scores(mm, obs_all.index)
        for tname, y in targets.items():
            pred, base = cv_predict(X, y); preds[(name, tname)] = pred
            rows_c.append({'model': name, 'target': tname, 'factors': X.shape[1], 'classes': y.nunique(),
                           'balanced accuracy': balanced_accuracy_score(y, pred), 'macro F1': f1_score(y, pred, average='macro'),
                           'baseline balanced accuracy': balanced_accuracy_score(y, base), 'baseline macro F1': f1_score(y, base, average='macro')})
        lab = leiden_factors(X); clusters[name] = lab
        rows_l.append({'model': name, 'Leiden clusters': lab.nunique(),
                       'ARI vs germ layer': adjusted_rand_score(obs_all.germ_layer, lab), 'NMI vs germ layer': normalized_mutual_info_score(obs_all.germ_layer, lab),
                       'ARI vs cell type': adjusted_rand_score(obs_all.celltype, lab), 'NMI vs cell type': normalized_mutual_info_score(obs_all.celltype, lab)})
    return dict(classifier=pd.DataFrame(rows_c).set_index(['model', 'target']), leiden=pd.DataFrame(rows_l).set_index('model'),
                preds=preds, clusters=clusters, obs=obs_all, targets=targets, big_types=big_types, type_counts=type_counts)

def type_palette(type_counts):
    """One distinct colour per atlas cell type, in order of frequency."""
    pal = [plt.cm.tab20(i) for i in range(20)] + [plt.cm.tab20b(i) for i in range(0, 20, 2)]
    return {t: pal[i % len(pal)] for i, t in enumerate(type_counts.index)}

def majority_colour(lab, ref, palette):
    """Colour each cluster by the atlas label it mostly contains; a second cluster with the same majority label
    gets a paler shade of that colour (mixed with white), a third paler still."""
    ct = pd.crosstab(lab, ref); seen, out = {}, {}
    for c in ct.index[np.argsort(-ct.max(axis=1).to_numpy())]:   # the biggest cluster per label keeps the full colour
        lab_ = ct.loc[c].idxmax(); n = seen.get(lab_, 0); seen[lab_] = n + 1
        rgb = np.array(to_rgb(palette[lab_])); out[c] = tuple(rgb + (1 - rgb) * min(0.45 * n, 0.8))
    return out

def fig_all_factors(models, res, path, title):
    """Figure 5. Row 0: atlas germ layer and cell type in x-y, both embryos. One row per model: germ-layer confusion matrix
    (cross-validated), per-cell-type recall, and the Leiden clusters in x-y for both embryos painted by majority atlas cell type.
    res is the output of classify_and_cluster(models)."""
    from sklearn.metrics import recall_score, confusion_matrix
    m = next(iter(models.values())); G = m['groups']; apc = {g: ap_coords(m['obs'][g]) for g in G}; nrow = 1 + len(models); germ = list(GERM)
    CLF, LEI, preds, leiden, obs = res['classifier'], res['leiden'], res['preds'], res['clusters'], res['obs']
    big_types, type_counts, targets = res['big_types'], res['type_counts'], res['targets']; ct_pooled = targets['cell type']
    TYPE_COL = type_palette(type_counts)
    fig = plt.figure(figsize=(12.5, 3.1 * nrow), dpi=150, constrained_layout=True)
    outer = fig.add_gridspec(nrow, 1); row0 = outer[0].subgridspec(1, 4); scat = []
    def xy(ax, g, colours, ttl):
        c = apc[g]; ax.scatter(c.lr, c.ap, s=1.2, c=colours, lw=0, rasterized=True); ax.set_aspect('equal'); ax.set_title(ttl, fontsize=7.5); scat.append(ax)
    for j, g in enumerate(G):
        o = m['obs'][g]; short = g.replace('E8.0_', '')
        ax = fig.add_subplot(row0[0, 2 * j]); xy(ax, g, [GERM[x] for x in o.germ_layer], f'{short}: atlas germ layer (n={len(o)})')
        if j == 0: ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', ms=5, color=c, label=l) for l, c in GERM.items()], frameon=False, fontsize=7, loc='lower left')
        ax = fig.add_subplot(row0[0, 2 * j + 1]); xy(ax, g, [TYPE_COL[t] for t in o.celltype], f'{short}: atlas cell type ({o.celltype.nunique()} types)')
    for i, name in enumerate(models):
        gs = outer[1 + i].subgridspec(1, 4, width_ratios=[0.95, 1.5, 1.2, 1.2])
        y, p = obs.germ_layer, preds[(name, 'germ layer')]; cm = confusion_matrix(y, p, labels=germ); frac = cm / cm.sum(axis=1, keepdims=True)
        ax = fig.add_subplot(gs[0, 0]); ax.imshow(frac, cmap='Blues', vmin=0, vmax=1)
        for a in range(3):
            for b in range(3): ax.text(b, a, f'{frac[a, b]:.2f}\n({cm[a, b]})', ha='center', va='center', fontsize=7, color='white' if frac[a, b] > 0.55 else '#0b0b0b')
        ax.set_xticks(range(3)); ax.set_xticklabels(germ, fontsize=7, rotation=30, ha='right'); ax.set_yticks(range(3)); ax.set_yticklabels(germ, fontsize=7)
        ax.set_xlabel('predicted', fontsize=8); ax.set_ylabel('atlas', fontsize=8); ax.spines[:].set_visible(False)
        r_ = CLF.loc[(name, 'germ layer')]
        ax.set_title(f'{name}: germ layer, cross-validated\nbalanced accuracy {r_["balanced accuracy"]:.2f}, macro F1 {r_["macro F1"]:.2f}', fontsize=8, loc='left')
        y, p = ct_pooled, preds[(name, 'cell type')]; labs = big_types + ['other']
        rec = pd.Series(recall_score(y, p, labels=labs, average=None, zero_division=0), index=labs); r_ = CLF.loc[(name, 'cell type')]
        ax = fig.add_subplot(gs[0, 1]); yy = np.arange(len(labs))
        ax.barh(yy, rec, color=[TYPE_COL.get(t, '#b0afab') for t in labs], height=0.7)
        ax.axvline(r_['balanced accuracy'], color='#52514e', ls='--', lw=0.8)
        ax.set_yticks(yy); ax.set_yticklabels([f'{t} ({type_counts.get(t, (ct_pooled == "other").sum())})' for t in labs], fontsize=6); ax.invert_yaxis()
        ax.set_xlim(0, 1); ax.set_xlabel('recall (cross-validated)', fontsize=8)
        ax.set_title(f'{name}: cell type recall, cross-validated\nbalanced accuracy {r_["balanced accuracy"]:.2f} (dashed), macro F1 {r_["macro F1"]:.2f}', fontsize=8, loc='left')
        lab = leiden[name]; cols = majority_colour(lab, obs.celltype, TYPE_COL); L = LEI.loc[name]
        for j, g in enumerate(G):
            o = m['obs'][g]; short = g.replace('E8.0_', ''); l = lab.loc[o.index]
            ttl = f'{name}: Leiden, {int(L["Leiden clusters"])} clusters, {short}'
            if j == 0: ttl += f'\nARI {L["ARI vs germ layer"]:.2f}, NMI {L["NMI vs germ layer"]:.2f} vs germ layer\nARI {L["ARI vs cell type"]:.2f}, NMI {L["NMI vs cell type"]:.2f} vs cell type'
            xy(fig.add_subplot(gs[0, 2 + j]), g, [cols[x] for x in l], ttl)
    for ax in scat:
        sp = ax.get_subplotspec()
        if sp.colspan.start in (0, 2): ax.set_ylabel('posterior -> anterior (um)', fontsize=8)
        if sp.get_topmost_subplotspec().rowspan.start == nrow - 1: ax.set_xlabel('left-right (um)', fontsize=8)
    handles = [plt.Line2D([], [], marker='o', ls='', ms=4, color=TYPE_COL[t], label=f'{t} ({type_counts[t]})') for t in type_counts.index]
    fig.legend(handles=handles, loc='outside lower center', ncol=5, fontsize=6, frameon=False, title='atlas cell types (cells, both embryos); each Leiden cluster takes the colour of its majority label, paler if that label is already taken', title_fontsize=7)
    fig.suptitle(title, x=0.01, ha='left', fontsize=10, color='#0b0b0b'); fig.savefig(path); plt.close(fig)

# ---------------------------------------------------------------- held-out validation (results written by heldout.py)
def heldout_table(results):
    """Mean over seeds (and the spread) of r and MSE per test, model, embryo, from heldout.py's results CSV (path or DataFrame);
    MOFA's test-B row stays empty since MOFA cannot place an unseen cell."""
    R = pd.read_csv(results) if isinstance(results, str) else results
    Rn = R.dropna(subset=['r'])
    tab = Rn.groupby(['test', 'model', 'embryo']).agg(r=('r', 'mean'), r_sd=('r', 'std'), mse=('mse', 'mean'), mse_sd=('mse', 'std'), seeds=('seed', 'nunique')).reset_index()
    order = {'MEFISTO': 0, 'MOFA': 1, f'kNN in space (k={HO_KNN})': 2, 'gene mean': 3}
    tab['o'] = tab.model.map(order).fillna(9); tab = tab.sort_values(['test', 'o', 'embryo']).drop(columns='o').set_index(['test', 'model', 'embryo'])
    for g in sorted(R.embryo.unique()):
        tab.loc[('B', 'MOFA', g), :] = [np.nan] * 4 + [0]
    return tab.sort_index(level=0, sort_remaining=False)

def fig_heldout(R, preds, path, seed=0):
    """Figure 6: predicted vs true hidden values (density, log colour), one row per embryo; the test-A panels compare MEFISTO and MOFA
    on the same masked entries, the test-B panels compare the MEFISTO GP prediction at unseen cells with the spatial kNN baseline.
    R is heldout.py's results table, preds the dict in models/heldout_predictions.npz."""
    groups = sorted(R.embryo.unique()); cols = [('A', 'MEFISTO', 'Test A, masked entries\nMEFISTO'), ('A', 'MOFA', 'Test A, masked entries\nplain MOFA'),
                                                ('B', 'MEFISTO', 'Test B, held-out cells\nMEFISTO (GP at new coordinates)'), ('B', 'kNN', f'Test B, held-out cells\nmean of {HO_KNN} nearest cells in space')]
    cmaps = {groups[0]: LinearSegmentedColormap.from_list('b', ['#fcfcfb', '#9dc4ef', BLUE, '#0d366b']), groups[1]: LinearSegmentedColormap.from_list('o', ['#fcfcfb', '#f7c3ad', ORANGE, '#7a2c0e'])}
    model_key = {'kNN': f'kNN in space (k={HO_KNN})'}
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.2), dpi=150, sharex=True, sharey=True)
    for i, g in enumerate(groups):
        for j, (t, mdl, title) in enumerate(cols):
            ax = axes[i, j]; true, pred = preds[f'{t}_{mdl}_{seed}_{g}']
            lim = (-0.4, 6.8)
            ax.plot(lim, lim, color='#b0afab', lw=0.8, zorder=1)
            ax.hexbin(true, pred, gridsize=70, bins='log', cmap=cmaps[g], mincnt=1, extent=(*lim, *lim), linewidths=0.1, zorder=2)
            sub = R[(R.test == t) & (R.model == model_key.get(mdl, mdl)) & (R.embryo == g)]
            ax.text(0.03, 0.97, f'r = {sub.r.mean():.3f}  (seeds {sub.r.min():.3f} to {sub.r.max():.3f})\nMSE = {sub.mse.mean():.4f}', transform=ax.transAxes, va='top', fontsize=8, color='#0b0b0b')
            if i == 0: ax.set_title(title, loc='left', fontsize=9)
            if j == 0: ax.set_ylabel(f'{g}\npredicted (log expression)')
            if i == 1: ax.set_xlabel('true hidden value (log expression)')
            ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal'); ax.grid(False)
    fig.suptitle(f'Figure 6. Held-out prediction: true hidden values against their predictions ({HO_N_PER:,} cells per embryo, 10% hidden, seed {seed} shown; '
                 f'colour = log density of entries).\nScores are means over {R.seed.nunique()} seeds. Test A: MEFISTO and plain MOFA on the same masked entries.\n'
                 f'Test B: MEFISTO predicts unseen cells from their coordinates; plain MOFA cannot.', x=0.01, ha='left', fontsize=9.5, color='#0b0b0b')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

if __name__ == '__main__':
    import sys, os
    from analysis import load_model
    mef = sys.argv[1] if len(sys.argv) > 1 else 'models/mefisto_e8_2500per_K12.hdf5'
    mof = sys.argv[2] if len(sys.argv) > 2 else 'models/mofa_e8_2500per_K12.hdf5'
    m, m0 = load_model(mef), load_model(mof)
    mt = fig_space_vs_mofa(m, m0, 'figures/fig4_mefisto_vs_mofa_space.png', 'Figure 4. Each active MEFISTO factor (left) next to its best-matched plain-MOFA factor (right), both embryos, x-y and x-z views; A-P vertical')
    print(mt.round(2).to_string())
    models = {'MEFISTO': m, 'MOFA': m0}; res = classify_and_cluster(models)
    res['classifier'].to_csv('classifier_summary.csv'); res['leiden'].to_csv('leiden_summary.csv')
    print(res['classifier'].round(3).to_string()); print(res['leiden'].round(3).to_string())
    fig_all_factors(models, res, 'figures/fig5_all_factors_vs_atlas.png',
                    'Figure 5. All active factors together: can the factor scores predict the atlas labels? Cross-validated logistic regression, and Leiden clusters in x-y, A-P vertical')
    if os.path.exists('heldout_results.csv'):
        R = pd.read_csv('heldout_results.csv'); print(heldout_table(R).round(4).to_string())
        if os.path.exists('models/heldout_predictions.npz'):
            fig_heldout(R, dict(np.load('models/heldout_predictions.npz')), 'figures/fig6_heldout.png', seed=0)
