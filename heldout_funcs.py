# Held-out validation of MEFISTO vs plain MOFA (the notebook loads the results).
# heldout.py runs this from the shell- compare.heldout_table and compare.fig_heldout read what it writes.
HO_N_PER = 1000              # cells per embryo here (random subsample, seed 0, of the 2,500-per joint data)
HO_SEEDS = [0, 1, 2]         # masking seeds
HO_HIDE = 0.10               # fraction hidden: entries (test A) or whole cells (test B)
HO_KNN = 10                  # neighbours for the spatial k-nearest-neighbour baseline (test B)
HO_RESULTS = 'heldout_results.csv'            # one row per test / seed / model / embryo
HO_TIMES = 'heldout_times.csv'                # wall time per fit
HO_PRED = 'models/heldout_predictions.npz'    # true and predicted hidden values, for the figure
RUN_HELDOUT = False          # True: redo the 9 fits (about 25 minutes); False: load the saved results

def ho_subsample(n_per=None, seed=0):
    """Random n_per cells per embryo from the joint 2,500-per object, as a dense-X AnnData (NaN needs dense)."""
    n_per = HO_N_PER if n_per is None else n_per
    a = ad.read_h5ad(JOINT); rng = np.random.default_rng(seed)
    keep = np.sort(np.concatenate([rng.choice(np.where(a.obs.embryo.values == g)[0], n_per, replace=False) for g in sorted(a.obs.embryo.unique())]))
    a = a[keep].copy(); a.X = np.asarray(a.X.todense()).astype(np.float64)
    del a.layers['sct_counts']
    return a

def ho_fit(a, out, use_gp, log=print):
    """Same settings as fit_mefisto (K factors, medium convergence, seed 0, sparse GP with FRAC_INDUCING inducing points,
    (x, y, z) covariates) on an in-memory AnnData whose X may contain NaN (= hidden entries). Plain MOFA when use_gp=False.
    Returns the live entry_point (needed for GP prediction), Z (cells x K, internal factor order), W (genes x K) and wall time."""
    from mofapy2.run.entry_point import entry_point
    a.obs['sample'] = a.obs_names; n_hidden = int(np.isnan(a.X).sum())   # count before the fit- mofapy2 zeroes the NaNs in place once it has built its missing-value mask
    t0 = time.time()
    ent = entry_point()
    ent.set_data_options(scale_views=False, center_groups=True, use_float32=True)
    ent.set_data_from_anndata(a, groups_label='embryo', use_raw=False, save_metadata=True)
    ent.set_model_options(factors=K, spikeslab_weights=True, ard_weights=True, ard_factors=True)
    ent.set_train_options(iter=MAX_ITER, convergence_mode='medium', startELBO=1, freqELBO=5, seed=0, verbose=False, quiet=True, gpu_mode=False)
    if use_gp:
        ent.set_covariates(['x', 'y', 'z'], covariates_names=['x', 'y', 'z'])
        ent.set_smooth_options(scale_cov=False, model_groups=True, sparseGP=True, frac_inducing=FRAC_INDUCING, n_grid=10, start_opt=25, opt_freq=25)
    ent.build(); ent.run()
    wall = time.time() - t0
    if os.path.exists(out): log(f'  not overwriting existing {out}')
    else: ent.save(out, save_data=True)
    names = [n for grp in ent.data_opts['samples_names'] for n in grp]
    Z = pd.DataFrame(ent.model.nodes['Z'].getExpectations()['E'], index=names)
    W = pd.DataFrame(ent.model.nodes['W'].getExpectations()[0]['E'], index=a.var_names)
    log(f'  {out}: {wall/60:.1f} min, {a.n_obs} cells, {n_hidden:,} hidden entries')
    return ent, Z, W, wall

def ho_score(true, pred):
    return dict(r=float(np.corrcoef(true, pred)[0, 1]), mse=float(np.mean((true - pred) ** 2)))

def ho_test_A(seed, rows, times, preds, log=print):
    """Test A, masked entries: hide 10% of cell x gene entries, fit both models on the rest, reconstruct W.Z at the hidden entries."""
    a = ho_subsample(); X = a.X.copy(); groups = sorted(a.obs.embryo.unique())
    mask = np.random.default_rng(100 + seed).random(X.shape) < HO_HIDE
    am = a.copy(); am.X = X.copy(); am.X[mask] = np.nan
    for model in ['MEFISTO', 'MOFA']:
        out = f'models/heldout_A_{model.lower()}_seed{seed}.hdf5'
        log(f'test A seed {seed} {model} ...')
        ent, Z, W, wall = ho_fit(am.copy(), out, use_gp=(model == 'MEFISTO'), log=log)
        times.append(dict(test='A', seed=seed, model=model, n_cells=a.n_obs, wall_minutes=round(wall / 60, 2)))
        for g in groups:
            idx = np.where(a.obs.embryo.values == g)[0]
            mu = np.nanmean(am.X[idx], axis=0)                       # per-gene training mean of this embryo (what the model centres on)
            P = Z.loc[a.obs_names[idx]].to_numpy() @ W.to_numpy().T + mu
            hid = mask[idx]; true, pred = X[idx][hid], P[hid]
            rows.append(dict(test='A', seed=seed, model=model, embryo=g, n_hidden=int(hid.sum()), **ho_score(true, pred)))
            if model == 'MEFISTO':
                rows.append(dict(test='A', seed=seed, model='gene mean', embryo=g, n_hidden=int(hid.sum()), **ho_score(true, np.broadcast_to(mu, P.shape)[hid])))
            preds[f'A_{model}_{seed}_{g}'] = np.vstack([true, pred]).astype(np.float32)
        del ent

def ho_test_B(seed, rows, times, preds, log=print):
    """Test B, held-out cells: hide 10% of the cells of each embryo, fit MEFISTO on the rest, predict the hidden cells' factor values
    from their coordinates with the fitted GP (mofapy2's native predict_factor: sparse-GP posterior mean, K_new,U Sigma^-1 U, per
    factor and group), reconstruct W.Z, and compare with the mean of the k nearest observed cells in space and the gene mean."""
    from scipy.spatial import cKDTree
    a = ho_subsample(); X = a.X; groups = sorted(a.obs.embryo.unique()); rng = np.random.default_rng(200 + seed)
    hid = np.zeros(a.n_obs, bool)
    for g in groups:
        idx = np.where(a.obs.embryo.values == g)[0]; hid[rng.choice(idx, int(round(HO_HIDE * len(idx))), replace=False)] = True
    train, test = a[~hid].copy(), a[hid].copy()
    out = f'models/heldout_B_mefisto_seed{seed}.hdf5'
    log(f'test B seed {seed} MEFISTO ...')
    ent, Z, W, wall = ho_fit(train, out, use_gp=True, log=log)
    times.append(dict(test='B', seed=seed, model='MEFISTO', n_cells=train.n_obs, wall_minutes=round(wall / 60, 2)))
    gnames = list(ent.data_opts['groups_names'])
    cov_dtype = ent.model.nodes['Sigma'].sample_cov_transformed.dtype
    new_cov = test.obs[['x', 'y', 'z']].to_numpy().astype(cov_dtype)
    t0 = time.time(); ent.predict_factor(new_covariates=new_cov, uncertainty=False, groups='all')
    Zp, vals = ent.Zpredictions['mean'], ent.Zpredictions['values']; M = vals.shape[0]
    lookup = {tuple(v): j for j, v in enumerate(vals)}
    log(f'  predict_factor: {time.time()-t0:.1f} s for {M} coordinates x {len(gnames)} groups')
    for g in groups:
        gi = gnames.index(g)
        tr = np.where(train.obs.embryo.values == g)[0]; te = np.where(test.obs.embryo.values == g)[0]
        mu = train.X[tr].mean(axis=0)
        Zt = np.vstack([Zp[gi * M + lookup[tuple(c)]] for c in new_cov[te]])
        P = Zt @ W.to_numpy().T + mu
        true = test.X[te]
        rows.append(dict(test='B', seed=seed, model='MEFISTO', embryo=g, n_hidden=int(true.size), **ho_score(true.ravel(), P.ravel())))
        preds[f'B_MEFISTO_{seed}_{g}'] = np.vstack([true.ravel(), P.ravel()]).astype(np.float32)
        tree = cKDTree(train.obs[['x', 'y', 'z']].to_numpy()[tr]); _, nn = tree.query(test.obs[['x', 'y', 'z']].to_numpy()[te], k=HO_KNN)
        Pk = train.X[tr][nn].mean(axis=1)
        rows.append(dict(test='B', seed=seed, model=f'kNN in space (k={HO_KNN})', embryo=g, n_hidden=int(true.size), **ho_score(true.ravel(), Pk.ravel())))
        preds[f'B_kNN_{seed}_{g}'] = np.vstack([true.ravel(), Pk.ravel()]).astype(np.float32)
        rows.append(dict(test='B', seed=seed, model='gene mean', embryo=g, n_hidden=int(true.size), **ho_score(true.ravel(), np.broadcast_to(mu, true.shape).ravel())))
        rows.append(dict(test='B', seed=seed, model='MOFA', embryo=g, n_hidden=int(true.size), r=np.nan, mse=np.nan))   # not possible: no covariates, no way to place a new cell
    del ent

def ho_run(tests=('A', 'B'), seeds=HO_SEEDS, log=print):
    rows, times, preds = [], [], {}
    for s in seeds:
        if 'A' in tests: ho_test_A(s, rows, times, preds, log)
        if 'B' in tests: ho_test_B(s, rows, times, preds, log)
    return pd.DataFrame(rows), pd.DataFrame(times), preds
