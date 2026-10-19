# Shell runner for the held-out validation (notebook section 3, MEFISTO vs plain MOFA). Usage: python -u heldout.py A|B|AB [n_per] [seeds]
import os, sys, json, time, numpy as np, pandas as pd, anndata as ad
K = 12; FRAC_INDUCING = 0.1; MAX_ITER = 300; JOINT = 'data/e8_joint_2500per.h5ad'
exec(open('heldout_funcs.py').read())
tests = sys.argv[1] if len(sys.argv) > 1 else 'AB'
if len(sys.argv) > 2: HO_N_PER = int(sys.argv[2])
if len(sys.argv) > 3: HO_SEEDS = [int(x) for x in sys.argv[3].split(',')]
if len(sys.argv) > 4: MAX_ITER = int(sys.argv[4])
tag = os.environ.get('HO_TAG', '')
def log(msg): print(time.strftime('%H:%M:%S'), msg, flush=True)
log(f'tests={tests} n_per={HO_N_PER} seeds={HO_SEEDS} K={K} frac={FRAC_INDUCING} iters={MAX_ITER}')
R, T, P = ho_run(tests=tests, seeds=HO_SEEDS, log=log)
R.to_csv(f'heldout_results_{tag}{tests}.csv', index=False); T.to_csv(f'heldout_times_{tag}{tests}.csv', index=False)
np.savez_compressed(f'models/heldout_predictions_{tag}{tests}.npz', **P)
log('done'); print(R.round(4).to_string())
