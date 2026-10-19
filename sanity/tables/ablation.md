| flip (one setting from ours toward official) | file:line | status | total R2 % | delta vs ours | active factors | Ttr factor | mean smoothness | iterations | train min |
|---|---|---|---|---|---|---|---|---|---|
| ours (reference) |  | done | 32.20 | +0.00 | 9 | F4 | 0.68 | 300 | 12.6 |
| scale_views False -> True | fit.py:10 | done | 33.40 | +1.20 | 12 | F6 | 0.66 | 270 | 7.3 |
| inducing points 10% (269) -> 1,000 (37%) | fit.py argv frac | killed at 40 min (143 it, ELBO 3096295) |  |  |  |  |  |  |  |
| sparse GP -> full GP (all 2,695 spots) | fit.py:16 sparseGP | killed at 40 min (49 it, ELBO 3132165) |  |  |  |  |  |  |  |
| K 12 -> 4 | fit.py argv K | done | 26.62 | -5.58 | 4 | F4 | 0.61 | 300 | 3.6 |
| n_grid 10 -> 20 | fit.py:16 NGRID | done | 32.23 | +0.03 | 9 | F6 | 0.65 | 300 | 11.1 |
| start_opt/opt_freq 25/25 -> 20/10 | fit.py:16 OPTFREQ | done | 32.28 | +0.08 | 9 | F6 | 0.61 | 300 | 11.7 |
| convergence medium (ELBO every 5) -> fast (every 1) | fit.py:13 | done | 32.25 | +0.05 | 9 | F7 | 0.64 | 94 | 3.3 |
| convergence medium -> slow, cap 450 | fit.py:13 | done | 32.14 | -0.06 | 9 | F5 | 0.66 | 450 | 13.9 |
| float32 -> float64 | fit.py:10 | done | 32.24 | +0.04 | 9 | F7 | 0.62 | 170 | 6.3 |
| seed 0 -> 2021 | fit.py:13 | done | 32.47 | +0.27 | 9 | F6 | 0.61 | 300 | 9.9 |
| ard_factors True -> False | fit.py:12 | done | 32.36 | +0.16 | 9 | F5 | 0.62 | 300 | 8.5 |
| HVG flavor seurat (dispersion on log) -> seurat_v3 (vst on counts) | prep.py:19 | done | 39.64 | +7.44 | 10 | F7, F10 | 0.61 | 300 | 10.8 |
| genes in >= 10 spots before HVG selection | prep.py (no line; new filter before :19) | done | 31.32 | -0.88 | 8 | F6 | 0.59 | 300 | 10.8 |
| every fit setting official (prep.py preprocessing kept) | fit.py:10-16 | done | 29.39 | -2.81 | 4 | F4 | 0.63 | 54 | 9.6 |
| every fit setting official + vst HVGs (= official pipeline on 10x counts) | prep.py:19 + fit.py:10-16 | done | 35.19 | +2.99 | 4 | F4 | 0.63 | 74 | 15.2 |