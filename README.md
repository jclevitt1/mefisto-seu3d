# MEFISTO on the SEU-3D E8.0 embryos

MEFISTO and plain MOFA (mofapy2 0.7.5) fit to the two E8.0 mouse embryos of the SEU-3D digital embryo (Xie et al., Cell 2025;
GEO GSE278603). This is a known-answer test of the method before applying it to the lab's spatial data.

## Start here

`stuck_here.ipynb` is where I am stuck: why MEFISTO scores below plain MOFA on these embryos. It is laid out as three
hypotheses, each followed by the evidence for and against it. `variance_explained.ipynb` is the earlier report- why per-entry variance
explained is low on this data, the score that is informative here (R2 on cell-type averages), and the sanity checks against the MEFISTO
paper's own Visium example. Both only load saved models. To rebuild, run `python build_stuck_here.py` then execute.

## Code

| file | what it does |
|---|---|
| `prep.py` | joint AnnData for the two embryos: SCT counts, normalise, log1p, 2,000 variable genes, 2,500 cells per embryo |
| `fit.py` | one MEFISTO fit (embryos as groups, x/y/z as covariates) or plain MOFA with `nogp` |
| `analysis.py` | model loader, factor summary, figures 1 to 3, R2 on cell-type averages (`pseudobulk_r2`) |
| `compare.py` | MEFISTO vs MOFA: figure 4, classifier and Leiden against the atlas labels, held-out tables, figure 6 |
| `heldout.py`, `heldout_funcs.py` | masked-entry and held-out-cell validation |
| `paper_recipe.py` | the MEFISTO paper's Visium recipe (vst genes, paper fit settings) applied to the embryos, plus the PCA ceiling |
| `sanity/` | the paper's Visium example re-run with this code: `scripts/` (fits and summaries), `tables/`, `figures/` |
| `extra/` | the common-frame (anterior-posterior aligned) refit and its figures |

## Reproduce

Python 3.12 with mofapy2 0.7.5, scanpy, anndata, scikit-learn, scikit-misc, leidenalg, igraph, jupyter. Data: GSE278603_RAW.tar from
GEO into `data/`. Run in order (MEFISTO fits take 15 to 30 minutes each on a laptop, plain MOFA under a minute):

```
python prep.py 2500
OPTFREQ=25 python fit.py data/e8_joint_2500per.h5ad models/mefisto_e8_2500per_K12.hdf5 12 300 0.1
python fit.py data/e8_joint_2500per.h5ad models/mofa_e8_2500per_K12.hdf5 12 300 0.1 nogp
python paper_recipe.py prep
python paper_recipe.py fit 4; python paper_recipe.py fit 12; python paper_recipe.py fit 12 nogp
python heldout.py A; python heldout.py B
python analysis.py models/mefisto_e8_2500per_K12.hdf5 "" models/mofa_e8_2500per_K12.hdf5
python compare.py
jupyter nbconvert --to notebook --execute --inplace variance_explained.ipynb
```

The Visium sanity test needs the 10x `V1_Mouse_Brain_Sagittal_Anterior` counts and the paper's model from figshare (13233860) in
`sanity/data/`. `sanity/scripts/run_fit.py` lists every fit by name and `run_batch.sh` runs them.

`data/` and `models/` are not in the repo (about 2 GB per MEFISTO model).
