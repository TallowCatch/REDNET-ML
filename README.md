
# REDNET-ML

This repository contains an end-to-end pipeline for harmful algal bloom (HAB) risk modelling over Oman. The system combines:

- Remote sensing features (Sentinel‑2 chip-level indices and MODIS L3 products)
- HAB labelling and candidate mining
- Object detector signals
- Tabular baseline models
- Decision-level fusion into a unified risk score
- Reproducible evaluation and logging

The main reproducibility path is the notebook suite in `rednet_notebooks/`, which runs the pipeline in numeric order and writes outputs into the `runs/` directory.

---

## Repository layout

```
rednet_notebooks/    Reproducibility notebooks (00 → 11)
scripts/             Pipeline scripts (download, HAB prep, fusion, evaluation)
data/                Generated datasets and intermediate tables
runs/                Model artifacts, plots, predictions, metrics
```

Key script folders:

- `scripts/download/` – Sentinel‑2 + MODIS acquisition helpers
- `scripts/HAB/` – HAB labelling, candidate mining, baseline models
- `scripts/fusion/` – Fusion table construction and decision-level fusion
- `scripts/eval/` – Evaluation and benchmarking tools

`REDNET_ML.ipynb` is preserved as a legacy exploratory notebook. The numbered notebooks are the canonical reproducible path.

---

## Quickstart

### 1. Create environment

```
conda create -n rednet-ml python=3.11 -y
conda activate rednet-ml
```

Install dependencies:

If available:

```
conda env update -n rednet-ml -f environment.yml
```

Otherwise:

```
pip install -r requirements.txt
pip install tabulate
```

---

### 2. Configure OB.DAAC authentication (MODIS)

You must authenticate with Earthdata / OB.DAAC using one of:

- a valid `$HOME/.netrc`
- an app key via environment variable

```
export OBPG_APPKEY="YOUR_KEY"
```

---

## Reproducibility workflow

Start Jupyter:

```
conda activate rednet-ml
jupyter lab
```

Run notebooks in order:

```
rednet_notebooks/00_*.ipynb
...
rednet_notebooks/11_*.ipynb
```

The notebooks call scripts with fixed arguments and write outputs to `data/` and `runs/`. No manual parameter editing should be required for a clean reproduction.

---

## Pipeline overview

### Phase A — Sentinel‑2 chip indices

- STAC discovery and download
- 8‑day chip extraction
- Index computation into CSV tables

Outputs: chip-level feature tables under `data/`.

---

### Phase B — MODIS feature append

- Download MODIS L3 8‑day products
- Append features directly into chip CSVs
- Resumable download → append → cleanup loop

Temporary `.nc` files are deleted after successful append to control disk usage.

---

### Phase C — HAB labelling and mining

- Season-aware labelling
- Thresholding with quality controls
- Candidate mining for detectors

Produces labelled chip tables and training candidates.

---

### Phase D — Tabular baseline models

Baseline HAB risk models are trained on engineered features. Logistic regression is used by default, but the feature table supports stronger learners (e.g., CatBoost) without modification.

---

### Phase E — Detector outputs

Object detector runs generate CSV signals used later in fusion. Comparison helpers aggregate detector performance across runs.

---

### Phase F — Decision-level fusion

Fusion builds a unified training table and fits a final risk model.

Artifacts written to:

```
runs/fusion/
```

Including:

- predictions
- ROC / PR plots
- calibration outputs
- metrics
- model artifacts

---

### Phase G — Evaluation and generalization

- Consolidated run summaries
- Benchmark evaluation
- Cross-time generalization diagnostics

Canonical benchmark directory:

```
runs/eval/benchmark/labeled_bench_time_2017_2023_vs_2024_diag
```

---

## Outputs

```
data/   Raw + intermediate datasets
runs/   All evaluation artifacts and reported metrics
```

Re-running notebooks regenerates fresh metrics and plots.

---

## Common pitfalls

- Missing OB.DAAC authentication → download failures
- Incorrect MODIS product naming (case sensitive)
- Disk space pressure if cleanup is disabled

---

## Citation

If using this repository in a report or publication, cite the associated dissertation or preprint and reference the repository commit hash used.

---

## Contact

Open an issue for:

- environment setup problems
- dataset regeneration questions
- reproduction mismatches
