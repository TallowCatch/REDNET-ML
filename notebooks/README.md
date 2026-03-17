# REDNET-ML — Reproducibility Notebook Suite (v2)

This folder contains **14 Jupyter notebooks** (00 → 13) that reproduce the full REDNET-ML workflow **in the same phases as the codebase**:

1) environment + sanity checks  
2) AOI + plant AOIs  
3) MODIS filelists + downloads  
4) Sentinel‑2 chipping + indices  
5) MODIS feature append (8‑day alignment)  
6) label mining + candidates + non‑leaky training table  
7) detector datasets (COCO)  
8) detector training  
9) fusion model training  
10) benchmarking + generalization (recompute AUROC/AUPRC from predictions)  
11) end‑to‑end monthly inference (S2 + MODIS on-demand) + optional detectors + fusion rerun  
12) risk-field aggregation + Kepler package + viewer
13) external validation setup + claim boundaries

> **Design goal**: the notebooks are “no manual arg editing” by:
> - centralizing paths and parameters in the first cell of each notebook
> - printing `--help` for every script that may change flags over time
> - auto-detecting common locations (e.g., `data/filelists/8d` vs `filelists/8d`)
> - recomputing benchmark metrics from saved prediction CSVs (so you never rely on old figures)

---

## 0) Setup (terminal)

From the repo root:

```bash
mamba env create -f cfg/environment.yml -n rednet-ml   # or conda env create ...
conda activate rednet-ml

# If your env.yml doesn't include all python deps:
pip install -r requirements.txt

python -m ipykernel install --user --name rednet-ml --display-name "rednet-ml"

# OBDAAC downloads require:
export OBPG_APPKEY="YOUR_OBPG_KEY"

jupyter lab
```

Open the notebooks from this folder, using the `rednet-ml` kernel.

---

## Notebook execution order

| Phase | Notebook | What it produces |
|---|---|---|
| 0 | `00_setup_environment.ipynb` | env commands + repo sanity checks |
| 1 | `01_project_overview.ipynb` | repo map + Mermaid flow diagrams |
| 2 | `02_aoi_and_plants.ipynb` | AOI verification + optional plant AOI rebuild |
| 3 | `03_modis_filelists_downloads.ipynb` | filelists verified / regenerated |
| 4 | `04_s2_chipping_and_indices.ipynb` | `chips/index.csv`, `chips/chip_indices.csv` |
| 5 | `05_append_modis_features.ipynb` | downloads MODIS subsets and appends features into `chip_indices.csv` |
| 6 | `06_label_mining_and_nonleaky.ipynb` | candidate mining + `hab_train_nonleaky*.csv` |
| 7 | `07_detector_dataset_coco.ipynb` | COCO splits + QC tools |
| 8 | `08_train_detectors.ipynb` | detector weights under `detection_models/` / training runs |
| 9 | `09_tabular_fusion_training.ipynb` | fusion run directory with trained bundle (`*.joblib`) |
| 10 | `10_generalization_benchmark.ipynb` | loads **`runs/eval/benchmark/labeled_bench_time_2017_2023_vs_2024_diag`** and recomputes AUROC/AUPRC |
| 11 | `11_end_to_end_inference.ipynb` | per-month inference outputs + merged `inference_all_months.csv` |
| 12 | `12_risk_field_and_viewer.ipynb` | plant alerts, Kepler package, viewer steps |
| 13 | `13_external_validation.ipynb` | public event validation, advisory coverage, in-situ schema, and claim boundaries |

---

## Benchmark / generalization run directory (Notebook 10)

This suite treats the following folder as the **source of truth** for generalization:

```
runs/eval/benchmark/labeled_bench_time_2017_2023_vs_2024_diag
```

Notebook 10 will:
- list outputs in that folder
- locate a predictions CSV (if present)
- recompute AUROC/AUPRC using `sklearn` so metrics are always current

If only figures exist, Notebook 10 prints `--help` for `scripts/eval/rigorous_labeled_bench.py` and provides the exact rerun command.

---

## Notes / constraints

- Detector training is compute-heavy; the notebooks keep commands ready-to-run but you may not want to execute them in a lightweight environment.
- OBDAAC downloads require `OBPG_APPKEY`.
- Output roots default to the project’s `deployment/outputs/by_plant/...` structure to match the inference pipeline.
