Rigorous labeled benchmark
=========================
Input: rednet-risk-viewer/public/data/plant_*_hab.csv

Label col: hab_label
Score cols: ['hab_prob', 'p_frcnn_r50_med', 'p_frcnn_mb_med', 'p_ssd_mb_med']
Index baseline: fai_mean

Split mode: rolling_year
  - time: train<= 2023, test== 2024
  - group: GroupShuffleSplit test_size=0.2 per seed

Runs: 10 seeds
Threshold policy: best F1 on TRAIN only
Bootstrap: 1000 resamples per seed (ROC-AUC / PR-AUC)

RF baseline:
  - mode: auto
  - extra drop: []

Diagnostics printed to console:
  - group_col chosen
  - per-seed train/test sizes, pos rates
  - per-seed group overlap (must be 0)
  - RF top importances (if RF runs)

Outputs:
- per_seed_metrics.csv
- summary_mean_std.csv
- bootstrap_ci_per_seed.json
- delong_vs_main.csv
- mcnemar_vs_main.csv
