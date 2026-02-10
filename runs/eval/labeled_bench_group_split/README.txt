Rigorous labeled benchmark
=========================
Input: runs/datasets/fusion_training_with_plants_splitlabels.csv

Label col: hab_label
Score cols: ['p_frcnn_r50_med', 'p_frcnn_mb_med', 'p_ssd_mb_med']
Index baseline: fai_mean
RF baseline: yes

Split mode: group
  - time: train<= 2023, test== 2024
  - group: GroupShuffleSplit test_size=0.2 per seed

Runs: 10 seeds
Threshold policy: best F1 on TRAIN only
Bootstrap: 1000 resamples per seed (ROC-AUC / PR-AUC)

Outputs:
- per_seed_metrics.csv
- summary_mean_std.csv
- bootstrap_ci_per_seed.json
- delong_vs_main.csv (paired ROC-AUC significance vs main model)
- mcnemar_vs_main.csv (paired error shift significance vs main model)
