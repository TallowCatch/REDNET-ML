# 14 — Model Comparison and Selection Rationale

This note summarizes the model families that were actually tried in REDNET-ML and how they should be compared. It is meant to support the dissertation, viva preparation, and future manuscript revisions.

Because the repo contains multiple run directories from different comparison stages, the numbers below are **artifact-backed comparisons**, with an explicit note for the later Random Forest migration gate on the canonical all-labels v2 surface.

The key point is that the repo contains **different kinds of models for different roles**:

- simple optical baselines
- tabular fusion baselines
- detector models used as image-evidence generators
- the final decision-fusion model

Those should not all be compared in one flat leaderboard. The right comparison depends on the role of the model and the split regime used.

## How to compare the models

1. Compare **tabular and fusion models** only against other tabular and fusion models, using AUPRC and AUROC under the same split policy.
2. Compare **detectors** only against other detectors, using region-level metrics such as regional recall and mean area overlap.
3. Treat small diagnostic benchmarks as **stress tests**, not as the main deployment claim.
4. Keep the **label regime fixed** when comparing numbers: trusted-only runs are not directly comparable to all-label monitoring runs.
5. For viva and report purposes, the most defensible story is:
   - simple baselines establish a floor
   - detector models provide image evidence
   - fusion beats any single stream
   - CatBoost remains the deployed decision layer because the later RF migration gate failed on thresholded recall

## Tabular and fusion models

| Model | Role | Representative evidence in repo | Why it was tried | Outcome |
|---|---|---|---|---|
| FAI-only index baseline | Simplest explainable optical baseline | `runs/eval/benchmark/labeled_bench_time_2017_2023_vs_2024_v2/summary_mean_std.csv` shows `index::fai_mean` with AUROC `0.591` and AUPRC `0.229` on the 2024 labeled benchmark | To test whether a single interpretable optical signal already carried useful HAB information | Kept only as a sanity-check baseline. It was too weak to support the final monitoring system on its own. |
| Logistic regression | Transparent linear tabular baseline | `runs/fusion/baseline/logreg_recall60_timecv4/summary_cv.csv` shows mean AUPRC `0.606` and mean AUROC `0.574` | To provide a simple, inspectable baseline for engineered Sentinel, MODIS, and detector-summary features | Retained as the main linear baseline, but not selected as final because the fusion problem contains heterogeneous features and non-linear interactions that a linear model cannot express well. |
| Random forest | First-class non-linear challenger on the canonical plant-level fusion surface | `runs/fusion/plants/fusion_alllabels_cv5_rf/summary_cv.csv` shows mean AUPRC `0.759` and mean AUROC `0.882`; `runs/fusion/plants/fusion_alllabels_cv5_rf/gate_vs_catboost.md` records the official migration gate against CatBoost | To test whether the later plant-level fusion workflow improved if CatBoost was replaced by a stronger tree ensemble under the same folds and features | RF improved pooled AUPRC and AUROC, but it was **not** promoted to deployment because the thresholded recall dropped from `0.388` to `0.128` under the existing operating policy, which failed the migration gate. |
| CatBoost decision fusion | Current deployed tabular decision layer | `runs/fusion/baseline/catboost_recall60_timecv4/summary_cv.csv` shows mean AUPRC `0.639` and mean AUROC `0.624`; `runs/fusion/plants/fusion_alllabels_cv5_v2/summary_cv.csv` shows mean AUPRC `0.741` and mean AUROC `0.870` on the canonical deployed run | To model mixed feature types and non-linear interactions in the fused table without moving to a harder-to-defend end-to-end deep classifier | Remains the deployed decision-fusion model because it cleared the existing operating behavior requirements, whereas the later official RF migration failed the recall gate. |
| CatBoost on trusted-only labels | High-confidence upper-bound check rather than the main deployment setting | `runs/fusion/plants/fusion_trustedonly_cv5/summary_cv.csv` shows mean AUPRC `0.945` and mean AUROC `0.998` on `97` positives | To test how much cleaner supervision sharpens separability | Useful as an upper-bound reference, but not used as the main headline because it has much lower positive coverage than the all-label monitoring setting. |

## Single-stream versus fused evidence

This is the most useful comparison for explaining *why* fusion was necessary.

| Evidence stream | Representative metrics | Interpretation |
|---|---|---|
| Tabular-only score `P_TAB` | `runs/fusion/qc_model_comparison/all_model_metrics.csv`: AUROC `0.559`, F1 `0.531` | A single tabular stream was not enough. |
| Detector-only scores | Same file: AUROC roughly `0.430` to `0.459` across the detector-only score columns in that QC run | Individual detector summaries alone were weak as final decision scores. |
| Fused detector + environment + tabular score | Same file: AUROC `0.835`, F1 `0.776` | The main value came from fusion rather than any single evidence source. |

## Detector models

Detector models should be compared using the detector benchmark criteria, not the tabular AUPRC/AUROC criteria above. The detector role is to generate bloom-like spatial evidence for the later fusion stage.

| Detector | Regional recall | Mean area overlap | Why it was tried | Outcome |
|---|---|---:|---|---|
| Faster R-CNN ResNet50 | `0.94` | `0.69` | Strong two-stage detector with better localization capacity | Retained. Best overall detector in the repo benchmark. |
| SSD MobileNet | `0.94` | `0.66` | Lightweight one-stage detector with better speed-efficiency tradeoff | Retained. Best efficiency-accuracy tradeoff and complementary to FRCNN ResNet50. |
| Faster R-CNN MobileNet | `0.44` | `0.31` | Lighter baseline to test whether a cheaper Faster R-CNN backbone was sufficient | Not retained as a main evidence generator. It looked capacity-limited. |
| YOLOv8n3 | `0.92` | `0.50` | Strong one-stage baseline and common detector family | Not retained. Recall was competitive, but spatial alignment was weaker and it did not add enough ensemble value to justify the extra complexity. |

Detector numbers above match the dissertation detector table in the LaTeX report.

## Short answers for "why this model?"

### Why use CatBoost instead of logistic regression?

Because the final fusion table mixes detector summaries, environmental predictors, optical indices, ratios, and seasonal terms. Logistic regression is a useful linear baseline, but CatBoost is a better fit for heterogeneous tabular features and non-linear interactions. The direct time-CV comparison in the repo shows CatBoost outperforming logistic regression.

### Why not switch the whole pipeline to Random Forest if it scored higher on the later rerun?

Because the official RF migration gate was not based only on pooled ranking metrics. On the canonical all-labels v2 surface, RF did improve pooled AUPRC and AUROC, but it also dropped pooled recall from `0.388` to `0.128` under the existing threshold policy. That failed the migration acceptance gate, so CatBoost stayed as the deployed model.

### Why not claim Random Forest as the best model if it scored 1.0 in the older benchmark?

Because that result comes from a small diagnostic labeled benchmark, not from the main plant-level fusion experiment. It is useful evidence that the signal is learnable, but it is not the right basis for the final deployment claim.

### Why use detectors as evidence generators instead of final classifiers?

Because the image labels are limited and noisy. Treating detector outputs as summarized evidence is more defensible than asking a detector alone to decide final HAB risk. This preserves image structure while keeping the final decision in a tabular fusion layer that is easier to inspect and justify.

### Why keep both Faster R-CNN ResNet50 and SSD MobileNet?

Because they offered the best complementary tradeoff in the detector benchmark: Faster R-CNN ResNet50 had the best localization quality, while SSD MobileNet matched recall with a lighter, more efficient architecture.

### Why was YOLO not kept?

Because it did not materially improve ensemble coverage or overlap enough to justify extra complexity in the final system.

### Why is AUPRC more important than accuracy?

Because HAB monitoring is imbalanced and the main operational goal is to surface useful high-risk candidates. AUPRC reflects the precision-recall tradeoff more directly than accuracy.

## Recommended wording for the dissertation or viva

If you need one compact answer, use this:

> Logistic regression was kept as the transparent linear baseline, Random Forest was used as a diagnostic non-linear benchmark, detector models were used as image-evidence generators, and CatBoost was selected as the final decision-fusion model because it handled the heterogeneous tabular inputs and non-linear interactions best in the main reproducible fusion workflow.

For the later canonical migration gate, a more precise wording is:

> Random Forest was promoted to a first-class challenger on the canonical all-labels v2 fusion surface and improved pooled AUPRC and AUROC, but it was not adopted as the deployed model because it failed the operating-behavior gate by sharply reducing pooled recall under the existing threshold policy. CatBoost therefore remained the deployed fusion model.
