import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# -----------------------------
# Config
# -----------------------------
ROOT = Path("runs/fusion/plants/fusion_alllabels_cv5_v2")
FOLDS = [1, 2, 3, 4, 5]
LABEL_COL = "hab_label_final2"
SCORE_COL = "p_fused"

OUT_ROC = ROOT / "roc_cv_all.png"
OUT_PR  = ROOT / "pr_cv_all.png"

# -----------------------------
# Helpers
# -----------------------------
def style_ax(ax):
    # remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # optional: remove left/bottom too (uncomment if you want *no* frame at all)
    # ax.spines["left"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    ax.tick_params(direction="out")

def mean_roc_curve(fprs, tprs, n=300):
    """Interpolate TPRs onto a common FPR grid and average."""
    grid = np.linspace(0, 1, n)
    tprs_interp = []
    for fpr, tpr in zip(fprs, tprs):
        # ensure starts at 0 and ends at 1 for interpolation stability
        fpr = np.asarray(fpr)
        tpr = np.asarray(tpr)
        if fpr[0] != 0:
            fpr = np.r_[0.0, fpr]
            tpr = np.r_[0.0, tpr]
        if fpr[-1] != 1:
            fpr = np.r_[fpr, 1.0]
            tpr = np.r_[tpr, 1.0]
        tprs_interp.append(np.interp(grid, fpr, tpr))
    mean_tpr = np.mean(tprs_interp, axis=0)
    return grid, mean_tpr

def mean_pr_curve(recs, precs, n=300):
    """
    Average PR by interpolating precision as a function of recall on a common recall grid.

    Note: PR is not uniquely defined for averaging; this is a common, defensible visualization.
    """
    grid = np.linspace(0, 1, n)
    precs_interp = []
    for rec, prec in zip(recs, precs):
        rec = np.asarray(rec)
        prec = np.asarray(prec)

        # precision_recall_curve returns rec in increasing order already.
        # It can contain duplicates; keep the maximum precision for each recall.
        df = pd.DataFrame({"rec": rec, "prec": prec}).groupby("rec", as_index=False)["prec"].max()
        rec_u = df["rec"].values
        prec_u = df["prec"].values

        # Interp needs increasing x; we have that.
        # For recalls outside range, clamp to end values.
        prec_i = np.interp(grid, rec_u, prec_u, left=prec_u[0], right=prec_u[-1])
        precs_interp.append(prec_i)

    mean_prec = np.mean(precs_interp, axis=0)

    # Optional: enforce a non-increasing precision envelope (common PR presentation)
    mean_prec = np.maximum.accumulate(mean_prec[::-1])[::-1]
    return grid, mean_prec

# -----------------------------
# Load folds and compute curves
# -----------------------------
roc_aucs = []
pr_auprcs = []
baselines = []

fprs, tprs = [], []
recs, precs = [], []

for k in FOLDS:
    csv_path = ROOT / f"predictions_cv{k}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

    df = pd.read_csv(csv_path)
    y_true = df[LABEL_COL].astype(int).values
    y_score = df[SCORE_COL].astype(float).values

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    baseline = float(np.mean(y_true))

    fprs.append(fpr); tprs.append(tpr)
    recs.append(rec); precs.append(prec)

    roc_aucs.append(float(roc_auc))
    pr_auprcs.append(float(auprc))
    baselines.append(baseline)

roc_mean = float(np.mean(roc_aucs))
roc_std  = float(np.std(roc_aucs, ddof=1))

pr_mean  = float(np.mean(pr_auprcs))
pr_std   = float(np.std(pr_auprcs, ddof=1))

base_mean = float(np.mean(baselines))

# Mean curves
fpr_grid, mean_tpr = mean_roc_curve(fprs, tprs, n=400)
rec_grid, mean_prec = mean_pr_curve(recs, precs, n=400)

# -----------------------------
# Plot ROC (all folds + mean)
# -----------------------------
plt.figure(figsize=(6, 5))
ax = plt.gca()
style_ax(ax)

# individual folds (light)
for i, (fpr, tpr, a) in enumerate(zip(fprs, tprs, roc_aucs), start=1):
    plt.plot(fpr, tpr, lw=1.2, alpha=0.25, label=None)

# mean (bold)
plt.plot(fpr_grid, mean_tpr, lw=2.5, label=f"Mean ROC (AUC = {roc_mean:.3f} ± {roc_std:.3f})")

# diagonal
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Bloom Detection (5-fold CV)")
plt.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig(OUT_ROC, dpi=220)
plt.show()

# -----------------------------
# Plot PR (all folds + mean)
# -----------------------------
plt.figure(figsize=(6, 5))
ax = plt.gca()
style_ax(ax)

for rec, prec in zip(recs, precs):
    plt.step(rec, prec, where="post", lw=1.2, alpha=0.25)

plt.step(rec_grid, mean_prec, where="post", lw=2.5,
         label=f"Mean PR (AUPRC = {pr_mean:.3f} ± {pr_std:.3f})")

plt.hlines(base_mean, 0, 1, linestyles="--", color="gray", lw=1,
           label=f"Baseline = {base_mean:.2f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Bloom Detection (5-fold CV)")
plt.legend(loc="lower left", frameon=False)
plt.tight_layout()
plt.savefig(OUT_PR, dpi=220)
plt.show()

print(f"Saved:\n- {OUT_ROC}\n- {OUT_PR}")
