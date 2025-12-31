#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 140

# ---------- small metrics helpers ----------
def _auc(x, y):
    # assumes x monotonic; trapezoid
    x, y = np.asarray(x), np.asarray(y)
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))

def _roc(y_true, y_score):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score)
    thr = np.unique(s)[::-1]
    tpr, fpr = [], []
    P, N = (y==1).sum(), (y==0).sum()
    for t in thr:
        yp = (s >= t).astype(int)
        tp = ((yp==1)&(y==1)).sum()
        fp = ((yp==1)&(y==0)).sum()
        fn = ((yp==0)&(y==1)).sum()
        tn = ((yp==0)&(y==0)).sum()
        tpr.append(tp/max(P,1))
        fpr.append(fp/max(N,1))
    return np.array(fpr), np.array(tpr)

def _pr(y_true, y_score):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score)
    thr = np.unique(s)[::-1]
    prec, rec = [], []
    P = (y==1).sum()
    for t in thr:
        yp = (s >= t).astype(int)
        tp = ((yp==1)&(y==1)).sum()
        fp = ((yp==1)&(y==0)).sum()
        prec.append(tp/max(tp+fp,1))
        rec.append(tp/max(P,1))
    return np.array(rec), np.array(prec)

def _sweep_best(y_true, y_score, key="f1"):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score)
    thr = np.quantile(s, np.linspace(0.01, 0.99, 199))
    best = {"thr":0.5, "acc":0.0, "f1":0.0}
    for t in thr:
        yp = (s >= t).astype(int)
        tp = ((yp==1)&(y==1)).sum()
        fp = ((yp==1)&(y==0)).sum()
        fn = ((yp==0)&(y==1)).sum()
        tn = ((yp==0)&(y==0)).sum()
        prec = tp/max(tp+fp,1)
        rec  = tp/max(tp+fn,1)
        f1   = 2*prec*rec/max(prec+rec,1e-9)
        acc  = (tp+tn)/max(len(y),1)
        score = f1 if key=="f1" else acc
        if score > best[key]:
            best = {"thr":float(t), "acc":float(acc), "f1":float(f1)}
    return best

# ---------- load, merge, compute ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_dir", default="runs/fusion/fused_sets/B_mined_timecv_norm_f1")
    ap.add_argument("--root_sets", default="runs/fusion/fused_sets",
                    help="folder that contains p_frcnn_*.csv and tab_*.csv if needed")
    ap.add_argument("--env_csv", default="runs/datasets/hab_candidates_review.csv")
    ap.add_argument("--outdir", default="qc/showcase_mined_b/model_comp")
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # pick predictions file (validation, not *_train)
    fdir = Path(args.fusion_dir)
    pred_file = next((p for p in [fdir/"predictions_cv2.csv", fdir/"predictions.csv"] if p.exists()), None)
    if pred_file is None:
        raise SystemExit("No predictions file found (predictions_cv2.csv / predictions.csv).")
    df = pd.read_csv(pred_file)

    # If some model columns are missing, supplement from top-level fused_sets CSVs
    # Expected columns: p_tab, frcnn_r50, frcnn_mb, frcnn_ssd, maybe p_fused
    need_cols = {"frcnn_r50":"p_frcnn_r50.csv", "frcnn_mb":"p_frcnn_mb.csv", "frcnn_ssd":"p_frcnn_ssd.csv"}
    for col, fname in need_cols.items():
        if col not in df.columns:
            p = Path(args.root_sets) / fname
            if p.exists():
                aux = pd.read_csv(p)  # must have scene_id, score column named same as file? Normalize:
                # be tolerant: pick first numeric column that is not hab_label/scene_id
                numcols = [c for c in aux.select_dtypes(include=np.number).columns if c not in ["hab_label"]]
                if "scene_id" in aux.columns and numcols:
                    aux = aux[["scene_id", numcols[0]]].rename(columns={numcols[0]:col})
                    df = df.merge(aux, on="scene_id", how="left")

    # bring in env for display (optional)
    env = Path(args.env_csv)
    if env.exists():
        envdf = pd.read_csv(env)
        cols = [c for c in ["scene_id","chlor_a","kd490","nflh"] if c in envdf.columns]
        if cols:
            df = df.merge(envdf[cols], on="scene_id", how="left")

    # models dict (only keep those present)
    candidates = {
        "Fusion B": "p_fused",
        "Tabular B": "p_tab",
        "FRCNN R50": "frcnn_r50",
        "FRCNN MB": "frcnn_mb",
        "FRCNN SSD": "frcnn_ssd",
    }
    ycol = "hab_label" if "hab_label" in df.columns else ("y_true" if "y_true" in df.columns else None)
    if ycol is None:
        raise SystemExit("Need hab_label/y_true column in predictions CSV.")

    models = {name:col for name,col in candidates.items() if col in df.columns}
    if not models:
        raise SystemExit("No model score columns found (p_fused/p_tab/frcnn_*).")

    # compute curves + AUCs + best thresholds
    roc_png = out/"comp_roc.png"
    pr_png  = out/"comp_pr.png"
    bar_png = out/"comp_bar.png"
    metrics_csv = out/"comp_metrics.csv"

    rows=[]
    # ROC
    plt.figure()
    for name,col in models.items():
        fpr, tpr = _roc(df[ycol].values, df[col].values)
        auc = _auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUROC {auc:.3f})")
        rows.append({"model":name, "metric":"auroc", "value":auc})
        best = _sweep_best(df[ycol].values, df[col].values, key="f1")
        rows += [
            {"model":name, "metric":"thr_f1", "value":best["thr"]},
            {"model":name, "metric":"acc_at_f1", "value":best["acc"]},
            {"model":name, "metric":"f1_best", "value":best["f1"]},
        ]
    plt.plot([0,1],[0,1],"--",alpha=.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC comparison"); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(roc_png); plt.close()

    # PR
    plt.figure()
    for name,col in models.items():
        rec, prec = _pr(df[ycol].values, df[col].values)
        auprc = _auc(rec, prec)
        plt.plot(rec, prec, label=f"{name} (AUPRC {auprc:.3f})")
        rows.append({"model":name, "metric":"auprc", "value":auprc})
    base = float((df[ycol]==1).mean())  # class prior
    plt.hlines(base, 0, 1, colors="k", linestyles="--", label=f"baseline {base:.2f}", alpha=.35)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR comparison")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(pr_png); plt.close()

    # Bar chart (AUROC, AUPRC, ACC@F1)
    m = pd.DataFrame(rows)
    pivot = (m[m["metric"].isin(["auroc","auprc","acc_at_f1"])]
             .pivot(index="model", columns="metric", values="value").fillna(0.0))
    pivot = pivot.reindex(models.keys())  # preserve order
    ax = pivot.plot(kind="bar")
    ax.set_ylim(0,1.05)
    ax.set_ylabel("Score"); ax.set_title("Overall metrics")
    ax.legend(title="")
    plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(bar_png); plt.close()

    pivot.round(4).to_csv(metrics_csv)
    print(f"✓ wrote {roc_png}\n✓ wrote {pr_png}\n✓ wrote {bar_png}\n✓ wrote {metrics_csv}")

if __name__ == "__main__":
    main()
