#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from math import pi
from sklearn.metrics import roc_auc_score

plt.rcParams["figure.dpi"] = 140

# ----- metric helpers -----
def binary_metrics(y_true, y_score, thr=None):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score)
    if thr is None:
        thr = np.nanmedian(s)
    yp = (s >= thr).astype(int)
    tp = ((yp==1)&(y==1)).sum()
    fp = ((yp==1)&(y==0)).sum()
    fn = ((yp==0)&(y==1)).sum()
    tn = ((yp==0)&(y==0)).sum()
    prec = tp/max(tp+fp,1)
    rec  = tp/max(tp+fn,1)
    f1   = 2*prec*rec/max(prec+rec,1e-9)
    acc  = (tp+tn)/len(y)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1)

# ----- plotting -----
def plot_bar(df, out_png):
    ax = df.plot(x="model", kind="bar", rot=0, colormap="viridis")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.title("Model comparison — accuracy, F1, precision, recall, AUROC")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_radar(df, out_png):
    cats = ["acc","prec","rec","f1","auroc"]
    N = len(cats)
    angles = [n/float(N)*2*pi for n in range(N)]
    angles += angles[:1]
    plt.figure(figsize=(4,4))
    ax = plt.subplot(111, polar=True)
    for _,r in df.iterrows():
        vals=[r[c] for c in cats]+[r[cats[0]]]
        ax.plot(angles,vals,label=r["model"])
        ax.fill(angles,vals,alpha=.1)
    plt.xticks(angles[:-1],cats)
    plt.ylim(0,1)
    plt.title("Model comparison radar chart")
    plt.legend(loc="upper right",bbox_to_anchor=(1.35,1.1),fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png,bbox_inches="tight")
    plt.close()

# ----- main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_dir", default="runs/fusion/fused_sets")
    ap.add_argument("--outdir", default="qc/showcase_mined_b/detector_comp")
    ap.add_argument("--fusion_dir", default="runs/fusion/fused_sets/B_mined_timecv_norm_f1")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---------------- DETECTORS ----------------
    models = {
        "FRCNN_MB": "p_frcnn_with_HAB_label_mb.csv",
        "FRCNN_R50": "p_frcnn_with_HAB_label_r50.csv",
        "FRCNN_SSD": "p_frcnn_with_HAB_label_ssd.csv"
    }

    for name, f in models.items():
        p = Path(args.det_dir) / f
        if not p.exists():
            print(f"⚠️ Missing {f}")
            continue

        df = pd.read_csv(p)
        if "hab_label" not in df.columns:
            print(f"❌ {f} has no hab_label column, skipping")
            continue

        score_col = [c for c in df.columns if c.startswith("p_frcnn_")][0]
        df = df.dropna(subset=["hab_label", score_col])
        y = df["hab_label"].astype(int).values
        s = df[score_col].astype(float).values

        au = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
        thr = np.nanmedian(s)
        m = binary_metrics(y, s, thr)
        m["auroc"] = au
        m["model"] = name
        rows.append(m)

    # ---------------- TABULAR + FUSION ----------------
    fusion_dir = Path(args.fusion_dir)
    fusion_file = fusion_dir / "predictions_cv2.csv"
    if fusion_file.exists():
        df = pd.read_csv(fusion_file)
        if "hab_label" in df.columns:
            y = df["hab_label"].astype(int).values

            # Tabular model (p_tab)
            if "p_tab" in df.columns:
                s_tab = df["p_tab"].astype(float).fillna(0.5).values
                thr_tab = np.nanmedian(s_tab)
                m = binary_metrics(y, s_tab, thr_tab)
                m["auroc"] = roc_auc_score(y, s_tab)
                m["model"] = "TABULAR_B"
                rows.append(m)

            # Fusion model (p_fused)
            for col in ["p_fused", "p_fusion", "p_final"]:
                if col in df.columns:
                    s_fused = df[col].astype(float).fillna(0.5).values
                    thr_fused = np.nanmedian(s_fused)
                    m = binary_metrics(y, s_fused, thr_fused)
                    m["auroc"] = roc_auc_score(y, s_fused)
                    m["model"] = "FUSION"
                    rows.append(m)
                    break

    # ---------------- FUSION CV METRICS ----------------
    cv_json = fusion_dir / "metrics_cv2.json"
    cv_csv = fusion_dir / "summary_cv.csv"
    if cv_json.exists():
        with open(cv_json, "r") as f:
            metrics = json.load(f)
        f1_cv = np.nan
        prec_cv = np.nan
        rec_cv = np.nan
        acc_cv = np.nan

        if cv_csv.exists():
            df_cv = pd.read_csv(cv_csv)
            if "f1" in df_cv.columns:
                f1_cv = df_cv["f1"].mean()
            if "precision" in df_cv.columns:
                prec_cv = df_cv["precision"].mean()
            if "recall" in df_cv.columns:
                rec_cv = df_cv["recall"].mean()
            if "accuracy" in df_cv.columns:
                acc_cv = df_cv["accuracy"].mean()

        rows.append({
            "model": "FUSION_CV",
            "acc": acc_cv,
            "prec": prec_cv,
            "rec": rec_cv,
            "f1": f1_cv,
            "auroc": metrics.get("auroc", np.nan)
        })

    # ---------------- OUTPUT ----------------
    if not rows:
        raise SystemExit("❌ No valid CSVs found.")

    df = pd.DataFrame(rows)[["model","acc","prec","rec","f1","auroc"]]
    df = df.sort_values("f1", ascending=False)
    df.round(4).to_csv(out / "all_model_metrics.csv", index=False)

    plot_bar(df, out / "all_model_bar.png")
    plot_radar(df, out / "all_model_radar.png")

    print(f"✓ Saved comparison to {out}\n", df.round(3))

if __name__ == "__main__":
    main()
