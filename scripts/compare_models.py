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


# -------------------- extra metric helpers (from fusion charts script) --------------------
def _auc(x, y):
    """Trapezoidal AUC assuming x is monotonic."""
    x, y = np.asarray(x), np.asarray(y)
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def _roc(y_true, y_score):
    """Simple ROC curve (FPR, TPR) using unique thresholds on scores."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score)
    thr = np.unique(s)[::-1]
    tpr, fpr = [], []
    P, N = (y == 1).sum(), (y == 0).sum()
    for t in thr:
        yp = (s >= t).astype(int)
        tp = ((yp == 1) & (y == 1)).sum()
        fp = ((yp == 1) & (y == 0)).sum()
        fn = ((yp == 0) & (y == 1)).sum()
        tn = ((yp == 0) & (y == 0)).sum()
        tpr.append(tp / max(P, 1))
        fpr.append(fp / max(N, 1))
    return np.array(fpr), np.array(tpr)


def _pr(y_true, y_score):
    """Precision–Recall curve (recall, precision) using unique thresholds."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score)
    thr = np.unique(s)[::-1]
    prec, rec = [], []
    P = (y == 1).sum()
    for t in thr:
        yp = (s >= t).astype(int)
        tp = ((yp == 1) & (y == 1)).sum()
        fp = ((yp == 1) & (y == 0)).sum()
        prec.append(tp / max(tp + fp, 1))
        rec.append(tp / max(P, 1))
    return np.array(rec), np.array(prec)


# -------------------- metric helpers --------------------
def binary_metrics(y_true, y_score, thr=None):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)

    # default: median threshold if not given
    if thr is None:
        thr = np.nanmedian(s)

    yp = (s >= thr).astype(int)
    tp = ((yp == 1) & (y == 1)).sum()
    fp = ((yp == 1) & (y == 0)).sum()
    fn = ((yp == 0) & (y == 1)).sum()
    tn = ((yp == 0) & (y == 0)).sum()

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / len(y)

    return dict(acc=acc, prec=prec, rec=rec, f1=f1)


# -------------------- plotting --------------------
def plot_bar(df, out_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(x="model", kind="bar", rot=0, colormap="viridis", ax=ax)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison — accuracy, F1, precision, recall, AUROC")

    # Rotate x labels & reduce font size to avoid overlap
    ax.set_xticklabels(df["model"], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()


def plot_radar(df, out_png):
    cats = ["acc", "prec", "rec", "f1", "auroc"]
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure(figsize=(5.2, 5.2))
    ax = plt.subplot(111, polar=True)

    # improved radar colors and transparency
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(df)))

    for idx, (_, r) in enumerate(df.iterrows()):
        vals = [r[c] for c in cats] + [r[cats[0]]]
        ax.plot(angles, vals, label=r["model"], color=colors[idx], lw=1.8)
        ax.fill(angles, vals, alpha=0.1, color=colors[idx])

    plt.xticks(angles[:-1], cats, fontsize=9)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=8)
    plt.ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.title("Model comparison radar chart", fontsize=11, pad=20)
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.15, 1.02),
        fontsize=8,
        frameon=False,
        title="Model",
    )
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()


# -------------------- fusion helper --------------------
def add_fusion_block(rows, tag_name, fusion_dir: Path):
    """Read a fusion run folder (predictions_cv2*.csv, metrics_cv2.json) and append rows."""
    fusion_dir = Path(fusion_dir)
    pred_test = fusion_dir / "predictions_cv2.csv"
    pred_train = fusion_dir / "predictions_cv2_train.csv"
    metrics_json = fusion_dir / "metrics_cv2.json"

    added_any = False

    # TEST
    if pred_test.exists():
        dft = pd.read_csv(pred_test)
        if {"hab_label", "p_fused"}.issubset(dft.columns):
            y = dft["hab_label"].astype(int).values
            s = dft["p_fused"].astype(float).values
            m = binary_metrics(y, s)
            m["auroc"] = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
            m["model"] = tag_name
            rows.append(m)
            added_any = True
        else:
            print(f"⚠️ {pred_test} missing 'hab_label' or 'p_fused' columns.")

    # TRAIN (optional, useful for seeing overfit)
    if pred_train.exists():
        dft = pd.read_csv(pred_train)
        if {"hab_label", "p_fused"}.issubset(dft.columns):
            y = dft["hab_label"].astype(int).values
            s = dft["p_fused"].astype(float).values
            m = binary_metrics(y, s)
            m["auroc"] = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
            m["model"] = tag_name + "_TRAIN"
            rows.append(m)
            added_any = True
        else:
            print(f"⚠️ {pred_train} missing 'hab_label' or 'p_fused' columns.")

    # If nothing else, fall back to metrics json just to get AUROC
    if (not added_any) and metrics_json.exists():
        with open(metrics_json, "r") as f:
            met = json.load(f)
        rows.append(
            {
                "model": tag_name,
                "acc": np.nan,
                "prec": np.nan,
                "rec": np.nan,
                "f1": np.nan,
                "auroc": met.get("auroc", np.nan),
            }
        )
        added_any = True

    if not added_any:
        print(f"⚠️ No usable fusion predictions/metrics found in {fusion_dir}")

    return added_any


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tabular_csv",
        default="runs/fusion/fusion_training_table_clean.csv",
        help="Main training table with cols: tile, scene_id, datetime, month_key, env features, hab_label, p_*",
    )
    ap.add_argument(
        "--fusion_det_dir",
        default="",
        help="(Optional) fusion run folder for detectors-only fusion (predictions_cv2.csv there)",
    )
    ap.add_argument(
        "--fusion_tab_dir",
        default="runs/fusion/fusion_simple_v1",
        help="Fusion run folder for detectors+env+p_tab (predictions_cv2.csv there)",
    )
    ap.add_argument("--outdir", default="runs/fusion/qc_model_comparison")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # this will store y/scores for ROC+PR curves
    curve_sources = {}

    # ---------------- BASE TABLE MODELS ----------------
    tab_path = Path(args.tabular_csv)
    if not tab_path.exists():
        raise SystemExit(f"❌ tabular_csv not found: {tab_path}")

    tab = pd.read_csv(tab_path)
    cols = tab.columns.tolist()

    # labels
    if "hab_label" not in tab.columns:
        raise SystemExit(f"❌ 'hab_label' not found in {tab_path} (have: {cols})")
    y = tab["hab_label"].astype(int).values

    # model columns present in your fusion_training_table_clean.csv
    model_cols = {
        "P_TAB": "p_tab",
        "FRCNN_R50_MED": "p_frcnn_r50_med",
        "FRCNN_MB_MED": "p_frcnn_mb_med",
        "SSD_MB_MED": "p_ssd_mb_med",
    }

    rows = []

    for name, col in model_cols.items():
        if col not in tab.columns:
            print(f"⚠️ Skipping {name}: column '{col}' not in {tab_path}")
            continue

        s = tab[col].astype(float).values
        # drop NaNs if any
        mask = ~np.isnan(s)
        y_sub = y[mask]
        s_sub = s[mask]

        if len(y_sub) == 0:
            print(f"⚠️ {name}: all scores are NaN; skipping.")
            continue

        m = binary_metrics(y_sub, s_sub)
        m["auroc"] = roc_auc_score(y_sub, s_sub) if len(np.unique(y_sub)) > 1 else np.nan
        m["model"] = name
        rows.append(m)

        # store for ROC/PR curves
        curve_sources[name] = (y_sub, s_sub)

    # ---------------- FUSION RUNS ----------------
    if args.fusion_det_dir:
        add_fusion_block(rows, "FUSION_DET_ONLY", Path(args.fusion_det_dir))
        # also store predictions for ROC/PR if available
        det_pred = Path(args.fusion_det_dir) / "predictions_cv2.csv"
        if det_pred.exists():
            dft = pd.read_csv(det_pred)
            if {"hab_label", "p_fused"}.issubset(dft.columns):
                y_fd = dft["hab_label"].astype(int).values
                s_fd = dft["p_fused"].astype(float).values
                mask = ~np.isnan(s_fd)
                curve_sources["FUSION_DET_ONLY"] = (y_fd[mask], s_fd[mask])

    if args.fusion_tab_dir:
        add_fusion_block(rows, "FUSION_DET+ENV+TAB", Path(args.fusion_tab_dir))
        # also store predictions for ROC/PR if available
        tab_pred = Path(args.fusion_tab_dir) / "predictions_cv2.csv"
        if tab_pred.exists():
            dft = pd.read_csv(tab_pred)
            if {"hab_label", "p_fused"}.issubset(dft.columns):
                y_ft = dft["hab_label"].astype(int).values
                s_ft = dft["p_fused"].astype(float).values
                mask = ~np.isnan(s_ft)
                curve_sources["FUSION_DET+ENV+TAB"] = (y_ft[mask], s_ft[mask])

    # ---------------- Δ IMPROVEMENT (if both presence) ----------------
    df_tmp = pd.DataFrame(rows)
    if {"FUSION_DET_ONLY", "FUSION_DET+ENV+TAB"}.issubset(set(df_tmp["model"])):
        base_row = df_tmp[df_tmp["model"] == "FUSION_DET_ONLY"].iloc[0]
        tab_row = df_tmp[df_tmp["model"] == "FUSION_DET+ENV+TAB"].iloc[0]
        rows.append(
            {
                "model": "DELTA_TAB+ENV_vs_DET",
                "acc": (
                    tab_row["acc"] - base_row["acc"]
                    if pd.notna(tab_row["acc"]) and pd.notna(base_row["acc"])
                    else np.nan
                ),
                "prec": (
                    tab_row["prec"] - base_row["prec"]
                    if pd.notna(tab_row["prec"]) and pd.notna(base_row["prec"])
                    else np.nan
                ),
                "rec": (
                    tab_row["rec"] - base_row["rec"]
                    if pd.notna(tab_row["rec"]) and pd.notna(base_row["rec"])
                    else np.nan
                ),
                "f1": (
                    tab_row["f1"] - base_row["f1"]
                    if pd.notna(tab_row["f1"]) and pd.notna(base_row["f1"])
                    else np.nan
                ),
                "auroc": (
                    tab_row["auroc"] - base_row["auroc"]
                    if pd.notna(tab_row["auroc"]) and pd.notna(base_row["auroc"])
                    else np.nan
                ),
            }
        )

    # ---------------- OUTPUT ----------------
    if not rows:
        raise SystemExit("❌ No models to evaluate (no rows).")

    df = pd.DataFrame(rows)[["model", "acc", "prec", "rec", "f1", "auroc"]]

    # put DELTA row at bottom
    df = df.sort_values(
        by=["model"],
        key=lambda s: s.apply(lambda x: 1 if x.startswith("DELTA_") else 0)
    )

    df.round(4).to_csv(out / "all_model_metrics.csv", index=False)

    # only plot actual models
    df_plot = df[~df["model"].str.startswith("DELTA_")].reset_index(drop=True)
    if len(df_plot):
        plot_bar(df_plot, out / "all_model_bar.png")
        plot_radar(df_plot, out / "all_model_radar.png")

    # ---------------- ROC & PR COMPARISON (new charts) ----------------
    if curve_sources:
        # ROC
        roc_png = out / "all_model_roc.png"
        plt.figure()
        for model, (yt, ys) in curve_sources.items():
            if len(np.unique(yt)) < 2:
                continue  # skip degenerate
            fpr, tpr = _roc(yt, ys)
            auc = _auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{model} (AUROC {auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", alpha=0.4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC comparison")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(roc_png, dpi=200)
        plt.close()

        # PR
        pr_png = out / "all_model_pr.png"
        plt.figure()
        baseline_set = False
        baseline_val = 0.0
        for model, (yt, ys) in curve_sources.items():
            if len(np.unique(yt)) < 2:
                continue
            rec, prec = _pr(yt, ys)
            auprc = _auc(rec, prec)
            plt.plot(rec, prec, label=f"{model} (AUPRC {auprc:.3f})")
            if not baseline_set:
                baseline_val = float((np.asarray(yt) == 1).mean())
                baseline_set = True
        if baseline_set:
            plt.hlines(
                baseline_val,
                0,
                1,
                colors="k",
                linestyles="--",
                label=f"baseline {baseline_val:.2f}",
                alpha=0.35,
            )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR comparison")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(pr_png, dpi=200)
        plt.close()

    print(f"\n✓ Saved comparison to {out}")
    print(df.round(3))


if __name__ == "__main__":
    main()
