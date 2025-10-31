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


# -------------------- metric helpers --------------------
def binary_metrics(y_true, y_score, thr=None):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score)
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
    ax = df.plot(x="model", kind="bar", rot=0, colormap="viridis")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.title("Model comparison — accuracy, F1, precision, recall, AUROC")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_radar(df, out_png):
    cats = ["acc", "prec", "rec", "f1", "auroc"]
    N = len(cats)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    plt.figure(figsize=(4, 4))
    ax = plt.subplot(111, polar=True)
    for _, r in df.iterrows():
        vals = [r[c] for c in cats] + [r[cats[0]]]
        ax.plot(angles, vals, label=r["model"])
        ax.fill(angles, vals, alpha=0.1)
    plt.xticks(angles[:-1], cats)
    plt.ylim(0, 1)
    plt.title("Model comparison radar chart")
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# -------------------- file helper --------------------
def find_first_existing(candidates, roots):
    """
    candidates: ['p_frcnn_mb_mkey.csv', 'p_frcnn_mb.csv', ...]
    roots: list of folders to search, in order
    returns Path or None
    """
    for root in roots:
        root = Path(root)
        for cand in candidates:
            p = root / cand
            if p.exists():
                return p
    return None


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_dir", default="runs/fusion/fused_sets",
                    help="primary dir to look for detector CSVs (we also auto-search runs/fusion/)")
    ap.add_argument("--outdir", default="runs/fusion/qc_showcase_comparison")
    # detector-only fusion (your earlier run, detectors only)
    ap.add_argument("--fusion_det_dir", default="runs/fusion/fused_sets/B_mined_timecv_norm_f1")
    # detector + tabular fusion (your v2 run)
    ap.add_argument("--fusion_tab_dir", default="runs/fusion/fused_sets/fusion_enriched_norm_f1_v2")
    ap.add_argument("--label_csv", default="runs/datasets/hab_train_mined_aslabel.csv")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # we will search in several places, because your files are in runs/fusion/
    detector_search_roots = [
        Path(args.det_dir),
        Path(args.det_dir).parent,           # e.g. runs/fusion/
        Path("runs/fusion"),
        Path("runs/fusion/fused_sets"),
    ]

    # ---------------- INDIVIDUAL DETECTORS ----------------
    # these are the actual names you showed in the screenshot
    models = {
        "FRCNN_MB": ["p_frcnn_mb_mkey.csv", "p_frcnn_mb.csv", "p_frcnn_with_HAB_label_mb.csv"],
        "FRCNN_R50": ["p_frcnn_r50_mkey.csv", "p_frcnn_r50.csv", "p_frcnn_with_HAB_label_r50.csv"],
        "FRCNN_SSD": ["p_frcnn_ssd_mkey.csv", "p_frcnn_ssd.csv", "p_frcnn_with_HAB_label_ssd.csv"],
    }

    # load labels once
    labels = pd.read_csv(args.label_csv)
    labels.columns = labels.columns.str.strip().str.lower()
    label_col = next((c for c in ["hab_label_final", "hab_label", "hab_label_heuristic"] if c in labels.columns), None)
    if label_col is None:
        raise SystemExit(f"❌ No usable label column in {args.label_csv}")
    labels["tile"] = labels["tile"].astype(str).str.strip()

    for name, candidates in models.items():
        p = find_first_existing(candidates, detector_search_roots)
        if p is None:
            print(f"⚠️ Missing detector file for {name} (tried {candidates})")
            continue

        df = pd.read_csv(p)
        # normalize id col
        if "tile" in df.columns:
            df["tile"] = df["tile"].astype(str).str.strip()
        elif "chip_id" in df.columns:
            df = df.rename(columns={"chip_id": "tile"})
            df["tile"] = df["tile"].astype(str).str.strip()
        else:
            print(f"❌ {p.name} has no 'tile' or 'chip_id' column. Columns: {df.columns.tolist()}")
            continue

        # Merge by month_key (detector vs Sentinel month)
        if "month_key" not in df.columns or "month_key" not in labels.columns:
            print(f"⚠️ {p.name} missing month_key; skipping monthly merge")
            continue

        df_merged = df.merge(
            labels[["month_key", label_col]].rename(columns={label_col: "hab_label"}),
            on="month_key",
            how="left",
        )

        matched = df_merged["hab_label"].notna().sum()
        print(f"✅ {name}: matched {matched} / {len(df_merged)} from {p}")
        if matched == 0:
            continue

        # detector score col
        score_cols = [c for c in df_merged.columns if c.startswith("p_frcnn_") or c.startswith("p_ssd_")]
        if not score_cols:
            print(f"⚠️ {name}: no score column starting with p_frcnn_ or p_ssd_ in {p.name}")
            continue
        score_col = score_cols[0]

        df_merged = df_merged.dropna(subset=["hab_label", score_col])
        y = df_merged["hab_label"].astype(int).values
        s = df_merged[score_col].astype(float).values

        au = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
        m = binary_metrics(y, s)
        m["auroc"] = au
        m["model"] = name
        rows.append(m)

    # ---------------- FUSION (DETECTORS ONLY) ----------------
    # we read test first (predictions_cv2.csv)
    def add_fusion_block(tag_name, fusion_dir: Path):
        added_any = False
        pred_test = fusion_dir / "predictions_cv2.csv"
        pred_train = fusion_dir / "predictions_cv2_train.csv"
        metrics_json = fusion_dir / "metrics_cv2.json"

        # test
        if pred_test.exists():
            dft = pd.read_csv(pred_test)
            if "hab_label" in dft.columns and "p_fused" in dft.columns:
                y = dft["hab_label"].astype(int).values
                s = dft["p_fused"].astype(float).values
                m = binary_metrics(y, s)
                m["auroc"] = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
                m["model"] = tag_name  # e.g. FUSION_DET_ONLY
                rows.append(m)
                added_any = True

        # train (optional) – this is where your 0.637 lives
        if pred_train.exists():
            dft = pd.read_csv(pred_train)
            if "hab_label" in dft.columns and "p_fused" in dft.columns:
                y = dft["hab_label"].astype(int).values
                s = dft["p_fused"].astype(float).values
                m = binary_metrics(y, s)
                m["auroc"] = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
                m["model"] = tag_name + "_TRAIN"
                rows.append(m)
                added_any = True

        # if there is a metrics json, we can at least log AUROC
        if (not added_any) and metrics_json.exists():
            with open(metrics_json, "r") as f:
                met = json.load(f)
            rows.append({
                "model": tag_name,
                "acc": np.nan,
                "prec": np.nan,
                "rec": np.nan,
                "f1": np.nan,
                "auroc": met.get("auroc", np.nan),
            })
            added_any = True

        return added_any

    # detectors-only fusion
    add_fusion_block("FUSION_DET_ONLY", Path(args.fusion_det_dir))
    # detectors + tabular fusion
    add_fusion_block("FUSION_DET+TAB", Path(args.fusion_tab_dir))

    # ---------------- Δ IMPROVEMENT ----------------
    # if both present, add a delta row
    df_tmp = pd.DataFrame(rows)
    if {"FUSION_DET_ONLY", "FUSION_DET+TAB"}.issubset(set(df_tmp["model"])):
        base_row = df_tmp[df_tmp["model"] == "FUSION_DET_ONLY"].iloc[0]
        tab_row = df_tmp[df_tmp["model"] == "FUSION_DET+TAB"].iloc[0]
        rows.append({
            "model": "DELTA_TAB_vs_DET",
            "acc": (tab_row["acc"] - base_row["acc"]) if pd.notna(tab_row["acc"]) and pd.notna(base_row["acc"]) else np.nan,
            "prec": (tab_row["prec"] - base_row["prec"]) if pd.notna(tab_row["prec"]) and pd.notna(base_row["prec"]) else np.nan,
            "rec": (tab_row["rec"] - base_row["rec"]) if pd.notna(tab_row["rec"]) and pd.notna(base_row["rec"]) else np.nan,
            "f1": (tab_row["f1"] - base_row["f1"]) if pd.notna(tab_row["f1"]) and pd.notna(base_row["f1"]) else np.nan,
            "auroc": (tab_row["auroc"] - base_row["auroc"]) if pd.notna(tab_row["auroc"]) and pd.notna(base_row["auroc"]) else np.nan,
        })

    # ---------------- OUTPUT ----------------
    if not rows:
        raise SystemExit("❌ No valid CSVs found.")

    df = pd.DataFrame(rows)[["model", "acc", "prec", "rec", "f1", "auroc"]]
    # sort: real models first, delta last
    df = df.sort_values(
        by=["model"],
        key=lambda s: s.apply(lambda x: 1 if x.startswith("DELTA_") else 0)
    )
    df.round(4).to_csv(out / "all_model_metrics.csv", index=False)

    # only plot real models
    df_plot = df[~df["model"].str.startswith("DELTA_")].reset_index(drop=True)
    plot_bar(df_plot, out / "all_model_bar.png")
    plot_radar(df_plot, out / "all_model_radar.png")

    print(f"\n✓ Saved comparison to {out}")
    print(df.round(3))


if __name__ == "__main__":
    main()
