#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve,
    classification_report, confusion_matrix
)

# ---------------- plotting ----------------
def _pr_plot(rec, prec, auprc, base, outpng):
    plt.figure(figsize=(5.2, 4.2))
    plt.plot(rec, prec)
    if base is not None:
        plt.plot([0, 1], [base, base], "--", alpha=0.35)
    ttl = f"PR (AUPRC={auprc:.3f}"
    if base is not None:
        ttl += f", baseline={base:.3f}"
    ttl += ")"
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(ttl)
    plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close()

def _roc_plot(fpr, tpr, auroc, outpng):
    plt.figure(figsize=(5.2, 4.2))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--", alpha=0.35)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUROC={auroc:.3f})")
    plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close()

# --------------- helpers ------------------
def _clean_columns(df: pd.DataFrame, src: str = "") -> pd.DataFrame:
    """Strip whitespace and drop duplicate-named columns (keep first)."""
    df = df.copy()
    cols = [str(c).strip() for c in df.columns]
    dup_mask = pd.Index(cols).duplicated(keep="first")
    if dup_mask.any():
        dropped = [c for i, c in enumerate(cols) if dup_mask[i]]
        print(f"[warn] {src} had duplicated columns {dropped}; keeping first occurrence.")
        df = df.loc[:, ~dup_mask]
        cols = [c for i, c in enumerate(cols) if not dup_mask[i]]
    df.columns = cols
    # also drop unnamed index columns if any
    bad = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if bad:
        df = df.drop(columns=bad)
    return df

def _coerce_id(df: pd.DataFrame, id_col: str, src: str) -> pd.DataFrame:
    """Ensure df has id_col; accept common alternatives and normalize."""
    df = df.copy()
    if id_col not in df.columns:
        if "chip_id" in df.columns:
            df = df.rename(columns={"chip_id": id_col})
        elif "tile" in df.columns:
            df = df.rename(columns={"tile": id_col})
        elif "image" in df.columns:
            df[id_col] = df["image"].apply(lambda s: Path(str(s)).name)
        else:
            raise SystemExit(f"{src} has no id column '{id_col}'. Columns: {list(df.columns)}")
    return df

def _normalize_ids(s: pd.Series) -> pd.Series:
    """Force IDs to comparable 'name.ext' strings."""
    return s.astype(str).apply(lambda x: Path(x).name)

def _guess_score_col(df: pd.DataFrame, id_col: str) -> str:
    """Pick a numeric score column; prefer p_*/score*, ignore *_count."""
    numeric = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
    cand = [c for c in numeric if str(c).lower().startswith("p_") or "score" in str(c).lower()]
    cand = [c for c in cand if not str(c).lower().endswith("count")]
    if cand:
        return cand[0]
    # fallback: first numeric not *_count
    numeric = [c for c in numeric if not str(c).lower().endswith("count")]
    if numeric:
        return numeric[0]
    # last resort: first non-id column
    for c in df.columns:
        if c != id_col:
            return c
    raise SystemExit("Could not guess score column.")

# ---------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabular_csv", required=True,
                    help="CSV with tabular p_tab and labels (columns: id_col, scene_id, hab_label, p_tab)")
    ap.add_argument("--det", nargs="*", default=[],
                    help="named detector CSVs: name=path.csv (CSV must contain id_col and a score column)")
    ap.add_argument("--outdir", default="runs/fusion/decision_fusion")
    ap.add_argument("--id_col", default="chip_id")
    ap.add_argument("--group_by", default="scene_id")
    ap.add_argument("--min_pos_per_split", type=int, default=2)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--max_tries", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # --- load base/tabular ---
    base = pd.read_csv(args.tabular_csv)
    base = _clean_columns(base, args.tabular_csv)
    base = _coerce_id(base, args.id_col, args.tabular_csv)
    base[args.id_col] = _normalize_ids(base[args.id_col])

    # sanity: ensure single copy of group_by / required columns
    need = {args.id_col, args.group_by, "hab_label", "p_tab"}
    missing = need - set(base.columns)
    if missing:
        raise SystemExit(f"{args.tabular_csv} missing columns: {sorted(missing)}")

    # If somehow group_by still duplicated, collapse to the first non-null column
    if pd.Index(base.columns).duplicated().any():
        base = _clean_columns(base, f"{args.tabular_csv} (post)")

    base = base[[args.id_col, args.group_by, "hab_label", "p_tab"]].copy()

    # --- merge detector scores ---
    feats = ["p_tab"]
    for spec in args.det:
        if "=" not in spec:
            raise SystemExit("Use name=path.csv for --det (e.g., frcnn_r50=runs/fusion/p_frcnn_r50.csv)")
        name, path = spec.split("=", 1)
        df = pd.read_csv(path)
        df = _clean_columns(df, path)
        df = _coerce_id(df, args.id_col, path)
        df[args.id_col] = _normalize_ids(df[args.id_col])
        score_col = _guess_score_col(df, args.id_col)
        df = df.rename(columns={score_col: name})
        base = base.merge(df[[args.id_col, name]], on=args.id_col, how="left")
        if base[name].isna().all():
            print(f"[warn] Detector '{name}' merged with all-NaN scores. "
                  f"Likely ID mismatch between {args.tabular_csv} and {path}.")
        feats.append(name)

    # fill missing detector scores with 0
    base[feats] = base[feats].fillna(0.0)

    # --- matrices ---
    X = base[feats].values
    y = base["hab_label"].astype(int).values
    G = base[args.group_by].astype(str).values

    # --- group-aware split with positives on both sides ---
    rng = np.random.RandomState(args.seed)
    good = None
    for t in range(1, args.max_tries + 1):
        gss = GroupShuffleSplit(test_size=args.test_size, random_state=int(rng.randint(0, 10_000)))
        tr, te = next(gss.split(X, y, groups=G))
        if y[tr].sum() >= args.min_pos_per_split and y[te].sum() >= args.min_pos_per_split:
            good = (tr, te); tries = t; break
    if good is None:
        raise SystemExit("Could not find a split with positives in both partitions; lower --min_pos_per_split.")
    tr, te = good
    print(f"[Split] train rows={len(tr)} (pos={int(y[tr].sum())}) | "
          f"test rows={len(te)} (pos={int(y[te].sum())}) | tries={tries}")

    # --- fusion learner ---
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipe.fit(X[tr], y[tr])

    p_te = pipe.predict_proba(X[te])[:, 1]
    base_rate = float(y[te].mean()) if y[te].size else None
    auprc = average_precision_score(y[te], p_te)
    auroc = roc_auc_score(y[te], p_te)
    prec, rec, thr = precision_recall_curve(y[te], p_te)
    fpr, tpr, _ = roc_curve(y[te], p_te)

    f1s = (2 * prec * rec / (prec + rec + 1e-12))[:-1]
    i = int(np.argmax(f1s)) if len(f1s) else 0
    thr_star = float(thr[i]) if len(thr) else 0.5
    yhat = (p_te >= thr_star).astype(int)

    cm = confusion_matrix(y[te], yhat).tolist()
    rep = classification_report(y[te], yhat, digits=3)
    print(f"AUPRC: {auprc:.3f}  AUROC: {auroc:.3f}  thr*: {thr_star:.3f}  (baseline={base_rate:.3f})")
    print("Confusion matrix [[TN,FP],[FN,TP]]:", cm)
    print(rep)

    # --- save ---
    joblib.dump({"pipe": pipe, "features": feats}, outdir / "fusion_model.joblib")
    (outdir / "metrics.json").write_text(json.dumps({
        "feats": feats,
        "auprc": float(auprc),
        "auroc": float(auroc),
        "thr_star": thr_star,
        "base_rate": base_rate,
        "cm": cm,
        "n_test": int(len(te)),
        "pos_test": int(y[te].sum()),
    }, indent=2))

    _pr_plot(rec, prec, auprc, base_rate, outdir / "pr_fusion.png")
    _roc_plot(fpr, tpr, auroc, outdir / "roc_fusion.png")

    pd.DataFrame([{
        "model": "fusion(logreg)",
        "feats": "+".join(feats),
        "auprc": auprc, "auroc": auroc,
        "thr_star": thr_star,
        "test_pos": int(y[te].sum()),
        "test_total": int(len(te))
    }]).to_csv(outdir / "summary.csv", index=False)
    print(f"✓ saved {outdir/'fusion_model.joblib'}, metrics.json, PR/ROC plots, summary.csv")

if __name__ == "__main__":
    main()
