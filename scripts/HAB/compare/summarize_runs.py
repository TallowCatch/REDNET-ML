#!/usr/bin/env python3
"""
Summarize HAB baseline runs.

- Reads metrics.json from one or more run directories
- Computes extra metrics from the confusion matrix
- Writes:
    runs/summary/hab_runs_summary.csv
    runs/summary/hab_runs_summary.md
- Also prints a neat console table

Usage examples:
    python scripts/HAB/summarize_runs.py runs/hab_no_leak_*
    python scripts/HAB/summarize_runs.py runs/hab_no_leak_A runs/hab_no_leak_B_kd runs/hab_no_leak_C_spatial
"""
import argparse, json, os, sys
from pathlib import Path
import pandas as pd

def load_one(run_dir: Path) -> dict | None:
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        return None
    with open(mpath, "r") as f:
        m = json.load(f)

    # Pull what we can
    auprc = float(m.get("auprc", float("nan")))
    auroc = float(m.get("auroc", float("nan")))
    thr   = float(m.get("threshold", m.get("threshold_f1", float("nan"))))

    cm = m.get("confusion_matrix", None)
    tn = fp = fn = tp = None
    if cm and isinstance(cm, list) and len(cm)==2 and len(cm[0])==2 and len(cm[1])==2:
        tn, fp = cm[0]
        fn, tp = cm[1]

    pos = int(m.get("test_pos", tp if tp is not None else 0))
    neg = int(m.get("test_neg", tn if tn is not None else 0))
    n   = pos + neg
    base = pos / n if n else float("nan")  # prevalence (random PR baseline)

    # Derive a few more metrics when possible
    prec = tp / (tp + fp) if tp is not None and (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if tp is not None and (tp + fn) > 0 else float("nan")
    acc  = (tp + tn) / n if n else float("nan")
    spec = tn / (tn + fp) if tn is not None and (tn + fp) > 0 else float("nan")
    f1   = (2*prec*rec)/(prec+rec) if prec==prec and rec==rec and (prec+rec)>0 else float("nan")

    features = m.get("features", [])
    group_by = m.get("group_by", "")

    return {
        "run": run_dir.name,
        "dir": str(run_dir),
        "features": ", ".join(features) if isinstance(features, list) else str(features),
        "group_by": group_by,
        "test_rows": n,
        "test_pos": pos,
        "test_neg": neg,
        "baseline_prevalence": base,
        "AUPRC": auprc,
        "AUROC": auroc,
        "thr": thr,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "specificity": spec,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }

def to_markdown(df: pd.DataFrame) -> str:
    # Keep a compact set of columns for MD view
    cols = [
        "run", "AUPRC", "AUROC", "precision", "recall", "f1",
        "test_pos", "test_neg", "baseline_prevalence", "thr", "features"
    ]
    sub = df[cols].copy()
    # Format some numbers for readability
    fmt2 = ["AUPRC","AUROC","precision","recall","f1","baseline_prevalence","thr"]
    for c in fmt2:
        sub[c] = sub[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return sub.to_markdown(index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="One or more run directories (or globs).")
    ap.add_argument("--outdir", default="runs/summary", help="Where to write the summary files.")
    args = ap.parse_args()

    # Expand globs
    run_dirs: list[Path] = []
    for pat in args.runs:
        matches = list(Path().glob(pat))
        for m in matches:
            if (m / "metrics.json").exists():
                run_dirs.append(m.resolve())

    if not run_dirs:
        print("No runs with metrics.json found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for rd in sorted(set(run_dirs)):
        rec = load_one(rd)
        if rec:
            rows.append(rec)

    if not rows:
        print("No metrics loaded.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    # Order by AUPRC (desc), then AUROC
    df = df.sort_values(["AUPRC","AUROC"], ascending=[False, False]).reset_index(drop=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "hab_runs_summary.csv"
    md_path  = outdir / "hab_runs_summary.md"

    df.to_csv(csv_path, index=False)
    md_text = to_markdown(df)
    md_path.write_text(md_text + "\n")

    # Pretty console print (short)
    view_cols = ["run","AUPRC","AUROC","precision","recall","f1","test_pos","test_neg","thr"]
    print("\nSummary (sorted by AUPRC):")
    print(df[view_cols].to_string(index=False, formatters={
        "AUPRC": lambda x: f"{x:.3f}",
        "AUROC": lambda x: f"{x:.3f}",
        "precision": lambda x: f"{x:.3f}" if pd.notna(x) else "nan",
        "recall": lambda x: f"{x:.3f}" if pd.notna(x) else "nan",
        "f1": lambda x: f"{x:.3f}" if pd.notna(x) else "nan",
        "thr": lambda x: f"{x:.3f}" if pd.notna(x) else "nan",
    }))

    print(f"\n✓ CSV  -> {csv_path}")
    print(f"✓ MD   -> {md_path}")

if __name__ == "__main__":
    main()
