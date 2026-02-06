#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def pr_points(y: np.ndarray, p: np.ndarray):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return prec, rec, thr, f1


def best_f1_point(prec, rec, thr, f1) -> Dict:
    i = int(np.nanargmax(f1[1:])) + 1
    return {
        "threshold": float(thr[i - 1]),
        "precision": float(prec[i]),
        "recall": float(rec[i]),
        "f1": float(f1[i]),
    }


def pick_for_precision(prec, rec, thr, f1, target_prec: float) -> Optional[Dict]:
    idx = np.where(prec[1:] >= target_prec)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0] + 1)
    return {
        "threshold": float(thr[i - 1]),
        "precision": float(prec[i]),
        "recall": float(rec[i]),
        "f1": float(f1[i]),
    }


def pick_for_recall(prec, rec, thr, f1, target_rec: float) -> Optional[Dict]:
    idx = np.where(rec[1:] >= target_rec)[0]
    if len(idx) == 0:
        return None
    candidates = idx + 1
    best = candidates[np.argmax(prec[candidates])]
    i = int(best)
    return {
        "threshold": float(thr[i - 1]),
        "precision": float(prec[i]),
        "recall": float(rec[i]),
        "f1": float(f1[i]),
    }


def main():
    ap = argparse.ArgumentParser("Compute operating thresholds from CV prediction CSVs")
    ap.add_argument("--pred_glob", required=True, help='glob for predictions, e.g. "runs/.../predictions_cv*.csv"')
    ap.add_argument("--label_col", required=True, help="e.g. hab_label_final2")
    ap.add_argument("--prob_col", required=True, help="e.g. p_fused")
    ap.add_argument("--precision_targets", default="0.90,0.95")
    ap.add_argument("--recall_targets", default="0.60")
    ap.add_argument("--default_op", default="rec_0_60", help="e.g. rec_0_60 or prec_0_90 or best_f1")
    ap.add_argument("--out_json", required=True)

    args = ap.parse_args()

    paths = sorted(Path(".").glob(args.pred_glob))
    if not paths:
        raise SystemExit(f"❌ pred_glob matched nothing: {args.pred_glob}")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    if args.label_col not in df.columns:
        raise SystemExit(f"❌ label_col '{args.label_col}' not found. columns={list(df.columns)}")
    if args.prob_col not in df.columns:
        raise SystemExit(f"❌ prob_col '{args.prob_col}' not found. columns={list(df.columns)}")

    y = pd.to_numeric(df[args.label_col], errors="coerce")
    y = (y > 0.5).astype(int).to_numpy()

    p = pd.to_numeric(df[args.prob_col], errors="coerce").to_numpy()

    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = p[mask]

    if y.size < 50:
        raise SystemExit(f"❌ too few rows after filtering: {y.size}")

    prec, rec, thr, f1 = pr_points(y, p)

    ops: Dict[str, Optional[Dict]] = {}
    ops["best_f1"] = best_f1_point(prec, rec, thr, f1)

    prec_targets = [float(x) for x in args.precision_targets.split(",") if x.strip()]
    for t in prec_targets:
        key = f"prec_{str(t).replace('.', '_')}"
        ops[key] = pick_for_precision(prec, rec, thr, f1, t)

    rec_targets = [float(x) for x in args.recall_targets.split(",") if x.strip()]
    for t in rec_targets:
        key = f"rec_{str(t).replace('.', '_')}"
        ops[key] = pick_for_recall(prec, rec, thr, f1, t)

    # choose default
    default_op = args.default_op
    if default_op not in ops or ops[default_op] is None:
        fallback = ["rec_0_60", "prec_0_90", "prec_0_95", "best_f1"]
        for k in fallback:
            if k in ops and ops[k] is not None:
                default_op = k
                break
        else:
            raise SystemExit("❌ No valid operating point found (check labels/probs).")

    out = {
        "schema_version": "1.0",
        "default_operating_point": default_op,
        "default_threshold": float(ops[default_op]["threshold"]),
        "operating_points": ops,
        "n_rows": int(y.size),
        "label_col": args.label_col,
        "prob_col": args.prob_col,
        "source_glob": args.pred_glob,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"✅ wrote: {out_path}")
    print("default_operating_point:", out["default_operating_point"])
    print("default_threshold:", out["default_threshold"])
    print("best_f1 threshold:", ops["best_f1"]["threshold"])


if __name__ == "__main__":
    main()
