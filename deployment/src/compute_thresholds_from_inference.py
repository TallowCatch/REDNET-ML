#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def safe_name(p: Path) -> str:
    return p.stem.replace(".", "_").replace("/", "_")


def pr_points(y: np.ndarray, p: np.ndarray):
    prec, rec, thr = precision_recall_curve(y, p)
    # thr aligns with prec[1:], rec[1:]
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return prec, rec, thr, f1


def best_f1_point(prec, rec, thr, f1) -> Dict:
    # ignore point 0 (no threshold there)
    best_i = int(np.nanargmax(f1[1:])) + 1
    return {
        "threshold": float(thr[best_i - 1]),
        "precision": float(prec[best_i]),
        "recall": float(rec[best_i]),
        "f1": float(f1[best_i]),
    }


def pick_for_precision(prec, rec, thr, f1, target_prec: float) -> Optional[Dict]:
    """
    Pick the threshold that achieves precision >= target_prec with MAX recall.
    """
    ok = np.where(prec[1:] >= target_prec)[0]
    if len(ok) == 0:
        return None
    candidates = ok + 1
    best = candidates[np.nanargmax(rec[candidates])]
    i = int(best)
    return {
        "threshold": float(thr[i - 1]),
        "precision": float(prec[i]),
        "recall": float(rec[i]),
        "f1": float(f1[i]),
    }


def pick_for_recall(prec, rec, thr, f1, target_rec: float) -> Optional[Dict]:
    """
    Pick the threshold that achieves recall >= target_rec with MAX precision.
    """
    ok = np.where(rec[1:] >= target_rec)[0]
    if len(ok) == 0:
        return None
    candidates = ok + 1
    best = candidates[np.nanargmax(prec[candidates])]
    i = int(best)
    return {
        "threshold": float(thr[i - 1]),
        "precision": float(prec[i]),
        "recall": float(rec[i]),
        "f1": float(f1[i]),
    }


def main():
    ap = argparse.ArgumentParser("Compute operating thresholds from saved prediction CSVs")
    ap.add_argument("--pred_glob", required=True, help="glob for prediction csvs (e.g. runs/fusion/.../predictions_cv*.csv)")
    ap.add_argument("--label_col", default="hab_label", help="label column in prediction CSVs")
    ap.add_argument("--prob_col", default="hab_prob", help="probability column in prediction CSVs")

    ap.add_argument("--precision_targets", default="0.90,0.95",
                    help="comma-separated precision targets (e.g. 0.90,0.95)")
    ap.add_argument("--recall_targets", default="",
                    help="comma-separated recall targets (e.g. 0.80,0.90)")

    ap.add_argument("--default_op", default="rec_0_60",
                    help="default op key (e.g. rec_0_60, prec_0_90, best_f1)")
    ap.add_argument("--out_json", required=True, help="output json path (e.g. runs/eval/thresholds_cv.json)")

    args = ap.parse_args()

    paths = sorted([Path(p) for p in __import__("glob").glob(args.pred_glob)])
    if not paths:
        raise SystemExit(f"❌ no files match pred_glob: {args.pred_glob}")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    if args.label_col not in df.columns:
        raise SystemExit(f"❌ label_col '{args.label_col}' not found. columns={list(df.columns)[:60]}")
    if args.prob_col not in df.columns:
        raise SystemExit(f"❌ prob_col '{args.prob_col}' not found. columns={list(df.columns)[:60]}")

    y = pd.to_numeric(df[args.label_col], errors="coerce")
    y = (y > 0.5).astype(int).to_numpy()
    p = pd.to_numeric(df[args.prob_col], errors="coerce").to_numpy()

    m = np.isfinite(p) & np.isfinite(y)
    y = y[m]
    p = p[m]
    if y.sum() == 0:
        raise SystemExit("❌ no positive labels in provided predictions (check label_col).")

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

    # default
    default_op = args.default_op
    if default_op not in ops or ops[default_op] is None:
        # sensible fallback order
        fallback = ["rec_0_60", "prec_0_90", "best_f1"]
        found = None
        for k in fallback:
            if k in ops and ops[k] is not None:
                found = k
                break
        if found is None:
            raise SystemExit("❌ No valid operating point found (check labels/probs).")
        default_op = found

    out = {
        "schema_version": "1.0",
        "positive_class": 1,
        "default_operating_point": default_op,
        "default_threshold": float(ops[default_op]["threshold"]),
        "operating_points": ops,
        "notes": f"Computed from prediction CSVs: {args.pred_glob}",
        "n": int(len(y)),
        "pos": int(y.sum()),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"✅ wrote: {out_path}")
    print("default_operating_point:", out["default_operating_point"])
    print("default_threshold:", out["default_threshold"])
    print("best_f1 threshold:", ops["best_f1"]["threshold"])
    if "rec_0_60" in ops and ops["rec_0_60"] is not None:
        print("rec_0_60 threshold:", ops["rec_0_60"]["threshold"])


if __name__ == "__main__":
    main()
