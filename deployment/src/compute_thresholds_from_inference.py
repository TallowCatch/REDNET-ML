# deployment/src/compute_thresholds_from_inference_csv.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from inference import run_inference


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def safe_name(p: Path) -> str:
    return p.stem.replace(".", "_").replace("/", "_")


def pr_points(y: np.ndarray, p: np.ndarray):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # thr aligns with prec[1:], rec[1:]
    return prec, rec, thr, f1


def best_f1_point(prec, rec, thr, f1) -> Dict:
    best_i = int(np.nanargmax(f1[1:])) + 1
    return {
        "threshold": float(thr[best_i - 1]),
        "precision": float(prec[best_i]),
        "recall": float(rec[best_i]),
        "f1": float(f1[best_i]),
    }


def pick_for_precision(prec, rec, thr, f1, target_prec: float) -> Optional[Dict]:
    idx = np.where(prec[1:] >= target_prec)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0] + 1)  # earliest threshold meeting precision
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
    best = candidates[np.argmax(prec[candidates])]  # best precision among those meeting recall
    i = int(best)
    return {
        "threshold": float(thr[i - 1]),
        "precision": float(prec[i]),
        "recall": float(rec[i]),
        "f1": float(f1[i]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--eval_csv", required=True, help="CSV containing raw inputs + label")
    ap.add_argument("--label_col", default="hab_label")

    ap.add_argument("--precision_targets", default="0.90,0.95",
                    help="comma-separated precision targets (e.g. 0.90,0.95)")
    ap.add_argument("--recall_targets", default="",
                    help="comma-separated recall targets (e.g. 0.80,0.90)")

    ap.add_argument("--default_op", default="prec_0_90",
                    help="which operating point becomes default_threshold (e.g. prec_0_90, best_f1)")
    ap.add_argument("--out_json", default=None,
                    help="Output path. If omitted, writes versioned file next to artifacts as thresholds_<evalstem>.json")
    ap.add_argument("--write_as_active", action="store_true",
                    help="If set, ALSO copy result to deployment/artifacts/thresholds.json")

    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    eval_csv = Path(args.eval_csv)

    df = pd.read_csv(eval_csv)
    if args.label_col not in df.columns:
        raise SystemExit(f"❌ label_col '{args.label_col}' not found in eval_csv")

    y = df[args.label_col].astype(int).values

    # run real deployment inference (uses ensemble + calibrator if present)
    pred_df = run_inference(df, artifacts_dir=artifacts_dir)
    p = pred_df["prob"].astype(float).values

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

    # choose default op
    default_op = args.default_op
    if default_op not in ops or ops[default_op] is None:
        # fallback order
        fallback = ["prec_0_90", "prec_0_95", "best_f1"]
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
        "notes": f"Computed using deployment inference probs on {eval_csv.as_posix()}",
    }

    if args.out_json:
        out_path = Path(args.out_json)
    else:
        out_path = artifacts_dir / f"thresholds_{safe_name(eval_csv)}.json"

    out_path.write_text(json.dumps(out, indent=2))

    # optionally activate
    if args.write_as_active:
        active = artifacts_dir / "thresholds.json"
        active.write_text(json.dumps(out, indent=2))
        print(f"✅ wrote active: {active}")

    print(f"✅ wrote: {out_path}")
    print("default_operating_point:", out["default_operating_point"])
    print("default_threshold:", out["default_threshold"])
    print("best_f1 threshold:", ops["best_f1"]["threshold"])


if __name__ == "__main__":
    main()
