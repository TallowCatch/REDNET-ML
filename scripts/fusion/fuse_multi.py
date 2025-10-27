#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import re
from typing import Dict, Tuple, List

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
_TILE_RE = re.compile(r"_(\d{4})$")  # after removing extension

def _clean_columns(df: pd.DataFrame, src: str = "") -> pd.DataFrame:
    df = df.copy()
    cols = [str(c).strip() for c in df.columns]
    dup_mask = pd.Index(cols).duplicated(keep="first")
    if dup_mask.any():
        dropped = [c for i, c in enumerate(cols) if dup_mask[i]]
        print(f"[warn] {src} had duplicated columns {dropped}; keeping first occurrence.")
        df = df.loc[:, ~dup_mask]
        cols = [c for i, c in enumerate(cols) if not dup_mask[i]]
    df.columns = cols
    bad = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if bad:
        df = df.drop(columns=bad)
    return df

def _normalize_filename_series(s: pd.Series) -> pd.Series:
    return s.astype(str).map(lambda x: Path(x).name)

def _to_scene_from_name(name: str) -> str:
    stem = Path(str(name)).stem
    return _TILE_RE.sub("", stem)

def _guess_score_col(df: pd.DataFrame, id_col: str) -> str:
    numeric = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
    cand = [c for c in numeric if str(c).lower().startswith("p_") or "score" in str(c).lower()]
    cand = [c for c in cand if not str(c).lower().endswith("count")]
    if cand:
        return cand[0]
    numeric = [c for c in numeric if not str(c).lower().endswith("count")]
    if numeric:
        return numeric[0]
    for c in df.columns:
        if c != id_col:
            return c
    raise SystemExit("Could not guess score column.")

def _load_coco_map(coco_json: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    if not coco_json:
        return None, None
    with open(coco_json, "r") as f:
        coco = json.load(f)
    id2name = {int(im["id"]): str(im["file_name"]) for im in coco["images"]}
    name2id = {v: k for k, v in id2name.items()}
    return id2name, name2id

def _maybe_map_detector_id_to_filename(df: pd.DataFrame, id_col: str, id2name, src: str) -> pd.Series:
    """
    Return a Series 'file' with filename-like IDs for the detector df.
    If id2name is provided and ids look numeric -> map chip_id → filename via COCO.
    Else, try image/tile/chip_id columns.
    """
    if id_col in df.columns:
        ser = df[id_col]
    elif "chip_id" in df.columns:
        ser = df["chip_id"]
    elif "tile" in df.columns:
        ser = df["tile"]
    elif "image" in df.columns:
        ser = df["image"]
    else:
        # fall back to first column
        ser = df.iloc[:, 0]

    ser = ser.copy()

    # If numeric-like and COCO map exists, map → filenames
    numeric_like = pd.to_numeric(ser, errors="coerce")
    if id2name is not None and numeric_like.notna().mean() >= 0.9:
        keys = numeric_like.astype("Int64")
        mapped = keys.map(lambda k: id2name.get(int(k)) if pd.notna(k) and int(k) in id2name else None)
        n = mapped.notna().sum()
        print(f"[info] {src}: COCO mapped {n}/{len(mapped)} ids to filenames.")
        ser = mapped.fillna("")

    ser = _normalize_filename_series(ser)
    return ser

def _merge_detectors_into_base(
    base: pd.DataFrame,
    base_id_col: str,
    base_scene_col: str,
    det_name_to_path: Dict[str, str],
    id2name,
    agg: str = "max",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    For each detector CSV:
      1) read + clean
      2) produce 'file' and 'scene_id' for detector rows
      3) choose score column
      4) try filename join; if low coverage, do scene join (with aggregation)
      5) print coverage
    Returns updated base and list of detector column names added.
    """
    det_feature_names = []
    for name, path in det_name_to_path.items():
        print(f"\n[det] Loading detector '{name}' from {path}")
        df = pd.read_csv(path)
        df = _clean_columns(df, path)

        # build detector 'file' & 'scene'
        det_file = _maybe_map_detector_id_to_filename(df, base_id_col, id2name, path)
        df = df.assign(__det_file__=det_file, __det_scene__=det_file.map(_to_scene_from_name))

        # choose score column (p_* or *score*)
        score_col = _guess_score_col(df, "__det_file__")
        if score_col in ("chip_id", "tile", "image", base_id_col):
            # edge case: guessed id col by mistake; try another numeric
            others = [c for c in df.columns if c not in ("chip_id","tile","image",base_id_col,"__det_file__","__det_scene__")]
            numeric = [c for c in others if pd.api.types.is_numeric_dtype(df[c]) and not str(c).lower().endswith("count")]
            if numeric:
                score_col = numeric[0]
            else:
                raise SystemExit(f"[fatal] could not find a numeric score column in {path}")
        print(f"[det] Using score column: {score_col}")

        # 1) filename merge
        tmp = base.merge(
            df[[ "__det_file__", score_col ]].rename(columns={"__det_file__": base_id_col, score_col: name}),
            on=base_id_col, how="left"
        )
        cov1 = float((~tmp[name].isna()).mean())
        print(f"[det] Filename join coverage: {cov1*100:.1f}%")

        if cov1 < 0.10:
            # 2) scene merge with aggregation
            print(f"[det] Low coverage via filename; trying scene-level ({agg}) aggregation on '{base_scene_col}'...")
            agg_fn = {"max": "max", "mean": "mean", "median": "median"}[agg]
            df_scene = (
                df.groupby("__det_scene__")[score_col]
                  .agg(agg_fn)
                  .reset_index()
                  .rename(columns={"__det_scene__": base_scene_col, score_col: name})
            )
            tmp2 = base.merge(df_scene, on=base_scene_col, how="left")
            cov2 = float((~tmp2[name].isna()).mean())
            print(f"[det] Scene join coverage: {cov2*100:.1f}%  (filename={cov1*100:.1f}%)")
            if cov2 >= cov1:
                base = tmp2
            else:
                base = tmp
        else:
            base = tmp

        if name not in base.columns:
            base[name] = np.nan  # ensure presence
        det_feature_names.append(name)

        # print missing examples for debugging
        missing = base.loc[base[name].isna(), [base_id_col, base_scene_col]].head(5)
        if not missing.empty:
            print(f"[det] examples still missing after merge (showing up to 5):")
            print(missing.to_string(index=False))

    return base, det_feature_names

def _ensure_scene_column(df: pd.DataFrame, id_col: str, scene_col: str) -> pd.DataFrame:
    df = df.copy()
    if scene_col not in df.columns:
        print(f"[info] deriving '{scene_col}' from '{id_col}'...")
        df[scene_col] = _normalize_filename_series(df[id_col]).map(_to_scene_from_name)
    return df

# ---------------- run one dataset ----------------
def run_one_set(
    name: str,
    tab_csv: str,
    id_col: str,
    scene_col: str,
    det_map: Dict[str,str],
    outdir_root: Path,
    id2name,
    agg: str,
    seed: int,
    min_pos_per_split: int,
    test_size: float,
    max_tries: int,
):
    print(f"\n========== DATASET {name} ==========")
    outdir = outdir_root / name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[info] outputs → {outdir.resolve()}")

    # --- load base/tabular ---
    base = pd.read_csv(tab_csv)
    base = _clean_columns(base, tab_csv)

    # coerce id col
    if id_col not in base.columns:
        if "tile" in base.columns:
            base = base.rename(columns={"tile": id_col})
        elif "image" in base.columns:
            base[id_col] = _normalize_filename_series(base["image"])
        else:
            raise SystemExit(f"{tab_csv} missing id column '{id_col}'. Columns: {list(base.columns)}")
    base[id_col] = _normalize_filename_series(base[id_col])

    # ensure scene + required cols
    base = _ensure_scene_column(base, id_col=id_col, scene_col=scene_col)

    need = {id_col, scene_col, "hab_label", "p_tab"}
    missing = need - set(base.columns)
    if missing:
        raise SystemExit(f"{tab_csv} missing columns: {sorted(missing)}")

    base = base[[id_col, scene_col, "hab_label", "p_tab"]].copy()
    print(f"[base] rows={len(base)}, positives={int(base['hab_label'].sum())}, "
          f"unique scenes={base[scene_col].nunique()}")

    # --- detectors ---
    base, det_feats = _merge_detectors_into_base(
        base, id_col, scene_col, det_map, id2name, agg=agg
    )

    feats = ["p_tab"] + det_feats
    base[feats] = base[feats].fillna(0.0)

    # dump merged debug
    merged_csv = outdir / "merged_features_debug.csv"
    base.to_csv(merged_csv, index=False)
    print(f"[debug] wrote merged features: {merged_csv}")

    # --- matrices ---
    X = base[feats].values
    y = base["hab_label"].astype(int).values
    G = base[scene_col].astype(str).values

    # --- group-aware split with positives on both sides ---
    rng = np.random.RandomState(seed)
    good = None; tries = 0
    for t in range(1, max_tries + 1):
        gss = GroupShuffleSplit(test_size=test_size, random_state=int(rng.randint(0, 10_000)))
        tr, te = next(gss.split(X, y, groups=G))
        if y[tr].sum() >= min_pos_per_split and y[te].sum() >= min_pos_per_split:
            good = (tr, te); tries = t; break
    if good is None:
        raise SystemExit("Could not find a split with positives in both partitions; lower --min_pos_per_split.")
    tr, te = good
    print(f"[split] train={len(tr)} (pos={int(y[tr].sum())}) | test={len(te)} (pos={int(y[te].sum())}) | tries={tries}")

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
    print(f"[metrics] AUPRC={auprc:.3f}  AUROC={auroc:.3f}  thr*={thr_star:.3f}  (baseline={base_rate:.3f})")
    print("[metrics] Confusion matrix [[TN,FP],[FN,TP]]:", cm)
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
        "dataset": name,
        "model": "fusion(logreg)",
        "feats": "+".join(feats),
        "auprc": auprc, "auroc": auroc,
        "thr_star": thr_star,
        "test_pos": int(y[te].sum()),
        "test_total": int(len(te))
    }]).to_csv(outdir / "summary.csv", index=False)

    print(f"[save] {outdir/'fusion_model.joblib'}")
    print(f"[save] {outdir/'metrics.json'}")
    print(f"[save] {outdir/'pr_fusion.png'}")
    print(f"[save] {outdir/'roc_fusion.png'}")
    print(f"[save] {outdir/'summary.csv'}")

# ---------------- main --------------------
def parse_name_equals_path_list(items: List[str]) -> Dict[str,str]:
    out = {}
    for spec in items:
        if "=" not in spec:
            raise SystemExit("Use NAME=path.csv (e.g., B=runs/fusion/tab_B.csv)")
        k, v = spec.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def main():
    ap = argparse.ArgumentParser(description="Multi-dataset decision fusion with robust detector joins + diagnostics.")
    ap.add_argument("--tabular", nargs="+", required=True,
                    help="dataset name=path.csv (e.g., B=runs/fusion/tab_B.csv C=runs/fusion/tab_C.csv)")
    ap.add_argument("--det", nargs="+", required=True,
                    help="detector name=path.csv (e.g., frcnn_r50=runs/fusion/p_frcnn_r50.csv ...)")
    ap.add_argument("--id_col", default="tile", help="column with filename (tile) in tabular CSVs")
    ap.add_argument("--scene_col", default="scene_id", help="scene/group column name")
    ap.add_argument("--outdir", default="runs/fusion/decision_fusion")
    ap.add_argument("--coco_json", default="", help="COCO JSON to map numeric chip_id → file_name for detector CSVs")
    ap.add_argument("--det_agg", default="max", choices=["max","mean","median"],
                    help="aggregation to use when doing scene-level detector merge")
    ap.add_argument("--min_pos_per_split", type=int, default=2)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--max_tries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir_root = Path(args.outdir); outdir_root.mkdir(parents=True, exist_ok=True)
    tabs = parse_name_equals_path_list(args.tabular)
    dets = parse_name_equals_path_list(args.det)

    print("[info] tabular sets:", tabs)
    print("[info] detectors:", dets)
    print("[info] outdir root:", outdir_root.resolve())

    id2name, _ = _load_coco_map(args.coco_json)
    if id2name is None:
        print("[info] no COCO map supplied; will not translate numeric chip_id → filenames.")

    for ds_name, ds_path in tabs.items():
        run_one_set(
            name=ds_name,
            tab_csv=ds_path,
            id_col=args.id_col,
            scene_col=args.scene_col,
            det_map=dets,
            outdir_root=outdir_root,
            id2name=id2name,
            agg=args.det_agg,
            seed=args.seed,
            min_pos_per_split=args.min_pos_per_split,
            test_size=args.test_size,
            max_tries=args.max_tries,
        )

if __name__ == "__main__":
    main()
