#!/usr/bin/env python3
import argparse, re, os, shutil
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np

SCENE_RE = re.compile(r'(S2[AB]_MSIL2A_[0-9T_]+)')

def to_scene_base(s:str) -> str|None:
    if not isinstance(s, str): return None
    m = SCENE_RE.search(s)
    if m: return m.group(1)
    b = Path(str(s)).name
    b = re.sub(r'(_\d{4})?\.(jpg|png)$','', b, flags=re.I)
    if b.startswith("S2") and "_MSIL2A_" in b: return b
    return None

POSS_SCENE_COLS = ("scene_id","filename","file","image","img","coco_file","coco_path","img_path","tile")

def best_scene(row) -> str|None:
    for c in POSS_SCENE_COLS:
        if c in row and pd.notna(row[c]):
            base = to_scene_base(str(row[c]))
            if base: return base
    return None

def index_jpgs(roots):
    """Map base -> [fullpaths], searching recursively."""
    files = []
    for r in roots:
        r = str(r)
        files += glob(f"{r}/**/*.jpg", recursive=True)
    by_base = {}
    for p in files:
        stem = Path(p).stem
        base = re.sub(r'_(0000|0001)$', '', stem)
        by_base.setdefault(base, []).append(p)
    return by_base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_dir", default="runs/fusion/fused_sets/B_mined_timecv_norm_f1")
    # recursive roots: include data + qc in case you’ve already copied some
    ap.add_argument("--jpg_roots", nargs="+",
        default=["data", "qc", "runs"])
    ap.add_argument("--out_dir", default="qc/hab_hits_inspect")
    ap.add_argument("--k", type=int, default=120)
    ap.add_argument("--link_mode", choices=["copy","symlink"], default="copy")
    ap.add_argument("--score_col", default=None)
    ap.add_argument("--section", choices=["pred_pos","tp","fp"], default="pred_pos")
    args = ap.parse_args()

    fdir = Path(args.fusion_dir)
    merged = fdir / "merged_features_debug.csv"
    preds  = fdir / "predictions_cv2.csv"
    if merged.exists():
        df = pd.read_csv(merged)
    elif preds.exists():
        df = pd.read_csv(preds)
    else:
        raise SystemExit(f"No predictions file found in {fdir}")

    # choose score col
    if args.score_col and args.score_col in df.columns:
        sc = args.score_col
    else:
        prefer = ["p_fused","score","y_pred_score","p_tab"]
        sc = next((c for c in prefer if c in df.columns), None)
        if sc is None:
            dets = [c for c in df.columns if c.startswith("frcnn_")]
            sc = dets[0] if dets else None
        if sc is None:
            raise SystemExit("No score column found (tried p_fused/score/y_pred_score/p_tab or frcnn_*).")

    # threshold + preds
    thr_file = fdir / "threshold.txt"
    thr = float(thr_file.read_text().strip()) if thr_file.exists() else float(np.nanmedian(df[sc]))
    ycol = "hab_label" if "hab_label" in df.columns else ("y_true" if "y_true" in df.columns else None)
    if ycol is None:
        raise SystemExit("Need a label column (hab_label or y_true).")
    df["pred"] = (df[sc] >= thr).astype(int)

    # subset
    if args.section == "tp":
        sub = df[(df[ycol]==1) & (df["pred"]==1)].sort_values(sc, ascending=False)
    elif args.section == "fp":
        sub = df[(df[ycol]==0) & (df["pred"]==1)].sort_values(sc, ascending=False)
    else:
        sub = df[df["pred"]==1].sort_values(sc, ascending=False)  # predicted positives

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # keep qc_all; clean only jpgs at top level
    for p in out.glob("*.jpg"):
        try: p.unlink()
        except: pass

    by_base = index_jpgs(args.jpg_roots)

    rows, taken = [], 0
    for _,r in sub.iterrows():
        scene = best_scene(r)
        if not scene: continue
        imgs = sorted(by_base.get(scene, []))
        if not imgs:
            # fuzzy startswith
            for base, plist in by_base.items():
                if base.startswith(scene):
                    imgs = sorted(plist); break
        if not imgs: 
            continue

        used = []
        for j in imgs[:2]:
            dst = out / Path(j).name
            if not dst.exists():
                try:
                    if args.link_mode == "symlink":
                        try: dst.symlink_to(Path(j).resolve())
                        except FileExistsError: pass
                    else:
                        shutil.copy2(j, dst)
                except Exception:
                    continue
            used.append(dst.name)

        rows.append({
            "scene_id": r.get("scene_id", scene),
            "tile": r.get("tile",""),
            "score": float(r.get(sc, np.nan)),
            "thr": float(thr),
            "pred": int(r.get("pred",0)),
            "ytrue": int(r.get(ycol, -1)) if ycol else -1,
            "jpg_0": used[0] if len(used)>0 else "",
            "jpg_1": used[1] if len(used)>1 else "",
        })
        taken += 1
        if taken >= args.k: break

    pd.DataFrame(rows).to_csv(out/"manifest.csv", index=False)
    print(f"✓ repopulated {out} | subset={args.section} | k={len(rows)}")
    print(f"  score={sc} thr={thr:.3f}")
    print(f"  searched recursively under: {args.jpg_roots}")

if __name__ == "__main__":
    main()
