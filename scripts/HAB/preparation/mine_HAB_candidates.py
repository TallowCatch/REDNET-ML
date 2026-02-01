#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(20\d{2})[-_.]?(\d{2})")

def derive_month_key(row, group_col="scene_id"):
    # 1) try datetime
    if "datetime" in row and pd.notna(row["datetime"]):
        try:
            dt = pd.to_datetime(row["datetime"], errors="coerce", utc=True)
            if pd.notna(dt):
                dt = dt.tz_convert(None) if hasattr(dt, "tz_localize") else dt
                return f"{dt.year:04d}-{dt.month:02d}"
        except Exception:
            pass
    # 2) try scene_id
    m = MONTH_RE.search(str(row.get(group_col, "")))
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12: return f"{y:04d}-{mo:02d}"
    return None

def robust_scale_by_month(df, col, month_col):
    g = df.groupby(month_col)[col]
    med = g.median()
    mad = g.apply(lambda s: np.median(np.abs(s - np.median(s))) if len(s) else np.nan)
    # 1.4826*MAD ~ robust std
    z = (df[col] - df[month_col].map(med)) / (1e-9 + 1.4826 * df[month_col].map(mad))
    return z

def month_percentile(df, col, month_col, q=0.95):
    return df.groupby(month_col)[col].quantile(q)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True,
                    help="Your merged chip_indices_clean_hab.csv after append_modis")
    ap.add_argument("--out_train_csv", required=True,
                    help="Output labeled CSV for training (with hab_label_final)")
    ap.add_argument("--out_candidates_csv", required=True,
                    help="List of mined positives to review")
    ap.add_argument("--group_by", default="scene_id")
    ap.add_argument("--id_col", default="tile")
    ap.add_argument("--drop_multimonth_scenes", action="store_true",
                    help="If set, drop any scene_id that appears in >1 month (stricter non-leak)")
    ap.add_argument("--promote_heuristic", action="store_true",
                    help="If set, promote heuristic positives to label=1")
    ap.add_argument("--nflh_z", type=float, default=2.0)
    ap.add_argument("--chl_z", type=float, default=2.0)
    ap.add_argument("--nflh_q", type=float, default=0.95)
    ap.add_argument("--chl_q", type=float, default=0.95)
    ap.add_argument("--kd490_min", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv).copy()

    # basic numeric coercion
    for c in ["hab_label", "nflh", "flh", "chlor_a", "kd490"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # unify nflh/flh into 'nflh' if needed
    if "nflh" not in df.columns and "flh" in df.columns:
        df = df.rename(columns={"flh": "nflh"})

    # month_key
    df["month_key"] = df.apply(lambda r: derive_month_key(r, args.group_by), axis=1)

    # drop rows with no month
    df = df[df["month_key"].notna()].copy()

    # (optional) strict non-leak: drop scenes that span multiple months
    if args.drop_multimonth_scenes:
        cnt = df.groupby(args.group_by)["month_key"].nunique()
        bad_scenes = set(cnt[cnt > 1].index)
        before = len(df)
        df = df[~df[args.group_by].isin(bad_scenes)].copy()
        print(f"[nonleak] dropped {before-len(df)} rows from multi-month scenes ({len(bad_scenes)} scenes)")

    # robust z-scores by month
    if "nflh" in df.columns:
        df["nflh_z"] = robust_scale_by_month(df.fillna(0), "nflh", "month_key")
        p_nflh = month_percentile(df.fillna(0), "nflh", "month_key", q=args.nflh_q)
        df["nflh_p95"] = df["month_key"].map(p_nflh)
    else:
        df["nflh_z"] = np.nan; df["nflh_p95"] = np.nan

    if "chlor_a" in df.columns:
        df["chl_z"] = robust_scale_by_month(df.fillna(0), "chlor_a", "month_key")
        p_chl = month_percentile(df.fillna(0), "chlor_a", "month_key", q=args.chl_q)
        df["chl_p95"] = df["month_key"].map(p_chl)
    else:
        df["chl_z"] = np.nan; df["chl_p95"] = np.nan

    # heuristic positive rules
    conds = []
    if "nflh" in df.columns:
        conds.append((df["nflh_z"] >= args.nflh_z) | (df["nflh"] >= df["nflh_p95"]))
    if "chlor_a" in df.columns:
        conds.append((df["chl_z"] >= args.chl_z) | (df["chlor_a"] >= df["chl_p95"]))
    if "kd490" in df.columns:
        conds.append(df["kd490"] >= args.kd490_min)

    if conds:
        heuristic_pos = conds[0]
        for c in conds[1:]:
            heuristic_pos = heuristic_pos | c
    else:
        heuristic_pos = pd.Series(False, index=df.index)

    df["hab_label_heuristic"] = heuristic_pos.astype(int)

    # final label
    if "hab_label" in df.columns:
        base = (pd.to_numeric(df["hab_label"], errors="coerce") > 0.5).astype(int)
    else:
        base = pd.Series(0, index=df.index)

    df["hab_label_final"] = base
    if args.promote_heuristic:
        df.loc[df["hab_label_heuristic"] == 1, "hab_label_final"] = 1

    # outputs
    out_train_cols = [args.id_col, args.group_by, "datetime", "month_key",
                      "fai_mean","rednir_mean","ndwi_mean","kd490","chlor_a","nflh", "sst",
                      "month_sin","month_cos","ndwi_std","rednir_std",
                      "hab_label","hab_label_heuristic","hab_label_final"]
    out_train_cols = [c for c in out_train_cols if c in df.columns]

    Path(args.out_train_csv).parent.mkdir(parents=True, exist_ok=True)
    df[out_train_cols].to_csv(args.out_train_csv, index=False)

    cands = df.loc[(df["hab_label"] != 1) & (df["hab_label_heuristic"] == 1)].copy()
    cands_cols = [args.id_col, args.group_by, "datetime", "month_key",
                  "kd490","chlor_a","nflh","nflh_z","chl_z","hab_label","hab_label_heuristic"]
    cands_cols = [c for c in cands_cols if c in cands.columns]
    cands[cands_cols].to_csv(args.out_candidates_csv, index=False)

    print(f"✓ wrote mined train CSV: {args.out_train_csv} (rows={len(df)}, positives_final={int(df['hab_label_final'].sum())})")
    print(f"✓ wrote candidates for review: {args.out_candidates_csv} (rows={len(cands)})")

if __name__ == "__main__":
    main()
