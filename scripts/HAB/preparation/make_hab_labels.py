#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────────
def season_of_month(m: int) -> str:
    # DJF=win, MAM=spr, JJA=sum, SON=aut
    return ("winter", "spring", "summer", "autumn")[(m % 12) // 3]

def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in df.columns:
        return df
    dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    out = df.copy()
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["season"] = dt.dt.month.apply(season_of_month)
    return out

def chip_centroid_from_bounds(row) -> tuple[float,float]:
    try:
        lon = (float(row["xmin"]) + float(row["xmax"])) * 0.5
        lat = (float(row["ymin"]) + float(row["ymax"])) * 0.5
        return lon, lat
    except Exception:
        return np.nan, np.nan

def maybe_add_centroids(df: pd.DataFrame) -> pd.DataFrame:
    need = not ({"lon","lat"} <= set(df.columns))
    has_bounds = {"xmin","xmax","ymin","ymax"} <= set(df.columns)
    if need and has_bounds:
        lonlat = df.apply(chip_centroid_from_bounds, axis=1, result_type="expand")
        lonlat.columns = ["lon","lat"]
        return pd.concat([df, lonlat], axis=1)
    return df

def pick_numeric(df: pd.DataFrame, names: list[str]) -> dict[str, pd.Series]:
    out = {}
    for n in names:
        if n in df.columns:
            s = pd.to_numeric(df[n], errors="coerce")
            if s.notna().any():
                out[n] = s
    return out

def quantile_or_none(s: pd.Series, q: float) -> float | None:
    s = pd.to_numeric(s, errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return None
    return float(s.quantile(q))

def rank01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    r = s.rank(method="average", pct=True)
    return r.fillna(0.0)

# Core: build HAB labels for one group
def hab_labels_for_group(gdf: pd.DataFrame, init_q: float, q_floor: float,
                         kd_floor: float | None, flh_name: str,
                         return_scores: bool=False) -> pd.DataFrame:
    out = pd.DataFrame(index=gdf.index)
    feats = pick_numeric(gdf, [flh_name, "chlor_a", "kd490"])

    # Optional per-feature scores (0..1 quantile ranks)
    if return_scores:
        if flh_name in feats:  out[f"{flh_name}_score"] = rank01(feats[flh_name])
        if "chlor_a" in feats: out["chlor_a_score"] = rank01(feats["chlor_a"])
        if "kd490"   in feats: out["kd490_score"]   = rank01(feats["kd490"])

    label = pd.Series(0, index=gdf.index, dtype=int)
    made = False
    q = init_q

    while q >= q_floor and not made:
        conds = []
        if flh_name in feats:
            thr = quantile_or_none(feats[flh_name], q)
            if thr is not None:
                conds.append(feats[flh_name] >= thr)
        if "chlor_a" in feats:
            thr = quantile_or_none(feats["chlor_a"], q)
            if thr is not None:
                conds.append(feats["chlor_a"] >= thr)

        if conds:
            pos = np.logical_or.reduce(conds)
            if "kd490" in feats and kd_floor is not None and np.isfinite(kd_floor):
                pos = pos & (feats["kd490"] >= kd_floor)
            if int(pos.sum()) > 0:
                label = pos.astype(int)
                made = True
        q -= 0.02

    if not made:
        # Conservative absolute fallbacks
        conds = []
        if flh_name in feats:
            conds.append(feats[flh_name] >= 0.002)  # nFLH ~ 1e-3..1e-2 W m^-2 um^-1 sr^-1
        if "chlor_a" in feats:
            conds.append(feats["chlor_a"] >= 3.0)   # mg m^-3
        if conds:
            pos = np.logical_or.reduce(conds)
            if "kd490" in feats and kd_floor is not None and np.isfinite(kd_floor):
                pos = pos & (feats["kd490"] >= kd_floor)
            label = pos.astype(int)

    out["hab_label"] = label
    return out

# ────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv")
    ap.add_argument("--q", type=float, default=0.95, help="initial percentile")
    ap.add_argument("--q_floor", type=float, default=0.80, help="lowest percentile to try")
    ap.add_argument("--min_valid", type=int, default=2000, help="min valid_px to keep (0=disable)")
    ap.add_argument("--group", choices=["none","year","season","year_season"],
                    default="season", help="adapt thresholds by group")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON","MINLAT","MAXLON","MAXLAT"),
                    help="optional bbox filter (e.g. 52 12 60 26)")
    ap.add_argument("--kd_floor", type=float, default=0.12,
                    help="Kd_490 lower bound to avoid clear offshore water; use -1 to disable")
    ap.add_argument("--write_scores", action="store_true",
                    help="also write 0..1 quantile scores for flh/chlor_a/kd490")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = args.out_csv or str(in_csv.with_name(in_csv.stem + "_hab.csv"))

    df = pd.read_csv(in_csv)
    df = add_time_fields(df)
    df = maybe_add_centroids(df)

    # keep only strong-water chips if requested
    if "valid_px" in df.columns and args.min_valid > 0:
        df = df[df["valid_px"].fillna(0).astype(float) >= float(args.min_valid)].copy()

    # bbox filter (if lon/lat available)
    if args.bbox and {"lon","lat"} <= set(df.columns):
        mnlon, mnlat, mxlon, mxlat = args.bbox
        df = df[(df["lon"]>=mnlon)&(df["lon"]<=mxlon)&(df["lat"]>=mnlat)&(df["lat"]<=mxlat)].copy()

    # choose FLH column (prefer 'flh', else 'nflh')
    flh_name = "flh" if "flh" in df.columns else ("nflh" if "nflh" in df.columns else None)
    if flh_name is None and "chlor_a" not in df.columns:
        raise SystemExit("No FLH/nFLH or chlor_a columns found — cannot label.")

    kd_floor = None if args.kd_floor is None or float(args.kd_floor) < 0 else float(args.kd_floor)

    # groupwise thresholds
    if args.group == "none":
        add = hab_labels_for_group(df, args.q, args.q_floor, kd_floor, flh_name, args.write_scores)
        out = pd.concat([df.reset_index(drop=True), add.reset_index(drop=True)], axis=1)
    else:
        if args.group == "season":
            gcols = ["season"]
        elif args.group == "year":
            gcols = ["year"]
        else:
            gcols = ["year","season"]

        parts = []
        for _, g in df.groupby(gcols, dropna=False):
            add = hab_labels_for_group(g, args.q, args.q_floor, kd_floor, flh_name, args.write_scores)
            parts.append(pd.concat([g, add], axis=1))
        out = pd.concat(parts, axis=0).sort_index()

    out.to_csv(out_csv, index=False)

    pos = int(out["hab_label"].sum())
    total = int(len(out))
    print(f"✓ Wrote {out_csv}  (positives={pos}, total={total})")

if __name__ == "__main__":
    main()
