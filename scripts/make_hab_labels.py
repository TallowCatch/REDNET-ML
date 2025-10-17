#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
from pathlib import Path

def pick_numeric(df, names):
    out = {}
    for n in names:
        if n in df.columns:
            s = pd.to_numeric(df[n], errors="coerce")
            if s.notna().any():
                out[n] = s
    return out

def season_of_month(m):
    # DJF=win, MAM=spr, JJA=sum, SON=aut
    return ("winter","spring","summer","autumn")[(m%12)//3]

def add_time_fields(df):
    if "datetime" not in df.columns: return df
    dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.copy()
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["season"] = dt.dt.month.apply(season_of_month)
    return df

def chip_centroid_from_bounds(row):
    try:
        return (float(row["xmin"])+float(row["xmax"])) * 0.5, (float(row["ymin"])+float(row["ymax"])) * 0.5
    except Exception:
        return (np.nan, np.nan)

def maybe_add_centroids(df):
    need = not ({"lon","lat"} <= set(df.columns))
    has_bounds = {"xmin","xmax","ymin","ymax"} <= set(df.columns)
    if need and has_bounds:
        lonlat = df.apply(chip_centroid_from_bounds, axis=1, result_type="expand")
        lonlat.columns = ["lon","lat"]
        return pd.concat([df, lonlat], axis=1)
    return df

def hab_labels_for_group(gdf, q, q_floor, kd_floor, flh_name):
    feats = pick_numeric(gdf, [flh_name, "chlor_a", "kd490"])
    label = pd.Series(0, index=gdf.index, dtype=int)

    # Percentile sweep (tight → relaxed)
    made = False
    qq = q
    while qq >= q_floor and not made:
        conds = []
        if flh_name in feats:
            flh_thr = float(feats[flh_name].quantile(qq))
            conds.append(feats[flh_name] >= flh_thr)
        if "chlor_a" in feats:
            chl_thr = float(feats["chlor_a"].quantile(qq))
            conds.append(feats["chlor_a"] >= chl_thr)
        if conds:
            pos = np.logical_or.reduce(conds)
            # optionally require some turbidity if kd present
            if "kd490" in feats and kd_floor is not None:
                pos = pos & (feats["kd490"] >= kd_floor)
            if int(pos.sum()) > 0:
                label = pos.astype(int)
                made = True
                break
        qq -= 0.02

    if not made:
        # Absolute fallbacks tuned for coastal Arabian Sea (conservative)
        conds = []
        if flh_name in feats:  conds.append(feats[flh_name] >= 0.002)   # nFLH ~ O(1e-3 to 1e-2) for blooms
        if "chlor_a" in feats: conds.append(feats["chlor_a"] >= 3.0)   # mg m^-3
        if conds:
            pos = np.logical_or.reduce(conds)
            if "kd490" in feats and kd_floor is not None:
                pos = pos & (feats["kd490"] >= kd_floor)
            label = pos.astype(int)

    return label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv")
    ap.add_argument("--q", type=float, default=0.95, help="initial percentile")
    ap.add_argument("--q_floor", type=float, default=0.80, help="lowest percentile to try")
    ap.add_argument("--min_valid", type=int, default=2000, help="min valid_px to keep (0=ignore)")
    ap.add_argument("--group", choices=["none","year","season","year_season"], default="season",
                    help="adapt thresholds by group")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON","MINLAT","MAXLON","MAXLAT"),
                    help="optional bbox filter (e.g. 52 12 60 26)")
    ap.add_argument("--kd_floor", type=float, default=0.12,
                    help="Kd_490 lower bound to avoid clear offshore water (None disables)")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = args.out_csv or str(in_csv.with_name(in_csv.stem + "_hab.csv"))

    df = pd.read_csv(in_csv)
    df = add_time_fields(df)
    df = maybe_add_centroids(df)

    # keep only strong-water chips if requested
    if "valid_px" in df.columns and args.min_valid > 0:
        df = df[df["valid_px"] >= args.min_valid].copy()

    # bbox filter for Oman coast if asked (roughly 52–60E, 12–26N)
    if args.bbox and {"lon","lat"} <= set(df.columns):
        mnlon, mnlat, mxlon, mxlat = args.bbox
        df = df[(df["lon"]>=mnlon)&(df["lon"]<=mxlon)&(df["lat"]>=mnlat)&(df["lat"]<=mxlat)].copy()

    # pick FLH column name present
    flh_name = "flh" if "flh" in df.columns else ("nflh" if "nflh" in df.columns else None)
    if flh_name is None and "chlor_a" not in df.columns:
        raise SystemExit("No FLH/nFLH or chlor_a columns found — cannot label.")

    if args.group == "none":
        df["hab_label"] = hab_labels_for_group(df, args.q, args.q_floor, args.kd_floor, flh_name)
    else:
        if args.group == "season":
            groups = ["season"]
        elif args.group == "year":
            groups = ["year"]
        else:  # year_season
            groups = ["year","season"]
        labels = []
        for keys, g in df.groupby(groups):
            lab = hab_labels_for_group(g, args.q, args.q_floor, args.kd_floor, flh_name)
            labels.append(lab)
        df["hab_label"] = pd.concat(labels).sort_index()

    df.to_csv(out_csv, index=False)
    pos = int(df["hab_label"].sum())
    print(f"✓ Wrote {out_csv}  (positives={pos}, total={len(df)})")

if __name__ == "__main__":
    main()
