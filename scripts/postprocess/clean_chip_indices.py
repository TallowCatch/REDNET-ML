#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, math
from pathlib import Path
from collections import defaultdict

def to_float(s):
    try:
        return float(s)
    except:
        return None

def near_zero(x, eps=1e-10):
    return x is None or abs(x) <= eps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", default=None)

    # Core quality gates
    ap.add_argument("--min_valid_px", type=int, default=1024)
    ap.add_argument("--ndwi_max", type=float, default=0.25,
                    help="Drop if ndwi_mean > this (likely bright land/shallow surf)")

    # Relaxed behavior flags
    ap.add_argument("--require_l3", action="store_true",
                    help="If set, require at least one of chlor_a/kd490/flh to be present")
    ap.add_argument("--drop_all_zero_features", action="store_true",
                    help="If set, drop rows where ndwi,fai,rednir means AND stds are all ~0 (clearly empty/invalid tiles)")
    ap.add_argument("--dedupe_by", default="scene_id",
                    help="Key to dedupe on. Use 'scene_id' (default) or 'tile'. Set to '' to disable.")
    ap.add_argument("--keep", default=None,
                    help="Optional CSV of column=value filters to keep (e.g. 'tile=T39QXG')")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = args.out_csv or str(in_csv.with_name(in_csv.stem.replace(".csv","") + "_clean.csv"))

    rows = list(csv.DictReader(open(in_csv, newline="")))
    if not rows:
        print("No input rows.")
        return

    kept = []
    reasons = defaultdict(int)

    # Optional simple keep filter like tile=...,scene_id=...
    keep_filters = {}
    if args.keep:
        for kv in args.keep.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                keep_filters[k.strip()] = v.strip()

    for r in rows:
        # Optional include-only filtering
        if keep_filters:
            include = True
            for k, v in keep_filters.items():
                if str(r.get(k, "")).strip() != v:
                    include = False
                    break
            if not include:
                reasons["excluded_by_keep_filter"] += 1
                continue

        ndwi = to_float(r.get("ndwi_mean"))
        ndwi_std = to_float(r.get("ndwi_std"))
        fai = to_float(r.get("fai_mean"))
        fai_std = to_float(r.get("fai_std"))
        rednir = to_float(r.get("rednir_mean"))
        rednir_std = to_float(r.get("rednir_std"))

        chl  = to_float(r.get("chlor_a"))
        kd   = to_float(r.get("kd490"))
        flh  = to_float(r.get("flh"))
        vpx  = int(r.get("valid_px", "0") or 0)

        # 1) pixel count gate
        if vpx < args.min_valid_px:
            reasons["low_valid_px"] += 1
            continue

        # 2) obvious land/shallow (very bright) gate
        if ndwi is not None and ndwi > args.ndwi_max:
            reasons["ndwi_too_high"] += 1
            continue

        # 3) obviously empty/invalid tile (all features constant ~0)
        if args.drop_all_zero_features:
            if (near_zero(ndwi) and near_zero(ndwi_std) and
                near_zero(fai)  and near_zero(fai_std)  and
                near_zero(rednir) and near_zero(rednir_std)):
                reasons["all_zero_features"] += 1
                continue

        # 4) require at least one L3 value? (off by default)
        if args.require_l3 and all(x is None for x in (chl, kd, flh)):
            reasons["no_l3_values"] += 1
            continue

        kept.append(r)

    if not kept:
        print("No rows kept; consider relaxing filters (e.g., remove --require_l3 or disable --drop_all_zero_features).")
        # Print summary anyway
        if reasons:
            print("Drop summary:")
            for k, v in sorted(reasons.items()):
                print(f"  - {k}: {v}")
        return

    # Optional de-duplication
    if args.dedupe_by:
        key = args.dedupe_by.strip()
        buckets = {}
        for r in kept:
            k = r.get(key, "")
            # Keep the one with higher valid_px (fallback to first seen)
            vpx = int(r.get("valid_px", "0") or 0)
            if (k not in buckets) or (vpx > int(buckets[k].get("valid_px", "0") or 0)):
                buckets[k] = r
        output = list(buckets.values())
    else:
        output = kept

    # Write result
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(output)

    print(f"✓ Wrote {out_csv} ({len(output)} rows kept from {len(rows)} input)")

    if reasons:
        print("Drop summary:")
        for k, v in sorted(reasons.items()):
            print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
