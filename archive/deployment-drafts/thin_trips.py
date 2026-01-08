#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--every_n", type=int, default=10, help="keep every Nth point per pid")
    ap.add_argument("--min_time_gap_s", type=int, default=0, help="optional: also enforce min seconds between kept points")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # expected columns
    # pid, lon, lat, time
    if "pid" not in df.columns or "lon" not in df.columns or "lat" not in df.columns or "time" not in df.columns:
        raise SystemExit("CSV must have columns: pid, lon, lat, time")

    df = df.sort_values(["pid", "time"])

    # Keep every Nth row per pid
    df["_k"] = df.groupby("pid").cumcount()
    df2 = df[df["_k"] % args.every_n == 0].copy()
    df2.drop(columns=["_k"], inplace=True)

    # Optional: enforce minimum time gap between kept points per pid
    if args.min_time_gap_s > 0:
        df2 = df2.sort_values(["pid", "time"])
        keep_rows = []
        last_t = {}
        for idx, r in df2.iterrows():
            pid = r["pid"]
            t = r["time"]
            lt = last_t.get(pid, None)
            if lt is None or (t - lt) >= args.min_time_gap_s:
                keep_rows.append(idx)
                last_t[pid] = t
        df2 = df2.loc[keep_rows].copy()

    df2.to_csv(args.out_csv, index=False)
    print(f"✅ wrote {args.out_csv} rows={len(df2):,} (from {len(df):,})")

if __name__ == "__main__":
    main()
