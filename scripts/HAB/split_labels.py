#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)

    # label columns
    ap.add_argument("--final_col", default="hab_label_final")  # your current final training label
    ap.add_argument("--trusted_source_col", default="hab_label_source_fusion")  # binary: 1 means trusted-origin

    args = ap.parse_args()
    df = pd.read_csv(args.in_csv)

    for c in [args.final_col, args.trusted_source_col]:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # trusted = positives that came from the "trusted" source (fusion table)
    df["hab_label_trusted"] = ((df[args.final_col] == 1) & (df[args.trusted_source_col] == 1)).astype(int)

    # weak = positives that are final positives but not trusted-origin
    df["hab_label_weak"] = ((df[args.final_col] == 1) & (df[args.trusted_source_col] == 0)).astype(int)

    # final2 = trusted OR weak (should equal your final_col positives)
    df["hab_label_final2"] = ((df["hab_label_trusted"] == 1) | (df["hab_label_weak"] == 1)).astype(int)

    df.to_csv(args.out_csv, index=False)

    print(f"✓ wrote {args.out_csv}")
    print("final positives:", int(df[args.final_col].sum()))
    print("trusted positives:", int(df["hab_label_trusted"].sum()))
    print("weak positives:", int(df["hab_label_weak"].sum()))
    print("final2 positives:", int(df["hab_label_final2"].sum()))
    print("rows:", len(df))

if __name__ == "__main__":
    main()
