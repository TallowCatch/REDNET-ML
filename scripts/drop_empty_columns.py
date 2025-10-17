#!/usr/bin/env python3
import argparse, csv, sys
from pathlib import Path

EMPTY_TOKENS = {"", "nan", "NaN", "None", "null", "NULL"}

def is_empty_cell(v):
    if v is None:
        return True
    s = str(v).strip()
    return s in EMPTY_TOKENS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV")
    ap.add_argument("--out", dest="outp", help="Output CSV (default: <input>_noempty.csv)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite the input file")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        sys.exit(f"Input not found: {inp}")

    rows = list(csv.DictReader(open(inp, newline="")))
    if not rows:
        sys.exit("No rows found in input CSV.")

    header = rows[0].keys()
    # Track if every cell in a column is empty
    all_empty = {col: True for col in header}

    for r in rows:
        for col in header:
            if all_empty[col] and not is_empty_cell(r.get(col)):
                all_empty[col] = False

    # Columns to drop/keep
    drop_cols = [c for c, empty in all_empty.items() if empty]
    keep_cols = [c for c in header if c not in drop_cols]

    # Choose output path
    if args.inplace:
        outp = inp
    else:
        outp = Path(args.outp) if args.outp else inp.with_name(inp.stem + "_noempty.csv")

    # Write cleaned CSV
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keep_cols})

    print(f"Input:  {inp}")
    print(f"Output: {outp}")
    if drop_cols:
        print("Dropped empty columns:")
        for c in drop_cols:
            print(f"  - {c}")
    else:
        print("No empty columns detected.")

if __name__ == "__main__":
    main()
