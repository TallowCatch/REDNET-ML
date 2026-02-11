import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def find_col(cols, *candidates, contains=None):
    cols_lower = {c.lower(): c for c in cols}
    # exact candidates first
    for cand in candidates:
        if cand in cols: 
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # then substring search
    if contains:
        want = [w.lower() for w in contains]
        for c in cols:
            lc = c.lower()
            if all(w in lc for w in want):
                return c
    return None

def main(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    # Try to locate common metric names across Ultralytics versions
    col_map50   = find_col(cols, "metrics/mAP50(B)", "metrics/mAP50", "map50", contains=["map50"])
    col_map5095 = find_col(cols, "metrics/mAP50-95(B)", "metrics/mAP(B)", "map50-95", contains=["map50", "95"])
    col_prec    = find_col(cols, "metrics/precision(B)", "precision", contains=["precision"])
    col_recall  = find_col(cols, "metrics/recall(B)", "recall", contains=["recall"])
    tr_box      = find_col(cols, "train/box_loss", contains=["train","box","loss"])
    va_box      = find_col(cols, "val/box_loss",   contains=["val","box","loss"])

    # Choose best epoch: prefer mAP50; else mAP50-95; else last row
    if col_map50 and df[col_map50].notna().any():
        best_idx = int(df[col_map50].idxmax())
    elif col_map5095 and df[col_map5095].notna().any():
        best_idx = int(df[col_map5095].idxmax())
    else:
        best_idx = len(df) - 1

    # Build a small summary dict
    row = df.loc[best_idx]
    summary = {
        "epoch": best_idx,
        "mAP50":   (row[col_map50]   if col_map50   else None),
        "mAP50-95":(row[col_map5095] if col_map5095 else None),
        "precision":(row[col_prec]   if col_prec    else None),
        "recall":   (row[col_recall] if col_recall  else None),
    }

    # Write summary.txt
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k,v in summary.items():
            if v is None: continue
            f.write(f"{k},{v:.3f}\n" if isinstance(v, (int,float)) else f"{k},{v}\n")

    # Pretty print for terminal
    nice = ", ".join([f"{k}={v:.3f}" if isinstance(v,(int,float)) else f"{k}={v}"
                      for k,v in summary.items() if v is not None])
    print(f"Best epoch {best_idx} | {nice}")
    print(f"Saved summary → {os.path.join(out_dir,'summary.txt')}")

    # Plot curves: losses (left) + mAP50 or mAP50-95 (right)
    plt.figure(figsize=(9,4))
    ax1 = plt.gca()
    plotted_left = False
    if tr_box and tr_box in df: 
        ax1.plot(df.index, df[tr_box], label="train box loss"); plotted_left = True
    if va_box and va_box in df: 
        ax1.plot(df.index, df[va_box], label="val box loss");   plotted_left = True
    if plotted_left:
        ax1.set_ylabel("loss"); ax1.legend(loc="upper left")
    ax1.set_xlabel("epoch")

    ax2 = ax1.twinx()
    if col_map50 and col_map50 in df:
        ax2.plot(df.index, df[col_map50], "g-", label="mAP50")
        ax2.set_ylabel("mAP50")
    elif col_map5095 and col_map5095 in df:
        ax2.plot(df.index, df[col_map5095], "g-", label="mAP50-95")
        ax2.set_ylabel("mAP50-95")

    plt.title("YOLO training")
    plt.tight_layout()
    out_png = os.path.join(out_dir, "results_plot.png")
    plt.savefig(out_png, dpi=220)
    plt.close()
    print("Saved plot  →", out_png)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/yolo/hab_yolov8n3/results.csv")
    ap.add_argument("--out", default="runs/yolo/hab_yolov8n3")
    args = ap.parse_args()
    main(args.csv, args.out)
