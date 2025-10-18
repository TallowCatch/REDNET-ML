#!/usr/bin/env bash
set -euo pipefail

CSV_GLOB='data/aerial_*_20*/chip_indices_clean.csv'
HAB_GLOB='data/aerial_*_20*/chip_indices_clean_hab.csv'

# 1) Drop empty columns
for f in $CSV_GLOB; do
  echo "Input:  $f"
  python scripts/drop_empty_columns.py --in "$f" --inplace
done

# 2) Make HAB labels
for f in $CSV_GLOB; do
  echo "Labeling: $f"
  python scripts/make_hab_labels.py \
    --in_csv "$f" \
    --group season \
    --q 0.95 --q_floor 0.80 \
    --kd_floor 0.12 \
    --min_valid 2000
done

# 3) QC plots
mkdir -p qc/qc_all
python scripts/HAB_qc_report.py \
  --csv_glob "$HAB_GLOB" \
  --outdir qc/qc_all \
  --title "HAB QC"

# 4) HAB hits HTML gallery
OUTDIR="qc/hab_hits_inspect"
rm -rf "$OUTDIR"                  # <<< prevents SameFileError on reruns
mkdir -p "$OUTDIR"
python scripts/export_hab_hits.py \
  --glob "$HAB_GLOB" \
  --out_dir "$OUTDIR" \
  --min_chl 1.0 \
  --min_flh 0.2 \
  --link_mode copy

# Open gallery
if [ -f "$OUTDIR/index.html" ]; then
  command -v open >/dev/null 2>&1 && open "$OUTDIR/index.html"
  command -v xdg-open >/dev/null 2>&1 && xdg-open "$OUTDIR/index.html"
fi

# 5) Quick positive counts
for f in $HAB_GLOB; do
  printf "%s: " "$(basename "$(dirname "$f")")"
  python - "$f" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
print(int(df.get("hab_label", 0).sum()))
PY
done

echo "✅ Done."
