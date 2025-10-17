#!/usr/bin/env bash
set -euo pipefail

# ---------- config ----------
LISTDIR="filelists_8d"
CHIPS_GLOB='data/aerial_*_20*/chip_indices_clean.csv'
TMPROOT="data/l3/tmp_8d"        # temporary working directory
FINALROOT="data/l3/aqua_8d"     # where your long-term files live (will be cleaned after append)
DL="python scripts/obdaac_download.py"
APP="python scripts/append_modis_features_8d.py"
MAX_DAYS=30

# Map product → filelist path (bash-3 safe)
filelist_for() {
  case "$1" in
    chlor_a) echo "$LISTDIR/filelist_8d_chlor_a.txt" ;;
    Kd_490)  echo "$LISTDIR/filelist_8d_Kd_490.txt"  ;;
    nflh)    echo "$LISTDIR/filelist_8d_nflh.txt"    ;;
    *) echo "Unknown product: $1" >&2; exit 2 ;;
  esac
}

# Require appkey
: "${OBPG_APPKEY:?OBPG_APPKEY not set}"

mkdir -p "$TMPROOT" "$FINALROOT"

for PROD in chlor_a Kd_490 nflh; do
  echo "=== ${PROD}: check → download → append → delete ==="
  FLIST="$(filelist_for "$PROD")"
  ODIR="${TMPROOT}/${PROD}"
  FINALDIR="${FINALROOT}/${PROD}"
  mkdir -p "$ODIR" "$FINALDIR"

  if [[ ! -f "$FLIST" ]]; then
    echo "⚠️  Missing filelist: $FLIST — skipping ${PROD}."
    continue
  fi

  # Decide source dir
  if compgen -G "${FINALDIR}/*.nc" > /dev/null; then
    echo "↪ ${PROD}: found existing files in ${FINALDIR} — skipping download, proceeding to append."
    USE_DIR="$FINALDIR"
  else
    echo "↓ ${PROD}: no existing files found — downloading fresh into ${ODIR} ..."
    USE_DIR="$ODIR"
    $DL --filelist "$FLIST" --odir "$ODIR" --appkey "$OBPG_APPKEY" -v
  fi

  # Keep files if anything fails after this point
  keep_on_error() {
    echo "❌ ${PROD}: error occurred — keeping files in ${USE_DIR}."
  }
  trap keep_on_error ERR

  # Append
  echo "➕ ${PROD}: appending features into CSVs ..."
  $APP --chips_csv_glob "$CHIPS_GLOB" --modis_root "$USE_DIR" --max_days "$MAX_DAYS" --products "$PROD"

  # Success: remove trap, now cleanup used files
  trap - ERR

  if [[ "$USE_DIR" == "$ODIR" ]]; then
    echo "🧹 ${PROD}: removing temporary ${ODIR}"
    rm -rf "$ODIR"
  else
    echo "🧹 ${PROD}: append done — deleting used files from ${FINALDIR}"
    # Delete only data files; keep the folder
    rm -f "${FINALDIR}/"*.nc 2>/dev/null || true
    rm -f "${FINALDIR}/"*.nc.gz 2>/dev/null || true
    rm -f "${FINALDIR}/"*.bz2 2>/dev/null || true
  fi

  echo "✔ ${PROD}: done."
done

echo "✅ All products processed. CSVs updated and used files removed."
