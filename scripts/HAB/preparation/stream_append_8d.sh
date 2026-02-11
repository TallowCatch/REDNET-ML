#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
LISTDIR="data/filelists/8d"   # ✅ updated path
CHIPS_GLOB='data/aerial_*_20*/chip_indices_clean.csv'
TMPROOT="data/l3/tmp_8d"
FINALROOT="data/l3/aqua_8d"
DL="python scripts/obdaac_download.py"
APP="python scripts/append_modis_features_8d.py"
MAX_DAYS=30

# Require appkey
: "${OBPG_APPKEY:?OBPG_APPKEY not set}"

# ---- mappers (Bash 3 compatible) ----
filelist_for() {
  case "$1" in
    chlor_a) echo "$LISTDIR/filelist_8d_chlor_a_filtered.txt" ;;
    Kd_490)  echo "$LISTDIR/filelist_8d_Kd_490_filtered.txt"  ;;
    nflh)    echo "$LISTDIR/filelist_8d_nflh_filtered.txt"    ;;
    sst)     echo "$LISTDIR/filelist_8d_sst.txt"              ;;  
    *) echo "Unknown product: $1" >&2; exit 2 ;;
  esac
}

# Folder name under FINALROOT to keep things tidy
final_subdir_for() {
  case "$1" in
    chlor_a) echo "chlor_a" ;;
    Kd_490)  echo "Kd_490"  ;;
    nflh)    echo "nFLH"    ;;
    sst)     echo "sst"     ;;  # ✅ new
    *) echo "Unknown product: $1" >&2; exit 2 ;;
  esac
}

mkdir -p "$TMPROOT" "$FINALROOT"

# ======== Product selection ========
if [[ "$#" -gt 0 ]]; then
  PRODS=("$@")
else
  PRODS=(chlor_a Kd_490 nflh sst)  # ✅ include sst by default (optional)
fi

echo "Products to process: ${PRODS[*]}"

for PROD in "${PRODS[@]}"; do
  echo "=== ${PROD}: check → download → append → delete ==="

  FLIST="$(filelist_for "$PROD")"
  SUBDIR="$(final_subdir_for "$PROD")"
  ODIR="${TMPROOT}/${SUBDIR}"
  FINALDIR="${FINALROOT}/${SUBDIR}"

  if [[ ! -f "$FLIST" ]]; then
    echo "⚠️  Missing filelist: $FLIST — skipping ${PROD}."
    continue
  fi

  mkdir -p "$ODIR" "$FINALDIR"

  # --- (1) Decide source directory ---
  if compgen -G "${FINALDIR}/*.nc"* > /dev/null; then
    echo "↪ ${PROD}: found existing files in ${FINALDIR} — skipping download, will append from there."
    USE_DIR="$FINALDIR"
  else
    echo "↓ ${PROD}: no existing files — downloading into ${FINALDIR} ..."
    USE_DIR="$FINALDIR"
    $DL --filelist "$FLIST" --odir "$FINALDIR" --appkey "$OBPG_APPKEY" -v
    if ! compgen -G "${FINALDIR}/*.nc"* > /dev/null; then
      echo "⚠️  ${PROD}: download produced no .nc files — skipping append for this product."
      continue
    fi
  fi

  # Keep files if any error occurs during append
  keep_on_error() {
    echo "❌ ${PROD}: error during append — keeping files in ${USE_DIR}."
  }
  trap keep_on_error ERR

  # --- (2) Append into season CSVs ---
  echo "➕ ${PROD}: appending into CSVs ..."
  $APP --chips_csv_glob "$CHIPS_GLOB" --modis_root "$USE_DIR" --max_days "$MAX_DAYS" --products "$PROD"

  # --- (3) Cleanup used files after successful append ---
  trap - ERR
  echo "🧹 ${PROD}: append done — deleting used files from ${FINALDIR}"
  rm -f "${FINALDIR}/"*.nc* 2>/dev/null || true

  echo "✔ ${PROD}: done."
done

echo "✅ Selected products processed. CSVs updated and data files cleaned."
