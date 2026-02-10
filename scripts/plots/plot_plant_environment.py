from __future__ import annotations
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TILES_CSV = Path("runs/datasets/fusion_training_with_plants_splitlabels.csv")
OUT_PNG   = Path("runs/plots/study_area_tiles_heatmap.png")

COLOR_BY = "chlor_a"   # or "hab_prob" / "p_frcnn_r50_med" / "p_ssd_mb_med"
AGG = "mean"           # mean | median | max

# Your plants (for labeling only; no geo placement in this plot)
PLANTS = [
    {"name":"Sharqiyah","lat":21.9310725,"lon":59.6321193},
    {"name":"Sur","lat":22.622075,"lon":59.452973},
    {"name":"Musandam","lat":25.6716585,"lon":56.2667608},
    {"name":"Ghubrah","lat":23.6018997,"lon":58.4137212},
]

def parse_s2_parts(tile_name: str):
    """
    Example:
    S2A_MSIL2A_20170101T064252_R120_T40QFM_20210529T105726_0000.jpg
                            ^^^^ tile_id            ^^^^ chip
    We extract:
      - sentinel_mgrs_tile = T40QFM
      - chip_index = 0000 (int)
    """
    m_tile = re.search(r"_T([0-9]{2}[A-Z]{3})_", tile_name)
    m_chip = re.search(r"_(\d{4})\.(jpg|png)$", tile_name, re.IGNORECASE)
    if not m_tile or not m_chip:
        return None
    return "T" + m_tile.group(1), int(m_chip.group(1))

def agg(s: pd.Series, how: str) -> float:
    if how == "mean": return float(s.mean())
    if how == "median": return float(s.median())
    if how == "max": return float(s.max())
    raise ValueError(how)

def main():
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TILES_CSV)
    if "tile" not in df.columns:
        raise SystemExit("CSV must contain a 'tile' column.")
    if COLOR_BY not in df.columns:
        raise SystemExit(f"COLOR_BY='{COLOR_BY}' not in CSV. Columns: {list(df.columns)}")

    parts = df["tile"].apply(parse_s2_parts)
    if parts.isna().any():
        bad = df.loc[parts.isna(), "tile"].head(10).tolist()
        raise SystemExit(
            "Could not parse Sentinel-2 naming pattern for some tiles.\n"
            "Examples:\n" + "\n".join(bad)
        )

    df["mgrs_tile"] = [p[0] for p in parts]
    df["chip_idx"]  = [p[1] for p in parts]

    # Aggregate per (mgrs_tile, chip_idx)
    g = df.groupby(["mgrs_tile", "chip_idx"])[COLOR_BY].apply(lambda s: agg(s, AGG)).reset_index(name="val")

    tiles = sorted(g["mgrs_tile"].unique())
    chips = sorted(g["chip_idx"].unique())

    tile_to_y = {t:i for i,t in enumerate(tiles)}
    chip_to_x = {c:i for i,c in enumerate(chips)}

    grid = np.full((len(tiles), len(chips)), np.nan, dtype=float)
    for _, r in g.iterrows():
        y = tile_to_y[r["mgrs_tile"]]
        x = chip_to_x[int(r["chip_idx"])]
        grid[y, x] = float(r["val"])

    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    im = ax.imshow(grid, aspect="auto")

    ax.set_title(f"Tile Coverage Heatmap (Sentinel-2 chips) — {COLOR_BY} ({AGG})")
    ax.set_xlabel("chip index (_0000, _0001, ...)")
    ax.set_ylabel("Sentinel-2 MGRS tile (T..)")
    ax.set_yticks(range(len(tiles)))
    ax.set_yticklabels(tiles, fontsize=9)

    # keep x ticks light
    if len(chips) <= 20:
        ax.set_xticks(range(len(chips)))
        ax.set_xticklabels([f"{c:04d}" for c in chips], rotation=90, fontsize=8)
    else:
        ax.set_xticks([])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(COLOR_BY)

    # Optional annotation: plants list (since no geo mapping here)
    txt = "Plants in study:\n" + "\n".join([f"• {p['name']} ({p['lat']:.2f},{p['lon']:.2f})" for p in PLANTS])
    ax.text(1.02, 0.5, txt, transform=ax.transAxes, va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=220)
    print(f"Saved: {OUT_PNG}")

if __name__ == "__main__":
    main()
