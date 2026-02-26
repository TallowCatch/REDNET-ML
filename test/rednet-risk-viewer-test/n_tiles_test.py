from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.data
def test_tile_and_scene_aggregation_from_example_csv() -> None:
    csv_path = Path(__file__).resolve().parent / "examples" / "cv4_test_infer.csv"
    df = pd.read_csv(csv_path)

    df["scene_root"] = df["scene_id"].str.split("_20").str[0]
    summary = (
        df.groupby("scene_root")
        .agg(
            n_tiles=("tile", "count"),
            n_acquisitions=("scene_id", "nunique"),
        )
        .sort_values("n_tiles", ascending=False)
    )

    assert not summary.empty
    assert int(summary["n_tiles"].min()) >= 1
    assert int(summary["n_acquisitions"].min()) >= 1
    assert bool((summary["n_tiles"] >= summary["n_acquisitions"]).all())
