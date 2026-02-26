from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.data
def test_unique_scene_roots_from_example_csv() -> None:
    csv_path = Path(__file__).resolve().parent / "examples" / "cv4_test_infer.csv"
    df = pd.read_csv(csv_path)

    df["scene_root"] = df["scene_id"].map(lambda s: "_".join(str(s).split("_")[:5]))
    counts = df.groupby("scene_root").size().sort_values(ascending=False)

    assert "scene_root" in df.columns
    assert df["scene_root"].nunique() > 0
    assert int(counts.iloc[0]) >= 1
