from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.data
def test_alert_rate_from_example_predictions() -> None:
    csv_path = Path(__file__).resolve().parent / "examples" / "cv4_test_infer.csv"
    df = pd.read_csv(csv_path)

    alert_rate = pd.to_numeric(df["pred"], errors="coerce").mean()

    assert pd.notna(alert_rate)
    assert 0.0 <= float(alert_rate) <= 1.0
