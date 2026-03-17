#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Matched CSV not found: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_predictions(glob_str: str) -> pd.DataFrame:
    frames = []
    for plant_id, df in common.load_prediction_series(glob_str).items():
        tmp = df.copy()
        tmp["plant_id"] = plant_id
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _event_window_mask(preds: pd.DataFrame, events: pd.DataFrame, expand_days: int = 0) -> pd.Series:
    mask = pd.Series(False, index=preds.index)
    for _, row in events.iterrows():
        pid = common.normalize_plant_id(row.get("assigned_plant_id"))
        if not pid:
            continue
        start = common.parse_date_start(row.get("event_date_start"))
        end = common.parse_date_end(row.get("event_date_end"))
        if pd.isna(start) or pd.isna(end):
            continue
        if expand_days > 0:
            start = start - pd.Timedelta(days=expand_days)
            end = end + pd.Timedelta(days=expand_days)
        local = (preds["plant_id"] == pid) & (preds["datetime"] >= start) & (preds["datetime"] <= end)
        mask = mask | local
    return mask


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def score_event_mode(
    matched_csv: Path,
    prediction_glob: str,
    outdir: Path,
    watch_threshold: float,
    action_threshold: float,
) -> None:
    df = _read_csv_or_empty(matched_csv)
    outdir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "mode": "event_validation",
        "input_csv": str(matched_csv),
        "watch_threshold": watch_threshold,
        "action_threshold": action_threshold,
        "n_rows": int(len(df)),
    }

    if df.empty:
        summary["note"] = "No external event rows available."
        common.write_json(outdir / "event_validation_summary.json", summary)
        _write_markdown(outdir / "event_validation_report.md", "# Event Validation\n\nNo external event rows available.\n")
        return

    df["external_positive"] = pd.to_numeric(df.get("external_positive"), errors="coerce").fillna(0).astype(int)
    df["matched_ops_risk"] = pd.to_numeric(df.get("matched_ops_risk"), errors="coerce")
    df["matched_hab_prob"] = pd.to_numeric(df.get("matched_hab_prob"), errors="coerce")
    df["match_day_diff"] = pd.to_numeric(df.get("match_day_diff"), errors="coerce")
    df["within_primary_window"] = df.get("within_primary_window", False).astype(bool)
    df["within_sensitivity_window"] = df.get("within_sensitivity_window", False).astype(bool)
    score_col = "matched_ops_risk" if df["matched_ops_risk"].notna().any() else "matched_hab_prob"
    summary["primary_score_col"] = score_col

    primary = df[df["within_primary_window"] & df[score_col].notna()].copy()
    summary["n_primary_matched"] = int(len(primary))
    summary["n_sensitivity_matched"] = int(
        len(df[df["within_sensitivity_window"] & df[score_col].notna()].copy())
    )
    summary["n_positive_primary"] = int((primary["external_positive"] == 1).sum())
    summary["n_negative_primary"] = int((primary["external_positive"] == 0).sum())

    pos = primary[primary["external_positive"] == 1]
    neg = primary[primary["external_positive"] == 0]

    summary["event_hit_rate_watch"] = (
        float((pos[score_col] >= watch_threshold).mean()) if not pos.empty else None
    )
    summary["event_hit_rate_action"] = (
        float((pos[score_col] >= action_threshold).mean()) if not pos.empty else None
    )
    summary["median_positive_score"] = float(pos[score_col].median()) if not pos.empty else None
    summary["median_negative_score"] = float(neg[score_col].median()) if not neg.empty else None
    summary["median_positive_hab_prob"] = float(pos["matched_hab_prob"].median()) if not pos.empty else None
    summary["median_negative_hab_prob"] = float(neg["matched_hab_prob"].median()) if not neg.empty else None

    metrics = common.compute_binary_metrics(
        primary["external_positive"].to_numpy(dtype=int),
        primary[score_col].to_numpy(dtype=float),
    )
    summary.update(metrics)

    top = pos.sort_values("matched_hab_prob", ascending=False).head(10).copy()
    top.to_csv(outdir / "top_ranked_event_matches.csv", index=False)
    primary[["event_id", "assigned_plant_id", "match_day_diff", "matched_prediction_datetime"]].to_csv(
        outdir / "event_lead_lag_table.csv", index=False
    )

    preds = _load_predictions(prediction_glob)
    event_windows = pos[["event_id", "assigned_plant_id", "event_date_start", "event_date_end"]].copy()
    if not preds.empty and not event_windows.empty:
        mask = _event_window_mask(preds, event_windows, expand_days=common.DEFAULT_PRIMARY_WINDOW_DAYS)
        preds_score_col = "ops_risk" if "ops_risk" in preds.columns else "hab_prob"
        event_scores = pd.to_numeric(preds.loc[mask, preds_score_col], errors="coerce").dropna()
        nonevent_scores = pd.to_numeric(preds.loc[~mask, preds_score_col], errors="coerce").dropna()
        summary["event_window_median_score"] = float(event_scores.median()) if not event_scores.empty else None
        summary["nonevent_window_median_score"] = (
            float(nonevent_scores.median()) if not nonevent_scores.empty else None
        )
        if not event_scores.empty and not nonevent_scores.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot([event_scores.to_numpy(), nonevent_scores.to_numpy()], tick_labels=["event", "non-event"])
            ax.set_ylabel(preds_score_col)
            ax.set_title("Event vs non-event risk windows")
            fig.tight_layout()
            fig.savefig(outdir / "event_vs_nonevent_score.png", dpi=150)
            plt.close(fig)

    note = (
        "Primary external event validation uses matched plant-date windows only. "
        "This is event-based external concordance, not in-situ validation."
    )
    if summary.get("auroc") is None or summary.get("auprc") is None:
        note += " Sample size or class balance was insufficient for stable AUROC/AUPRC."
    summary["note"] = note
    common.write_json(outdir / "event_validation_summary.json", summary)

    report = [
        "# Event Validation",
        "",
        f"- Primary matched rows: {summary['n_primary_matched']}",
        f"- Positive primary events: {summary['n_positive_primary']}",
        f"- Negative primary events: {summary['n_negative_primary']}",
        f"- Primary score column: {summary['primary_score_col']}",
        f"- Event hit rate at WATCH: {summary['event_hit_rate_watch']}",
        f"- Event hit rate at ACTION: {summary['event_hit_rate_action']}",
        f"- AUROC: {summary.get('auroc')}",
        f"- AUPRC: {summary.get('auprc')}",
        "",
        note,
        "",
        "Use this as supplementary external validation only. Do not call it in-situ validation.",
        "",
        "Key outputs:",
        "- `top_ranked_event_matches.csv`",
        "- `event_lead_lag_table.csv`",
        "- `event_vs_nonevent_score.png` when enough event and non-event windows exist",
    ]
    _write_markdown(outdir / "event_validation_report.md", "\n".join(report) + "\n")


def score_insitu_mode(
    matched_csv: Path,
    outdir: Path,
    watch_threshold: float,
    action_threshold: float,
) -> None:
    df = _read_csv_or_empty(matched_csv)
    outdir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "mode": "insitu_validation",
        "input_csv": str(matched_csv),
        "watch_threshold": watch_threshold,
        "action_threshold": action_threshold,
        "n_rows": int(len(df)),
    }

    if df.empty:
        summary["note"] = "No in-situ records were available."
        common.write_json(outdir / "insitu_validation_summary.json", summary)
        return

    df["hab_event"] = pd.to_numeric(df.get("hab_event"), errors="coerce").fillna(0).astype(int)
    df["matched_ops_risk"] = pd.to_numeric(df.get("matched_ops_risk"), errors="coerce")
    df["matched_hab_prob"] = pd.to_numeric(df.get("matched_hab_prob"), errors="coerce")
    df["within_primary_window"] = df.get("within_primary_window", False).astype(bool)
    df["within_sensitivity_window"] = df.get("within_sensitivity_window", False).astype(bool)
    score_col = "matched_ops_risk" if df["matched_ops_risk"].notna().any() else "matched_hab_prob"
    summary["primary_score_col"] = score_col

    primary = df[df["within_primary_window"] & df[score_col].notna()].copy()
    sensitivity = df[df["within_sensitivity_window"] & df[score_col].notna()].copy()

    summary["n_primary_matched"] = int(len(primary))
    summary["n_sensitivity_matched"] = int(len(sensitivity))

    for label, subset in [("primary", primary), ("sensitivity", sensitivity)]:
        metrics = common.compute_binary_metrics(
            subset["hab_event"].to_numpy(dtype=int),
            subset[score_col].to_numpy(dtype=float),
        )
        summary[f"{label}_auroc"] = metrics.get("auroc")
        summary[f"{label}_auprc"] = metrics.get("auprc")
        summary[f"{label}_watch"] = common.confusion_at_threshold(
            subset["hab_event"].to_numpy(dtype=int),
            subset[score_col].to_numpy(dtype=float),
            watch_threshold,
        )
        summary[f"{label}_action"] = common.confusion_at_threshold(
            subset["hab_event"].to_numpy(dtype=int),
            subset[score_col].to_numpy(dtype=float),
            action_threshold,
        )

    summary["note"] = (
        "This summary is only strong enough to describe true in-situ validation if the input records are genuine field "
        "observations rather than proxy or public-event rows."
    )
    common.write_json(outdir / "insitu_validation_summary.json", summary)


def main() -> None:
    ap = argparse.ArgumentParser("Score matched external event or in-situ validation tables.")
    ap.add_argument("--mode", choices=["event", "insitu"], required=True)
    ap.add_argument("--matched_csv", required=True)
    ap.add_argument("--prediction_glob", default="rednet-risk-viewer/public/data/plant_*_hab.csv")
    ap.add_argument("--outdir", default="runs/eval/external_validation")
    ap.add_argument("--watch_threshold", type=float, default=common.DEFAULT_WATCH_THRESHOLD)
    ap.add_argument("--action_threshold", type=float, default=common.DEFAULT_ACTION_THRESHOLD)
    args = ap.parse_args()

    matched_csv = Path(args.matched_csv)
    outdir = Path(args.outdir)
    if args.mode == "event":
        score_event_mode(matched_csv, args.prediction_glob, outdir, args.watch_threshold, args.action_threshold)
    else:
        score_insitu_mode(matched_csv, outdir, args.watch_threshold, args.action_threshold)
    print(f"[ok] wrote validation outputs under {outdir}")


if __name__ == "__main__":
    main()
