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
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common


def _event_ts(row: pd.Series) -> pd.Timestamp:
    start = common.parse_date_start(row.get("event_date_start"))
    end = common.parse_date_end(row.get("event_date_end"))
    return common.midpoint_timestamp(start, end)


def _load_matched(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing matched event CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["assigned_plant_id"] = df["assigned_plant_id"].map(common.normalize_plant_id)
    df["matched_prediction_datetime"] = pd.to_datetime(df["matched_prediction_datetime"], errors="coerce", utc=True)
    df["match_day_diff"] = pd.to_numeric(df["match_day_diff"], errors="coerce")
    df["matched_hab_prob"] = pd.to_numeric(df["matched_hab_prob"], errors="coerce")
    df["within_primary_window"] = df["within_primary_window"].astype(bool)
    df["within_sensitivity_window"] = df["within_sensitivity_window"].astype(bool)
    df["event_ts"] = df.apply(_event_ts, axis=1)
    return df


def build_plant_summary(plants: pd.DataFrame, matched: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, plant in plants.iterrows():
        pid = str(plant["plant_id"])
        p_events = matched[matched["assigned_plant_id"] == pid].copy() if not matched.empty else matched.copy()
        p_primary = p_events[p_events["within_primary_window"]].copy() if not p_events.empty else p_events.copy()
        p_sens = p_events[p_events["within_sensitivity_window"]].copy() if not p_events.empty else p_events.copy()
        p_pred = preds.get(pid, pd.DataFrame()).copy()
        if not p_pred.empty:
            risk_col = "ops_risk" if "ops_risk" in p_pred.columns else "hab_prob"
            p_pred = p_pred.sort_values(risk_col, ascending=False)
            top = p_pred.iloc[0]
            max_hab_prob = float(top.get("hab_prob")) if pd.notna(top.get("hab_prob")) else None
            max_ops_risk = float(top.get("ops_risk")) if pd.notna(top.get("ops_risk")) else None
            top_datetime = top.get("datetime")
            top_tile = top.get("tile")
            series_start = p_pred["datetime"].min()
            series_end = p_pred["datetime"].max()
            n_datetimes = int(len(p_pred))
        else:
            max_hab_prob = None
            max_ops_risk = None
            top_datetime = None
            top_tile = None
            series_start = None
            series_end = None
            n_datetimes = 0

        pos_primary = p_primary[p_primary["external_positive"] == 1]
        neg_primary = p_primary[p_primary["external_positive"] == 0]
        rows.append(
            {
                "plant_id": pid,
                "plant_name": plant["name"],
                "n_events_total": int(len(p_events)),
                "n_primary_matched": int(len(p_primary)),
                "n_sensitivity_matched": int(len(p_sens)),
                "n_positive_total": int((p_events.get("external_positive", pd.Series(dtype=int)) == 1).sum()),
                "n_negative_total": int((p_events.get("external_positive", pd.Series(dtype=int)) == 0).sum()),
                "n_positive_primary": int((p_primary.get("external_positive", pd.Series(dtype=int)) == 1).sum()),
                "n_negative_primary": int((p_primary.get("external_positive", pd.Series(dtype=int)) == 0).sum()),
                "median_primary_hab_prob_positive": (
                    float(pos_primary["matched_hab_prob"].median()) if not pos_primary.empty else None
                ),
                "median_primary_hab_prob_negative": (
                    float(neg_primary["matched_hab_prob"].median()) if not neg_primary.empty else None
                ),
                "median_primary_ops_risk_positive": (
                    float(pos_primary["matched_ops_risk"].median())
                    if ("matched_ops_risk" in pos_primary.columns and not pos_primary.empty)
                    else None
                ),
                "median_primary_ops_risk_negative": (
                    float(neg_primary["matched_ops_risk"].median())
                    if ("matched_ops_risk" in neg_primary.columns and not neg_primary.empty)
                    else None
                ),
                "series_start": series_start,
                "series_end": series_end,
                "n_datetimes_2017_2024": n_datetimes,
                "max_hab_prob_2017_2024": max_hab_prob,
                "max_ops_risk_2017_2024": max_ops_risk,
                "top_datetime": top_datetime,
                "top_tile": top_tile,
            }
        )
    return pd.DataFrame(rows)


def build_top_risk_table(preds: dict[str, pd.DataFrame], plants: pd.DataFrame, topk: int = 5) -> pd.DataFrame:
    out = []
    name_map = plants.set_index("plant_id")["name"].to_dict()
    for pid, df in preds.items():
        risk_col = "ops_risk" if "ops_risk" in df.columns else "hab_prob"
        top = df.sort_values(risk_col, ascending=False).head(topk).copy()
        top["plant_id"] = pid
        top["plant_name"] = name_map.get(pid)
        cols = ["plant_id", "plant_name"] + [c for c in top.columns if c not in {"plant_id", "plant_name"}]
        top = top[cols]
        out.append(top)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def plot_plant_timeline(plant_name: str, pred_df: pd.DataFrame, event_df: pd.DataFrame, out_path: Path) -> None:
    if pred_df.empty:
        return
    risk_col = "ops_risk" if "ops_risk" in pred_df.columns else "hab_prob"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pred_df["datetime"], pred_df[risk_col], color="#2c7fb8", lw=1.6)
    ax.scatter(pred_df["datetime"], pred_df[risk_col], color="#2c7fb8", s=12)

    if not event_df.empty:
        for _, row in event_df.iterrows():
            event_ts = row["event_ts"]
            if pd.isna(event_ts):
                continue
            positive = int(row["external_positive"]) == 1
            color = "#d7301f" if positive else "#636363"
            ls = "-" if bool(row["within_primary_window"]) else "--"
            ax.axvline(event_ts, color=color, linestyle=ls, alpha=0.65, lw=1.2)

    ax.set_title(f"{plant_name}: 2017-2024 plant-date risk with external event markers")
    ax.set_ylabel(risk_col)
    ax.set_xlabel("datetime")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_event_windows(plant_name: str, pred_df: pd.DataFrame, event_df: pd.DataFrame, out_path: Path, window_days: int) -> None:
    if pred_df.empty or event_df.empty:
        return
    risk_col = "ops_risk" if "ops_risk" in pred_df.columns else "hab_prob"
    events = event_df.sort_values("event_ts").reset_index(drop=True)
    n = len(events)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3, 2.6 * n)), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, events.iterrows()):
        event_ts = row["event_ts"]
        if pd.isna(event_ts):
            continue
        lo = event_ts - pd.Timedelta(days=window_days)
        hi = event_ts + pd.Timedelta(days=window_days)
        sub = pred_df[(pred_df["datetime"] >= lo) & (pred_df["datetime"] <= hi)].copy()
        ax.plot(sub["datetime"], sub[risk_col], color="#2c7fb8", lw=1.6)
        ax.scatter(sub["datetime"], sub[risk_col], color="#2c7fb8", s=12)
        ax.axvline(event_ts, color="#111111", linestyle="--", lw=1.2)
        if pd.notna(row.get("matched_prediction_datetime")):
            ax.scatter(
                [pd.to_datetime(row["matched_prediction_datetime"], utc=True)],
                [row["matched_ops_risk"] if "matched_ops_risk" in row and pd.notna(row["matched_ops_risk"]) else row["matched_hab_prob"]],
                color="#d7301f" if int(row["external_positive"]) == 1 else "#636363",
                s=36,
                zorder=5,
            )
        label = "positive" if int(row["external_positive"]) == 1 else "negative"
        ax.set_title(
            f"{row['location_name']} | {row['event_date_start']} | {label} | "
            f"Δdays={None if pd.isna(row['match_day_diff']) else round(float(row['match_day_diff']), 2)}"
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(risk_col)
    axes[-1].set_xlabel("datetime")
    fig.suptitle(f"{plant_name}: event-window risk traces", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_event_count_bars(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(summary_df))
    ax.bar(x, summary_df["n_events_total"], color="#a6bddb", label="AOI events")
    ax.bar(x, summary_df["n_primary_matched"], color="#045a8d", label="primary matched")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["plant_name"], rotation=20, ha="right")
    ax.set_ylabel("count")
    ax.set_title("AOI external event coverage by plant")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_markdown(summary_df: pd.DataFrame, matched: pd.DataFrame) -> str:
    total_events = int(len(matched))
    primary = int(matched["within_primary_window"].sum()) if not matched.empty else 0
    sensitivity = int(matched["within_sensitivity_window"].sum()) if not matched.empty else 0
    lines = [
        "# AOI Event Validation",
        "",
        f"- AOI-specific external event rows: {total_events}",
        f"- Primary matched rows (±{common.DEFAULT_PRIMARY_WINDOW_DAYS} days): {primary}",
        f"- Sensitivity matched rows (±{common.DEFAULT_SENSITIVITY_WINDOW_DAYS} days): {sensitivity}",
        "",
        "This report is limited to the dissertation AOIs and uses event-level external validation only.",
        "It does not convert public reports into in-situ measurements.",
        "",
        "## By Plant",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"### {row['plant_name']}",
                f"- AOI event rows: {row['n_events_total']}",
                f"- Primary matched: {row['n_primary_matched']}",
                f"- Sensitivity matched: {row['n_sensitivity_matched']}",
                f"- Top historical hab_prob (2017-2024): {row['max_hab_prob_2017_2024']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser("Build AOI-specific historical validation tables and plots.")
    ap.add_argument("--plants_json", default="rednet-risk-viewer/public/data/plants.json")
    ap.add_argument("--prediction_glob", default="rednet-risk-viewer/public/data/plant_*_hab.csv")
    ap.add_argument("--matched_events_csv", default="runs/eval/external_validation/matched_external_events.csv")
    ap.add_argument("--outdir", default="runs/eval/external_validation")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--window_days", type=int, default=45)
    args = ap.parse_args()

    plants = common.load_plants(Path(args.plants_json))
    preds = common.load_prediction_series(args.prediction_glob)
    matched = _load_matched(Path(args.matched_events_csv))
    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"

    summary_df = build_plant_summary(plants, matched, preds)
    top_df = build_top_risk_table(preds, plants, topk=args.topk)

    summary_path = outdir / "aoi_event_validation_by_plant.csv"
    top_path = outdir / "top_risk_by_plant_2017_2024.csv"
    summary_df.to_csv(summary_path, index=False)
    top_df.to_csv(top_path, index=False)

    for _, plant in plants.iterrows():
        pid = str(plant["plant_id"])
        pred_df = preds.get(pid, pd.DataFrame())
        ev = matched[matched["assigned_plant_id"] == pid].copy() if not matched.empty else pd.DataFrame()
        plot_plant_timeline(plant["name"], pred_df, ev, plots_dir / f"plant_{pid}_timeline.png")
        ev_sens = ev[ev["within_sensitivity_window"]].copy() if not ev.empty else pd.DataFrame()
        if not ev_sens.empty:
            plot_event_windows(
                plant["name"], pred_df, ev_sens, plots_dir / f"plant_{pid}_event_windows.png", args.window_days
            )

    plot_event_count_bars(summary_df, plots_dir / "aoi_event_coverage.png")
    md = build_markdown(summary_df, matched)
    (outdir / "aoi_event_validation_overview.md").write_text(md, encoding="utf-8")
    print(f"[ok] wrote AOI validation tables and plots under {outdir}")


if __name__ == "__main__":
    main()
