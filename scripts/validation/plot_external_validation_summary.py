from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path('/Users/ameerfiras/REDNET-ML')
SUMMARY_PATH = ROOT / 'runs/eval/external_validation/event_validation_summary.json'
OUTPUT_PATH = ROOT / 'runs/eval/external_validation/plots/external_validation_strict_summary.png'


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    auroc = summary['auroc']
    auprc = summary['auprc']
    event_median = summary['median_positive_score']
    nonevent_median = summary['median_negative_score']
    n = summary['n']
    n_pos = summary['n_positive_primary']
    n_neg = summary['n_negative_primary']

    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)

    metrics = ['AUROC', 'AUPRC']
    metric_vals = [auroc, auprc]
    metric_colors = ['#1f77b4', '#ff7f0e']
    bars = axes[0].bar(metrics, metric_vals, color=metric_colors, width=0.55)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel('score')
    axes[0].set_title('Strict external concordance')
    axes[0].grid(axis='y', alpha=0.2)
    for bar, value in zip(bars, metric_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.03, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    axes[0].text(0.5, -0.22, f'matched subset n={n}', transform=axes[0].transAxes, ha='center', va='top', fontsize=10)

    groups = ['event', 'non-event']
    median_vals = [event_median, nonevent_median]
    median_colors = ['#d62728', '#7f7f7f']
    bars = axes[1].bar(groups, median_vals, color=median_colors, width=0.55)
    axes[1].set_ylim(0, 0.8)
    axes[1].set_ylabel('median ops_risk')
    axes[1].set_title('Matched-window median risk')
    axes[1].grid(axis='y', alpha=0.2)
    for bar, value in zip(bars, median_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.025, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    axes[1].text(0.5, -0.22, f'positives={n_pos}, negatives={n_neg}', transform=axes[1].transAxes, ha='center', va='top', fontsize=10)

    fig.suptitle('REDNET-ML strict external validation summary', fontsize=14)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
