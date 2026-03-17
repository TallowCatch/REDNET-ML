# Event Validation

- Primary matched rows: 4
- Positive primary events: 4
- Negative primary events: 0
- Primary score column: matched_ops_risk
- Event hit rate at WATCH: 0.5
- Event hit rate at ACTION: 0.0
- AUROC: None
- AUPRC: None

Primary external event validation uses matched plant-date windows only. This is event-based external concordance, not in-situ validation. Sample size or class balance was insufficient for stable AUROC/AUPRC.

Use this as supplementary external validation only. Do not call it in-situ validation.

Key outputs:
- `top_ranked_event_matches.csv`
- `event_lead_lag_table.csv`
- `event_vs_nonevent_score.png` when enough event and non-event windows exist
