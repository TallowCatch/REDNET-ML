# Event Validation

- Primary matched rows: 6
- Positive primary events: 2
- Negative primary events: 4
- Primary score column: matched_ops_risk
- Event hit rate at WATCH: 0.0
- Event hit rate at ACTION: 0.0
- AUROC: 0.875
- AUPRC: 0.8333333333333333

Primary external event validation uses matched plant-date windows only. This is event-based external concordance, not in-situ validation.

Use this as supplementary external validation only. Do not call it in-situ validation.

Key outputs:
- `top_ranked_event_matches.csv`
- `event_lead_lag_table.csv`
- `event_vs_nonevent_score.png` when enough event and non-event windows exist
