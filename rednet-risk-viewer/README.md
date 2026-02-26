# REDNET HAB Ops Console

This app is now an operations-oriented frontend for desalination intake monitoring. It is built around a generated payload (`public/ops/ops_payload.json`) that combines:

- per-plant inference time series (`deployment/outputs/by_plant/*/inference_all_months.csv`)
- drift diagnostics (`runs/eval/generalization/generalization_2025_watch055/*`)
- transport layers (density/envelope/trips GeoJSON)
- AOI GeoJSON per plant

## What Changed

The old viewer replayed a single `hab_prob` stream with a global time index. The new console now provides:

- Fleet risk board with watch/action status per plant
- Unified threshold policy (watch/action) across map and panels
- Drift context (overall + plant-level PSI/KS)
- Plant-level deep dive:
  - latest risk + cadence + disagreement
  - trend chart with watch/action lines
  - monthly regime table
  - top risk events
- Map layers:
  - plant markers
  - AOI polygons
  - transport density/envelope/trips
  - chip overlay if a month-level overlay file exists

## Thresholds Used

These are embedded in the generated payload:

- Watch: `0.55`
- Action: `0.6238688594003279`
- Legacy refs preserved: `0.3926301481609915`, `0.5327723842346281`

## Build Data Payload

From `rednet-risk-viewer/`:

```bash
npm run build:ops
```

This runs:

```bash
python ../scripts/viewer/build_ops_payload.py --repo_root ..
```

Outputs are written under:

- `public/ops/ops_payload.json`
- `public/ops/transport/...`
- `public/ops/aoi/...`

## Run the Frontend

```bash
npm install
npm run dev
```

For production build:

```bash
npm run build
```

## Data Notes

- Chip overlays are loaded from `/public/overlays/osm_way_<id>/<YYYY-MM>_tile_overlay.geojson`.
- If an overlay does not exist for the selected plant/month, the app keeps transport + AOI + plant layers active and shows a fallback note.
- Transport layers are copied from deployment outputs during `build:ops`.

## Primary Files

- App shell and state orchestration: `src/App.jsx`
- Map rendering and DeckGL layers: `src/components/DeckView.jsx`
- Trend chart: `src/components/TrendChart.jsx`
- Payload loader: `src/data/loadOpsPayload.js`
- Payload generation script: `../scripts/viewer/build_ops_payload.py`
