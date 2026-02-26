import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './App.css';

import DeckView from './components/DeckView';
import OceanDriversChart from './components/OceanDriversChart';
import TrendChart from './components/TrendChart';
import { loadOpsPayload } from './data/loadOpsPayload';

const FALLBACK_THRESHOLDS = {
  watch: 0.55,
  action: 0.6238688594003279,
  legacy_best_f1: 0.3926301481609915,
  legacy_default: 0.5327723842346281,
};

const DEFAULT_LAYER_TOGGLES = {
  aoi: true,
};

const MAP_INITIAL = {
  longitude: 58.8,
  latitude: 23.3,
  zoom: 6,
  pitch: 0,
  bearing: 0,
};

export default function App() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState(null);

  const [selectedPlantId, setSelectedPlantId] = useState(null);
  const [selectedMonth, setSelectedMonth] = useState('');
  const [layerToggles, setLayerToggles] = useState(DEFAULT_LAYER_TOGGLES);
  const [mapViewState, setMapViewState] = useState(MAP_INITIAL);
  const [ociPulse, setOciPulse] = useState(0);

  const [aoiGeojson, setAoiGeojson] = useState(null);
  const [overlayGeojson, setOverlayGeojson] = useState(null);

  const cacheRef = useRef(new Map());

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      try {
        const d = await loadOpsPayload();
        if (cancelled) return;
        setPayload(d);
        if (d?.plants?.length) {
          const first = d.plants[0];
          setSelectedPlantId(first.id);
          setSelectedMonth(latestMonthAcrossPlants(d.plants));
          setMapViewState((prev) => ({
            ...prev,
            longitude: Number(first.lon),
            latitude: Number(first.lat),
            zoom: 8.5,
            transitionDuration: 1000,
          }));
        }
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    }

    boot();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => {
      setOciPulse((x) => x + 0.32);
    }, 260);
    return () => window.clearInterval(id);
  }, []);

  const plants = useMemo(() => payload?.plants ?? [], [payload]);
  const thresholds = useMemo(() => payload?.thresholds ?? FALLBACK_THRESHOLDS, [payload]);

  const selectedPlant = useMemo(
    () => plants.find((p) => p.id === selectedPlantId) || null,
    [plants, selectedPlantId]
  );

  const monthlyDesc = useMemo(() => {
    if (!selectedPlant?.monthly) return [];
    return [...selectedPlant.monthly].sort((a, b) => String(b.month).localeCompare(String(a.month)));
  }, [selectedPlant]);

  const monthOptions = useMemo(() => {
    const set = new Set();
    plants.forEach((p) => {
      (p?.monthly || []).forEach((m) => {
        if (m?.month != null) set.add(String(m.month));
      });
    });
    return [...set].sort((a, b) => b.localeCompare(a));
  }, [plants]);

  const activeMonth =
    selectedMonth && monthOptions.includes(selectedMonth)
      ? selectedMonth
      : monthOptions[0] || '';

  const fleet = useMemo(() => {
    return [...plants]
      .map((p) => {
        const monthStats =
          activeMonth && Array.isArray(p?.monthly)
            ? p.monthly.find((m) => String(m.month) === String(activeMonth)) || null
            : null;
        const meanRisk = Number(monthStats?.mean);
        const p95Risk = Number(monthStats?.p95);
        const fallbackRisk = riskOf(p?.latest);
        const risk = Number.isFinite(meanRisk) ? meanRisk : fallbackRisk;
        const statusScore = Number.isFinite(p95Risk) ? p95Risk : risk;
        return {
          ...p,
          _risk: Number.isFinite(risk) ? risk : -1,
          _status: monthStats?.status || classify(statusScore, thresholds),
          _riskLabel: Number.isFinite(meanRisk) ? 'Avg Risk' : 'Risk',
          _period: monthStats?.month || null,
        };
      })
      .sort((a, b) => b._risk - a._risk);
  }, [plants, thresholds, activeMonth]);

  const handleSelectPlant = useCallback((plantId) => {
    const plant = plants.find((p) => p.id === plantId);
    if (!plant) return;

    setSelectedPlantId(plantId);
    setMapViewState((prev) => ({
      ...prev,
      longitude: Number(plant.lon),
      latitude: Number(plant.lat),
      zoom: 8.5,
      transitionDuration: 900,
    }));
  }, [plants]);

  async function loadJson(url, allowMissing = false) {
    if (!url) return null;
    if (cacheRef.current.has(url)) return cacheRef.current.get(url);

    const res = await fetch(url);
    if (!res.ok) {
      if (allowMissing && res.status === 404) {
        cacheRef.current.set(url, null);
        return null;
      }
      throw new Error(`Failed to fetch ${url}: ${res.status}`);
    }

    const data = await res.json();
    cacheRef.current.set(url, data);
    return data;
  }

  useEffect(() => {
    let cancelled = false;

    async function loadContextLayers() {
      if (!selectedPlant || !payload) {
        setAoiGeojson(null);
        return;
      }

      const pid = selectedPlant.id;
      const aoiUrl = payload.aoi_manifest?.[pid] || null;

      try {
        const aoi = await loadJson(aoiUrl, true);

        if (cancelled) return;
        setAoiGeojson(aoi);
      } catch (e) {
        if (cancelled) return;
        setError(String(e));
      }
    }

    loadContextLayers();
    return () => {
      cancelled = true;
    };
  }, [selectedPlant, payload]);

  useEffect(() => {
    let cancelled = false;

    async function loadOverlay() {
      setOverlayGeojson(null);

      if (!selectedPlant || !activeMonth) return;

      const url = `/overlays/osm_way_${selectedPlant.id}/${activeMonth}_tile_overlay.geojson`;

      try {
        const data = await loadJson(url, true);
        if (cancelled) return;
        setOverlayGeojson(data);
      } catch {
        if (cancelled) return;
        setOverlayGeojson(null);
      }
    }

    loadOverlay();
    return () => {
      cancelled = true;
    };
  }, [selectedPlant, activeMonth]);

  const trendSeries = useMemo(() => {
    if (!selectedPlant?.timeseries) return [];
    return selectedPlant.timeseries.slice(-50);
  }, [selectedPlant]);

  const topEvents = useMemo(() => {
    if (!selectedPlant?.timeseries || !activeMonth) return [];
    const monthly = selectedPlant.timeseries.filter((x) => x.month === activeMonth);
    const dedup = new Map();
    monthly.forEach((row) => {
      const key = `${row.scene_id || ''}|${row.datetime || ''}`;
      const score = riskOf(row);
      const prev = dedup.get(key);
      if (!prev || score > riskOf(prev)) {
        dedup.set(key, row);
      }
    });
    return [...dedup.values()]
      .sort((a, b) => riskOf(b) - riskOf(a))
      .slice(0, 10);
  }, [selectedPlant, activeMonth]);

  const latest = selectedPlant?.latest || null;
  const latestRisk = riskOf(latest);

  const activeMonthStats = useMemo(() => {
    if (!monthlyDesc.length || !activeMonth) return null;
    return monthlyDesc.find((m) => m.month === activeMonth) || null;
  }, [monthlyDesc, activeMonth]);

  const monthSeries = useMemo(() => {
    if (!selectedPlant?.timeseries || !activeMonth) return [];
    return selectedPlant.timeseries.filter((x) => x.month === activeMonth);
  }, [selectedPlant, activeMonth]);

  const displaySeries = useMemo(() => {
    if (monthSeries.length >= 2) return monthSeries;
    return trendSeries;
  }, [monthSeries, trendSeries]);

  const oceanSeries = displaySeries;

  const disagreementNow = useMemo(() => {
    const src = monthSeries.length ? monthSeries : latest ? [latest] : [];
    if (!src.length) return null;
    const vals = src
      .map((r) => {
        const hp = Number(r.hab_prob);
        const dets = [r.p_frcnn_r50_med, r.p_frcnn_mb_med, r.p_ssd_mb_med]
          .map(Number)
          .filter(Number.isFinite);
        if (!Number.isFinite(hp) || !dets.length) return null;
        const dm = dets.reduce((s, x) => s + x, 0) / dets.length;
        return Math.abs(hp - dm);
      })
      .filter(Number.isFinite);
    if (!vals.length) return null;
    return vals.reduce((s, x) => s + x, 0) / vals.length;
  }, [monthSeries, latest]);

  const statusNow =
    activeMonthStats?.status ||
    classify(activeMonthStats?.p95 ?? latestRisk, thresholds);
  const observationCount =
    monthSeries.length ||
    Number(activeMonthStats?.n) ||
    Number(selectedPlant?.summary?.n_obs) ||
    0;

  const driftOverall = payload?.drift_overall || null;
  const driftLevel = driftToLevel(driftOverall?.psi);
  const ociSurface = payload?.oci_surface_manifest?.[activeMonth] || null;
  const ociSurfaceOpacity = ociSurface ? (0.34 + 0.08 * Math.sin(ociPulse)) : 0;

  if (!payload && !error) {
    return <div className="loading-shell">Loading REDNET ops payload...</div>;
  }

  if (error) {
    return <div className="loading-shell error-shell">{error}</div>;
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>REDNET HAB Ops Console</h1>
          <p>Operational monitoring for desalination plant intake risk</p>
        </div>

        <div className="header-right">
          <div className={`drift-pill ${driftLevel}`}>
            Drift: {driftLabel(driftLevel)}
            {Number.isFinite(Number(driftOverall?.psi)) ? ` (PSI ${Number(driftOverall.psi).toFixed(3)})` : ''}
          </div>
          <div className="generated-at">Updated {fmtDate(payload.generated_at, true)}</div>
        </div>
      </header>

      <div className="app-grid">
        <aside className="fleet-panel">
          <h2>Fleet Risk</h2>
          <p className="panel-subtitle">Watch {pct(thresholds.watch)} | Action {pct(thresholds.action)}</p>

          <div className="fleet-list">
            {fleet.map((p) => (
              <button
                key={p.id}
                className={`fleet-item ${p.id === selectedPlantId ? 'active' : ''}`}
                onClick={() => handleSelectPlant(p.id)}
              >
                <div className="fleet-row">
                  <strong>{p.name}</strong>
                  <span className={`status-pill ${p._status}`}>{statusText(p._status)}</span>
                </div>
                <div className="fleet-meta">
                  <span>{p._riskLabel} {pct(p._risk)}</span>
                  <span>{p._period || fmtDate(p.latest?.datetime, false)}</span>
                </div>
              </button>
            ))}
          </div>
        </aside>

        <section className="map-panel">
          <div className="map-toolbar">
            <div className="toolbar-group">
              <label htmlFor="month-select">Month</label>
              <select
                id="month-select"
                value={activeMonth}
                onChange={(e) => setSelectedMonth(e.target.value)}
              >
                {monthOptions.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>

            <div className="toolbar-toggles">
              {Object.keys(layerToggles).map((k) => (
                <button
                  key={k}
                  className={`toggle ${layerToggles[k] ? 'on' : 'off'}`}
                  onClick={() =>
                    setLayerToggles((prev) => ({
                      ...prev,
                      [k]: !prev[k],
                    }))
                  }
                >
                  {k.toLowerCase() === 'aoi' ? 'AOI' : k}
                </button>
              ))}
            </div>
          </div>

          <div className="map-stage">
            <DeckView
              plants={fleet}
              selectedPlantId={selectedPlantId}
              onPlantClick={handleSelectPlant}
              viewState={mapViewState}
              onViewStateChange={setMapViewState}
              thresholds={thresholds}
              layerToggles={layerToggles}
              aoi={aoiGeojson}
              overlay={overlayGeojson}
              overlayScore="hab_prob"
              ociSurface={ociSurface}
              ociSurfaceOpacity={ociSurfaceOpacity}
            />
          </div>
        </section>

        <aside className="detail-panel">
          {!selectedPlant ? (
            <div className="empty-panel">Pick a plant from Fleet Risk.</div>
          ) : (
            <>
              <div className="plant-header">
                <h2>{selectedPlant.name}</h2>
                <span className={`status-pill ${statusNow}`}>{statusText(statusNow)}</span>
              </div>

              <div className="metric-grid">
                <MetricCard
                  label="Month P95"
                  hint="95th percentile of monthly operational risk (ops_risk)."
                  value={pct(activeMonthStats?.p95 ?? latestRisk)}
                />
                <MetricCard
                  label="Month Mean"
                  hint="Average monthly operational risk (ops_risk)."
                  value={pct(activeMonthStats?.mean ?? selectedPlant.summary?.last_30d_mean)}
                />
                <MetricCard
                  label="Disagreement"
                  hint="Absolute gap between fused model probability and detector consensus."
                  value={pct(disagreementNow)}
                />
                <MetricCard
                  label="Observations"
                  hint="Number of chip-level observations used in the selected month."
                  value={count(observationCount)}
                />
              </div>

              <section className="card">
                <div className="card-head">
                  <h3 className="title-with-info">
                    Risk Trend
                    <InfoIcon text="Operational risk time series (ops_risk) with watch/action thresholds." />
                  </h3>
                  <span>{monthSeries.length >= 2 ? `${activeMonth} observations` : `${trendSeries.length} most recent observations`}</span>
                </div>
                <TrendChart
                  series={displaySeries}
                  watch={thresholds.watch}
                  action={thresholds.action}
                />
              </section>

              <section className="card">
                <div className="card-head">
                  <h3 className="title-with-info">
                    Ocean Drivers (OCI Proxy)
                    <InfoIcon text="Normalized ocean condition indicators. OCI proxy is a weighted blend of SST, chlor-a, KD490, and NFLH." />
                  </h3>
                  <span>{oceanSeries.length} observations</span>
                </div>
                <OceanDriversChart series={oceanSeries} referenceSeries={selectedPlant?.timeseries || []} />
              </section>

              <section className="card">
                <div className="card-head">
                  <h3 className="title-with-info">
                    Monthly Risk Regime
                    <InfoIcon text="Month-level ops_risk distribution and exceedance rates vs watch/action thresholds." />
                  </h3>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Month</th>
                        <th>P95</th>
                        <th>Watch</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {monthlyDesc.slice(0, 8).map((m) => (
                        <tr key={m.month}>
                          <td>{m.month}</td>
                          <td>{pct(m.p95)}</td>
                          <td>{pct(m.watch_rate)}</td>
                          <td>{pct(m.action_rate)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="card">
                <div className="card-head">
                  <h3 className="title-with-info">
                    Top Events
                    <InfoIcon text="Highest-risk scene timestamps within the selected month." />
                  </h3>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Risk</th>
                        <th>SST</th>
                        <th>Chl-a</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topEvents.map((e, i) => (
                        <tr key={`${e.scene_id || i}-${i}`}>
                          <td>{fmtDate(e.datetime, false)}</td>
                          <td>{pct(riskOf(e))}</td>
                          <td>{num(e.sst, 2)}</td>
                          <td>{num(e.chlor_a, 3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="card">
                <div className="card-head">
                  <h3 className="title-with-info">
                    Drift Context
                    <InfoIcon text="Population shift diagnostics between train and current operating data (PSI/KS)." />
                  </h3>
                </div>
                <div className="drift-grid">
                  <div>
                    <span className="muted">Plant PSI</span>
                    <strong>{num(selectedPlant?.drift?.psi, 3)}</strong>
                  </div>
                  <div>
                    <span className="muted">Plant KS-D</span>
                    <strong>{num(selectedPlant?.drift?.ks_D, 3)}</strong>
                  </div>
                  <div>
                    <span className="muted">Overall PSI</span>
                    <strong>{num(driftOverall?.psi, 3)}</strong>
                  </div>
                  <div>
                    <span className="muted">Overall KS-D</span>
                    <strong>{num(driftOverall?.ks_D, 3)}</strong>
                  </div>
                </div>
              </section>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}

function MetricCard({ label, value, hint }) {
  return (
    <div className="metric-card">
      <span className="metric-label title-with-info">
        {label}
        {hint ? <InfoIcon text={hint} /> : null}
      </span>
      <strong className="metric-value">{value}</strong>
    </div>
  );
}

function InfoIcon({ text }) {
  return (
    <span className="info-wrap">
      <span className="info-icon" aria-label={text} tabIndex={0} role="img">
        <span className="info-icon-glyph" aria-hidden="true">i</span>
      </span>
      <span className="info-tooltip" role="tooltip">
        {text}
      </span>
    </span>
  );
}

function classify(score, thresholds) {
  const s = Number(score);
  if (!Number.isFinite(s)) return 'unknown';
  if (s >= thresholds.action) return 'action';
  if (s >= thresholds.watch) return 'watch';
  return 'normal';
}

function driftToLevel(psi) {
  const p = Number(psi);
  if (!Number.isFinite(p)) return 'unknown';
  if (p >= 0.5) return 'high';
  if (p >= 0.2) return 'moderate';
  return 'low';
}

function driftLabel(level) {
  if (level === 'high') return 'High shift';
  if (level === 'moderate') return 'Moderate shift';
  if (level === 'low') return 'Low shift';
  return 'Unknown';
}

function statusText(status) {
  if (status === 'action') return 'Action';
  if (status === 'watch') return 'Watch';
  if (status === 'normal') return 'Normal';
  return 'Unknown';
}

function riskOf(row) {
  const ops = Number(row?.ops_risk);
  if (Number.isFinite(ops)) return ops;
  const model = Number(row?.hab_prob);
  if (Number.isFinite(model)) return model;
  return NaN;
}

function pct(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return `${(n * 100).toFixed(1)}%`;
}

function num(v, digits = 2) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return n.toFixed(digits);
}

function count(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return String(Math.round(n));
}

function fmtDate(v, includeTime) {
  if (!v) return '—';
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return '—';
  const opts = includeTime
    ? {
        year: 'numeric',
        month: 'short',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        timeZone: 'UTC',
      }
    : {
        year: 'numeric',
        month: 'short',
        day: '2-digit',
      };
  return d.toLocaleString(undefined, opts);
}

function latestMonthAcrossPlants(plants) {
  const months = [];
  (plants || []).forEach((p) => {
    (p?.monthly || []).forEach((m) => {
      if (m?.month != null) months.push(String(m.month));
    });
  });
  if (!months.length) return '';
  months.sort((a, b) => b.localeCompare(a));
  return months[0];
}
