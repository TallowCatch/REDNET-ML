import { useEffect, useMemo, useState } from "react";
import "./App.css";

import { loadPlants } from "./data/loadPlants";
import { loadPlantRisk } from "./data/loadPlantRisk";
import { trendForecast } from "./forecast/trendForecast";
import MapView from "./components/DeckView";
import DeckView from "./components/DeckView";

const ALERTS = [
  { name: "Low", min: 0.0, max: 0.33, color: "#22c55e" },
  { name: "Moderate", min: 0.33, max: 0.66, color: "#f59e0b" },
  { name: "High", min: 0.66, max: 1.01, color: "#ef4444" },
];

export default function App() {
  const [plants, setPlants] = useState([]);
  const [plantId, setPlantId] = useState(null);

  const [riskSeries, setRiskSeries] = useState([]);
  const [timeIndex, setTimeIndex] = useState(0);

  const plant = useMemo(
    () => plants.find((p) => p.id === plantId) || null,
    [plants, plantId]
  );

  // Load plant registry
  useEffect(() => {
    loadPlants().then((ps) => {
      setPlants(ps);
      setPlantId(ps?.[0]?.id ?? null);
    });
  }, []);

  // Load selected plant CSV
  useEffect(() => {
    if (!plant) return;
    loadPlantRisk(plant.csv).then((data) => {
      setRiskSeries(data);
      setTimeIndex(Math.max(0, data.length - 1)); // default to latest
    });
  }, [plant?.id]);

  const current = riskSeries[timeIndex] || null;
  const currentRisk = current?.risk ?? 0;

  const alertBand = useMemo(() => getAlertBand(currentRisk), [currentRisk]);

  // Option 1 forecast
  const forecast = useMemo(() => {
    if (!riskSeries.length) return null;
    return trendForecast(riskSeries, 72, 12);
  }, [riskSeries]);

  return (
    <div style={styles.shell}>
      <div style={styles.map}>
        {plant && <DeckView plant={plant} risk={currentRisk} />}
        <div style={styles.topBar}>
          <div style={styles.title}>REDNET — HAB Risk Monitor</div>
          <select
            value={plantId ?? ""}
            onChange={(e) => setPlantId(e.target.value)}
            style={styles.select}
          >
            {plants.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div style={styles.bottomCard}>
        <div style={styles.row}>
          <div>
            <div style={styles.big}>
              {alertBand.name} Risk
            </div>
            <div style={{ ...styles.badge, borderColor: alertBand.color }}>
              {(currentRisk * 100).toFixed(1)}%
            </div>
          </div>

          <div style={styles.meta}>
            <div style={styles.smallLabel}>Timestamp</div>
            <div style={styles.smallValue}>
              {current ? current.time.toDateString() : "—"}
            </div>
          </div>
        </div>

        {riskSeries.length > 0 && (
          <input
            type="range"
            min={0}
            max={riskSeries.length - 1}
            value={timeIndex}
            onChange={(e) => setTimeIndex(Number(e.target.value))}
            style={styles.slider}
          />
        )}

        <div style={styles.forecastBox}>
          <div style={styles.smallLabel}>Next 72h (trend forecast envelope)</div>
          {forecast?.preds?.length ? (
            <div style={{ display: "grid", gap: 8 }}>
              {forecast.preds.map((p) => (
                <div key={p.time.toISOString()} style={styles.forecastRow}>
                  <div style={{ width: 110 }}>
                    {p.time.toDateString().slice(4)}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div>
                      mean: <b>{(p.mean * 100).toFixed(1)}%</b> &nbsp;|&nbsp;
                      low: {(p.low * 100).toFixed(1)}% &nbsp;|&nbsp;
                      high: {(p.high * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div style={styles.pill(getAlertBand(p.mean).color)}>
                    {getAlertBand(p.mean).name}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ opacity: 0.75 }}>Not enough history yet.</div>
          )}
        </div>
      </div>
    </div>
  );
}

function getAlertBand(r) {
  return (
    ALERTS.find((a) => r >= a.min && r < a.max) ?? ALERTS[0]
  );
}

const styles = {
  shell: { height: "100vh", width: "100vw", position: "relative" },
  map: { height: "100%", width: "100%" },
  topBar: {
    position: "absolute",
    top: 16,
    left: 16,
    right: 16,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 12,
    pointerEvents: "auto",
  },
  title: {
    padding: "10px 12px",
    background: "rgba(15, 23, 42, 0.75)",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 14,
    color: "white",
    fontWeight: 700,
  },
  select: {
    padding: "10px 12px",
    background: "rgba(15, 23, 42, 0.75)",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 14,
    color: "white",
    outline: "none",
  },
  bottomCard: {
    position: "absolute",
    left: 16,
    right: 16,
    bottom: 16,
    padding: 16,
    background: "rgba(15, 23, 42, 0.78)",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 18,
    color: "white",
    backdropFilter: "blur(10px)",
  },
  row: { display: "flex", justifyContent: "space-between", gap: 16 },
  big: { fontSize: 18, fontWeight: 800 },
  badge: {
    marginTop: 6,
    display: "inline-block",
    padding: "6px 10px",
    borderRadius: 999,
    border: "2px solid",
    fontWeight: 800,
  },
  meta: { textAlign: "right", opacity: 0.9 },
  smallLabel: { fontSize: 12, opacity: 0.8 },
  smallValue: { fontSize: 14, fontWeight: 600 },
  slider: { width: "100%", marginTop: 14 },
  forecastBox: {
    marginTop: 14,
    paddingTop: 10,
    borderTop: "1px solid rgba(255,255,255,0.12)",
  },
  forecastRow: {
    display: "flex",
    gap: 12,
    alignItems: "center",
    fontSize: 13,
  },
  pill: (borderColor) => ({
    padding: "4px 10px",
    borderRadius: 999,
    border: `1px solid ${borderColor}`,
    fontWeight: 700,
    fontSize: 12,
  }),
};
