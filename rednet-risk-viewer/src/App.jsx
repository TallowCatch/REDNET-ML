import { useEffect, useMemo, useState } from "react";
import "./App.css";

import { loadPlants } from "./data/loadPlants";
import { loadPlantRisk } from "./data/loadPlantRisk";
import { trendForecast } from "./forecast/trendForecast";
import DeckView from "./components/DeckView";

const ALERTS = [
  { name: "Low", min: 0.0, max: 0.33, color: "#22c55e" },
  { name: "Moderate", min: 0.33, max: 0.66, color: "#f59e0b" },
  { name: "High", min: 0.66, max: 1.01, color: "#ef4444" },
];

export default function App() {
  const [plants, setPlants] = useState([]);
  const [plantId, setPlantId] = useState(null);
  const [timeIndex, setTimeIndex] = useState(0);
  const [playing, setPlaying] = useState(false);

  // 🔘 UI toggles
  const [showPulseRing, setShowPulseRing] = useState(true);
  const [showAnnulus, setShowAnnulus] = useState(false);
  const [widgetOpen, setWidgetOpen] = useState(true);

  // ---------------- LOAD DATA ----------------
  useEffect(() => {
    async function loadAll() {
      const basePlants = await loadPlants();

      const withRisk = await Promise.all(
        basePlants.map(async (p) => {
          const series = await loadPlantRisk(p.csv);
          return { ...p, riskSeries: series };
        })
      );

      setPlants(withRisk);
      setPlantId(withRisk?.[0]?.id ?? null);

      const maxLen = Math.max(
        ...withRisk.map((p) => p.riskSeries.length)
      );
      setTimeIndex(Math.max(0, maxLen - 1));
    }

    loadAll();
  }, []);

  // ---------------- TIME PLAYBACK ----------------
  useEffect(() => {
    if (!playing || plants.length === 0) return;

    const maxIndex =
      Math.max(...plants.map((p) => p.riskSeries.length)) - 1;

    const id = setInterval(() => {
      setTimeIndex((t) => {
        if (t >= maxIndex) {
          setPlaying(false);
          return t;
        }
        return t + 1;
      });
    }, 800);

    return () => clearInterval(id);
  }, [playing, plants]);

  // ---------------- DERIVED DATA ----------------
  const plantsAtTime = useMemo(
    () =>
      plants.map((p) => ({
        ...p,
        currentRisk: p.riskSeries?.[timeIndex]?.risk ?? 0,
      })),
    [plants, timeIndex]
  );

  const focusedPlant = useMemo(
    () => plants.find((p) => p.id === plantId) || null,
    [plants, plantId]
  );

  const focusedSeries = focusedPlant?.riskSeries ?? [];
  const current = focusedSeries?.[timeIndex] ?? null;
  const currentRisk = current?.risk ?? 0;
  const alertBand = getAlertBand(currentRisk);

  const forecast = useMemo(() => {
    if (!focusedSeries.length) return null;
    return trendForecast(focusedSeries, 72, 12);
  }, [focusedSeries]);

  // ---------------- RENDER ----------------
  return (
    <div style={styles.shell}>
     <div style={styles.map}>
      <DeckView
        plants={plantsAtTime}
        focusedPlantId={plantId}
        onPlantClick={setPlantId}
        forecast={forecast}
        showPulseRing={showPulseRing}
        showAnnulus={showAnnulus}
      />
      </div>

      {/* TOP BAR */}
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

      {/* LEGEND / TOGGLES */}
      <div style={styles.legend}>
        <LegendToggle
          active={showPulseRing}
          color="#ef4444"
          label="Pulse Ring"
          onClick={() => setShowPulseRing((v) => !v)}
        />
        <LegendToggle
          active={showAnnulus}
          color="#f59e0b"
          label="Risk Envelope"
          onClick={() => setShowAnnulus((v) => !v)}
        />
      </div>

      {/* BOTTOM WIDGET */}
      <div
        style={{
          ...styles.bottomCard,
          height: widgetOpen ? "auto" : 42,
        }}
      >
        {/* HEADER / ARROW */}
        <div style={styles.widgetHeader}>
          <div style={{ fontWeight: 800 }}>
            {alertBand.name} Risk — {(currentRisk * 100).toFixed(1)}%
          </div>
          <button
            onClick={() => setWidgetOpen((v) => !v)}
            style={styles.collapseBtn}
          >
            {widgetOpen ? "▾" : "▴"}
          </button>
        </div>

        {widgetOpen && (
          <>
            <div style={styles.metaRow}>
              <div style={styles.smallLabel}>Timestamp</div>
              <div style={styles.smallValue}>
                {current ? current.time.toDateString() : "—"}
              </div>
            </div>

            {plants.length > 0 && (
              <input
                type="range"
                min={0}
                max={
                  Math.max(
                    ...plants.map((p) => p.riskSeries.length)
                  ) - 1
                }
                value={timeIndex}
                onChange={(e) => {
                  setPlaying(false);
                  setTimeIndex(Number(e.target.value));
                }}
                style={styles.slider}
              />
            )}

            <button
              onClick={() => setPlaying((p) => !p)}
              style={styles.play(playing)}
            >
              {playing ? "⏸ Pause" : "▶ Play"}
            </button>
          </>
        )}
      </div>
    </div>
  );
}

// ---------------- HELPERS ----------------
function getAlertBand(r) {
  return (
    ALERTS.find((a) => r >= a.min && r < a.max) ??
    ALERTS[0]
  );
}

// ---------------- UI COMPONENTS ----------------
function LegendToggle({ active, color, label, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "6px 12px",
        borderRadius: 999,
        border: `1px solid ${color}`,
        background: active ? `${color}33` : "rgba(15,23,42,0.7)",
        color: "white",
        fontWeight: 700,
        cursor: "pointer",
        fontSize: 12,
      }}
    >
      {label}
    </button>
  );
}

// ---------------- STYLES ----------------
const styles = {
  shell: {
    position: "fixed",
    inset: 0,
  },

  topBar: {
    position: "absolute",
    top: 14,
    left: 14,
    right: 14,
    display: "flex",
    justifyContent: "space-between",
    gap: 12,
    pointerEvents: "auto",
    zIndex: 10,
  },

  title: {
    padding: "8px 12px",
    background: "rgba(15,23,42,0.75)",
    borderRadius: 12,
    fontWeight: 800,
    color: "white",
  },

  select: {
    padding: "8px 12px",
    background: "rgba(15,23,42,0.75)",
    borderRadius: 12,
    color: "white",
  },

  legend: {
    position: "absolute",
    top: 60,
    left: 14,
    display: "flex",
    gap: 8,
    zIndex: 10,
  },

  bottomCard: {
    position: "absolute",
    left: 14,
    right: 14,
    bottom: 10,
    background: "rgba(15,23,42,0.85)",
    borderRadius: 14,
    color: "white",
    padding: 10,
    zIndex: 10,
  },

  map: {
    position: "absolute",
    inset: 0,
  },

  widgetHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },

  collapseBtn: {
    background: "transparent",
    border: "none",
    color: "white",
    fontSize: 18,
    cursor: "pointer",
  },

  metaRow: {
    marginTop: 6,
    display: "flex",
    justifyContent: "space-between",
  },

  smallLabel: { fontSize: 11, opacity: 0.7 },
  smallValue: { fontSize: 12, fontWeight: 600 },

  slider: { width: "100%", marginTop: 8 },

  play: (playing) => ({
    marginTop: 6,
    padding: "6px 12px",
    borderRadius: 999,
    border: "1px solid rgba(255,255,255,0.25)",
    background: playing
      ? "rgba(239,68,68,0.2)"
      : "rgba(255,255,255,0.1)",
    color: "white",
    fontWeight: 700,
    cursor: "pointer",
  }),
};
