import DeckGL from "@deck.gl/react";
import {
  ScatterplotLayer,
  BitmapLayer,
  SolidPolygonLayer,
} from "@deck.gl/layers";
import { TileLayer } from "@deck.gl/geo-layers";
import { FlyToInterpolator } from "@deck.gl/core";
import { useMemo, useState, useEffect } from "react";
import { load } from '@loaders.gl/core';
import { ImageLoader } from '@loaders.gl/images';

const DEFAULT_THRESHOLD = 0.5327723842346281;
const BEST_F1_THRESHOLD = 0.3926301481609915;

// ---------------- COLORS ----------------
function riskToColor(risk, alpha = 160) {
  if (risk < BEST_F1_THRESHOLD) return [34, 197, 94, alpha];
  if (risk < DEFAULT_THRESHOLD) return [245, 158, 11, alpha];
  return [239, 68, 68, alpha];
}

// ---------------- UNCERTAINTY HEURISTIC ----------------
function ringRadii(mean, low, high) {
  const base = 8000;
  const alpha = 25000;
  const beta = 40000;

  const severity = mean / DEFAULT_THRESHOLD;
  const uncertainty = Math.max(0, high - low);

  const inner = base + severity * alpha;
  const outer = inner + uncertainty * beta;

  return { inner, outer };
}

// ---------------- GEOMETRY ----------------
function makeAnnulus([lon, lat], innerR, outerR, steps = 96) {
  const R = 6378137;
  const toRad = (d) => (d * Math.PI) / 180;
  const toDeg = (r) => (r * 180) / Math.PI;

  const ring = (radius) =>
    Array.from({ length: steps + 1 }, (_, i) => {
      const a = (i / steps) * 2 * Math.PI;
      return [
        lon + toDeg((radius * Math.cos(a)) / (R * Math.cos(toRad(lat)))),
        lat + toDeg((radius * Math.sin(a)) / R),
      ];
    });

  return [ring(outerR), ring(innerR).reverse()];
}

export default function DeckView({
  plants,
  focusedPlantId,
  onPlantClick,
  forecast,
  showPulseRing = true,
  showAnnulus = false,
  aoi,
}) {
  const [hover, setHover] = useState(null);

  // 🔁 Pulse animation
  const [pulse, setPulse] = useState(0);
  useEffect(() => {
    let raf;
    const tick = () => {
      setPulse((p) => (p + 0.015) % 1);
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, []);

  const focusedPlant = useMemo(
    () => plants?.find((p) => p.id === focusedPlantId),
    [plants, focusedPlantId]
  );

  // ---------------- VIEW STATE (FREE PAN / ZOOM) ----------------
  const [viewState, setViewState] = useState({
    longitude: 59.5,
    latitude: 22.5,
    zoom: 6,
    pitch: 0,
    bearing: 0,
  });

  // Zoom ONLY when plant changes
  useEffect(() => {
    if (!focusedPlant) return;
    setViewState((v) => ({
      ...v,
      longitude: Number(focusedPlant.lon),
      latitude: Number(focusedPlant.lat),
      zoom: 8,
      transitionDuration: 1200,
      transitionInterpolator: new FlyToInterpolator(),
    }));
  }, [focusedPlant]);

  // ---------------- ANNULUS ----------------
  const annulus = useMemo(() => {
    if (!focusedPlant || !forecast?.preds?.length) return null;
    const { mean, low, high } = forecast.preds[0];
    const { inner, outer } = ringRadii(mean, low, high);
    return makeAnnulus(
      [Number(focusedPlant.lon), Number(focusedPlant.lat)],
      inner,
      outer
    );
  }, [focusedPlant, forecast]);

  // ---------------- LAYERS ----------------
  const layers = useMemo(() => {
    const layers = [
      // 🌍 Basemap
      new TileLayer({
        id: 'osm',
        data: 'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png',
        tileSize: 256,
      
        // 🔑 THIS IS THE MISSING PIECE
        getTileData: ({ url }) =>
          load(url, ImageLoader, { imagebitmap: true }),
      
        renderSubLayers: (props) => {
          const { west, south, east, north } = props.tile.bbox;
      
          return new BitmapLayer(props, {
            image: props.data, // now GUARANTEED ImageBitmap
            bounds: [west, south, east, north],
          });
        },
      }),      

      // 🟢 Plant core
      new ScatterplotLayer({
        id: "plant-core",
        data: plants ?? [],
        getPosition: (d) => [Number(d.lon), Number(d.lat)],
        getRadius: 5000,
        radiusUnits: "meters",
        getFillColor: (d) => riskToColor(d.currentRisk, 220),
        pickable: true,
        onHover: (info) =>
          setHover(info.object ? { x: info.x, y: info.y, plant: info.object } : null),
        onClick: (info) =>
          info.object && onPlantClick?.(info.object.id),
      }),
    ];

    // 🔴 Pulsing ring (OPTIONAL)
    if (showPulseRing) {
      layers.push(
        new ScatterplotLayer({
          id: "risk-pulse",
          data: plants ?? [],
          getPosition: (d) => [Number(d.lon), Number(d.lat)],
          getRadius: (d) =>
            12000 +
            (d.currentRisk / DEFAULT_THRESHOLD) * 25000 +
            pulse * 6000,
          radiusUnits: "meters",
          stroked: true,
          filled: false,
          getLineColor: (d) => riskToColor(d.currentRisk, 220),
          getLineWidth: 2,
          lineWidthUnits: "pixels",
          opacity: 0.5,
          parameters: { depthTest: false },
        })
      );
    }

    // 🟡 Scientific annulus (OPTIONAL, FIXED)
    if (showAnnulus && annulus) {
      layers.push(
        new SolidPolygonLayer({
          id: "risk-annulus",
          data: [{ polygon: annulus }],
          getPolygon: (d) => d.polygon,
          getFillColor: riskToColor(forecast.preds[0].mean, 80),
          stroked: false,
          filled: true,
          parameters: { depthTest: false },
        })
      );
    }

    // 🟦 AOI
    if (aoi) {
      layers.push(
        new SolidPolygonLayer({
          id: "aoi",
          data: aoi.features,
          getPolygon: (f) => f.geometry.coordinates,
          getFillColor: [59, 130, 246, 40],
          getLineColor: [59, 130, 246, 160],
          stroked: true,
          filled: true,
        })
      );
    }

    return layers;
  }, [plants, pulse, annulus, forecast, showPulseRing, showAnnulus, aoi]);

  // ---------------- RENDER ----------------
  return (
    <>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState }) => setViewState(viewState)}
        controller
        layers={layers}
        style={{
            position: "absolute",
            inset: 0,
        }}
      />

      {hover && (
        <div
          style={{
            position: "absolute",
            left: hover.x + 10,
            top: hover.y + 10,
            background: "rgba(15,23,42,0.85)",
            color: "white",
            padding: "8px 10px",
            borderRadius: 10,
            fontSize: 12,
            pointerEvents: "none",
          }}
        >
          <b>{hover.plant.name}</b>
          <div>
            Risk {(hover.plant.currentRisk * 100).toFixed(1)}%
          </div>
        </div>
      )}
    </>
  );
}
