import DeckGL from "@deck.gl/react";
import { ScatterplotLayer, BitmapLayer } from "@deck.gl/layers";
import { TileLayer } from "@deck.gl/geo-layers";
import { FlyToInterpolator } from "@deck.gl/core";
import { useMemo, useState } from "react";

function riskToColor(risk) {
  if (risk < 0.33) return [34, 197, 94];
  if (risk < 0.66) return [245, 158, 11];
  return [239, 68, 68];
}

export default function DeckView({
  plants,
  focusedPlantId,
  onPlantClick,
}) {
  const [hover, setHover] = useState(null);

  // ---------------- MAP VIEW STATE ----------------
  const viewState = useMemo(() => {
    const focused = plants?.find(
      (p) => p.id === focusedPlantId
    );

    if (!focused) {
      return {
        longitude: 59.5,
        latitude: 22.5,
        zoom: 6,
      };
    }

    return {
      longitude: Number(focused.lon),
      latitude: Number(focused.lat),
      zoom: 8,
      transitionDuration: 800,
      transitionInterpolator: new FlyToInterpolator(),
    };
  }, [plants, focusedPlantId]);

  // ---------------- LAYERS ----------------
  const layers = useMemo(() => [
    new TileLayer({
      id: "osm",
      data: "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
      minZoom: 0,
      maxZoom: 19,
      tileSize: 256,
      getTileData: ({ url }) => url,

      renderSubLayers: (props) => {
        const {
          tile: {
            bbox: { west, south, east, north },
          },
        } = props;

        return new BitmapLayer(props, {
          image: props.data,
          bounds: [west, south, east, north],
        });
      },
    }),

    new ScatterplotLayer({
      id: "plants",
      data: plants ?? [],
      getPosition: (d) => [Number(d.lon), Number(d.lat)],
      getRadius: 6000,
      radiusUnits: "meters",
      getFillColor: (d) => riskToColor(d.currentRisk),
      opacity: 0.9,
      pickable: true,

      onHover: (info) => {
        if (info.object) {
          setHover({
            x: info.x,
            y: info.y,
            plant: info.object,
          });
        } else {
          setHover(null);
        }
      },

      onClick: (info) => {
        if (info.object && onPlantClick) {
          onPlantClick(info.object.id);
        }
      },
    }),
  ], [plants, onPlantClick]);

  // ---------------- RENDER ----------------
  return (
    <>
      <DeckGL
        viewState={viewState}
        controller
        layers={layers}
        style={{ position: "absolute", inset: 0 }}
      />

      {/* HOVER TOOLTIP */}
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
            whiteSpace: "nowrap",
          }}
        >
          <div style={{ fontWeight: 700 }}>
            {hover.plant.name}
          </div>
          <div>
            Risk: {(hover.plant.currentRisk * 100).toFixed(1)}%
          </div>
        </div>
      )}
    </>
  );
}
