import DeckGL from "@deck.gl/react";
import { ScatterplotLayer, BitmapLayer } from "@deck.gl/layers";
import { TileLayer } from "@deck.gl/geo-layers";
import { useMemo } from "react";

function riskToColor(risk) {
  if (risk < 0.33) return [34, 197, 94];
  if (risk < 0.66) return [245, 158, 11];
  return [239, 68, 68];
}

export default function DeckView({ plants }) {
  const layers = useMemo(() => [
    new TileLayer({
      id: "osm",
      data: "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
      minZoom: 0,
      maxZoom: 19,
      tileSize: 256,

      // ✅ THIS WAS MISSING
      getTileData: ({ url }) => url,

      renderSubLayers: (props) => {
        const {
          tile: {
            bbox: { west, south, east, north },
          },
        } = props;

        return new BitmapLayer(props, {
          image: props.data, // ✅ now this is a URL
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
    }),
  ], [plants]);

  return (
    <DeckGL
      viewState={{ longitude: 59.5, latitude: 22.5, zoom: 6 }}
      controller
      layers={layers}
      style={{ position: "absolute", inset: 0 }}
    />
  );
}
