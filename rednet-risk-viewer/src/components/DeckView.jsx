import DeckGL from "@deck.gl/react";
import { ScatterplotLayer } from "@deck.gl/layers";
import { TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer } from "@deck.gl/layers";
import { useMemo } from "react";

export default function DeckView({ plant, risk }) {
  const layers = useMemo(() => {
    const baseMap = new TileLayer({
      id: "osm-basemap",
      data: "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
      minZoom: 0,
      maxZoom: 19,
      tileSize: 256,
      renderSubLayers: (props) => {
        const {
          bbox: { west, south, east, north },
        } = props.tile;

        return new BitmapLayer(props, {
          data: null,
          image: props.data,
          bounds: [west, south, east, north],
        });
      },
    });

    if (!plant) return [baseMap];

    const color =
      risk < 0.33
        ? [34, 197, 94]   // green
        : risk < 0.66
        ? [245, 158, 11]  // amber
        : [239, 68, 68];  // red

    const plantLayer = new ScatterplotLayer({
      id: "plant-risk",
      data: [plant],
      getPosition: (d) => [d.lon, d.lat],
      getRadius: 400 + risk * 1200,
      radiusUnits: "meters",
      getFillColor: color,
      opacity: 0.85,
      stroked: false,
      pickable: false,
    });

    return [baseMap, plantLayer];
  }, [plant, risk]);

  return (
    <DeckGL
      initialViewState={{
        longitude: plant?.lon ?? 59.5,
        latitude: plant?.lat ?? 22.0,
        zoom: 7,
        pitch: 0,
        bearing: 0,
      }}
      controller={true}
      layers={layers}
      style={{ position: "absolute", inset: 0 }}
    />
  );
}
