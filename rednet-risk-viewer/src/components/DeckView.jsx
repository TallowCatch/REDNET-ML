import { useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { TileLayer } from '@deck.gl/geo-layers';
import { BitmapLayer, GeoJsonLayer, ScatterplotLayer } from '@deck.gl/layers';

function scoreToColor(score, thresholds, alpha = 180) {
  const s = Number(score);
  if (!Number.isFinite(s)) return [148, 163, 184, alpha];
  if (s >= thresholds.action) return [220, 38, 38, alpha];
  if (s >= thresholds.watch) return [217, 119, 6, alpha];
  return [22, 163, 74, alpha];
}

function statusToColor(status, alpha = 230) {
  if (status === 'action') return [220, 38, 38, alpha];
  if (status === 'watch') return [217, 119, 6, alpha];
  if (status === 'normal') return [22, 163, 74, alpha];
  return [148, 163, 184, alpha];
}

export default function DeckView({
  plants,
  selectedPlantId,
  onPlantClick,
  viewState,
  onViewStateChange,
  thresholds,
  layerToggles,
  aoi,
  overlay,
  overlayScore,
  ociSurface,
  ociSurfaceOpacity = 0.42,
}) {
  const layers = useMemo(() => {
    const out = [
      new TileLayer({
        id: 'base-map',
        data: 'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
        minZoom: 0,
        maxZoom: 19,
        tileSize: 256,
        renderSubLayers: (props) => {
          const { west, south, east, north } = props.tile.bbox;
          return new BitmapLayer({
            id: `${props.id}-bitmap`,
            image: props.data,
            bounds: [west, south, east, north],
            data: null,
            pickable: false,
          });
        },
      }),
    ];

    if (ociSurface?.image && Array.isArray(ociSurface?.bounds) && ociSurface.bounds.length === 4) {
      out.push(
        new BitmapLayer({
          id: `oci-surface-${ociSurface.image}`,
          image: ociSurface.image,
          bounds: ociSurface.bounds,
          opacity: Number.isFinite(Number(ociSurfaceOpacity)) ? Number(ociSurfaceOpacity) : 0.42,
          pickable: false,
        })
      );
    }

    if (layerToggles.overlay && overlay) {
      out.push(
        new GeoJsonLayer({
          id: 'chip-overlay',
          data: overlay,
          pickable: true,
          filled: true,
          stroked: true,
          getLineColor: [15, 23, 42, 140],
          getLineWidth: 1,
          lineWidthUnits: 'pixels',
          getFillColor: (f) => scoreToColor(f?.properties?.[overlayScore], thresholds, 120),
        })
      );
    }

    if (layerToggles.aoi && aoi) {
      out.push(
        new GeoJsonLayer({
          id: 'plant-aoi',
          data: aoi,
          filled: true,
          stroked: true,
          pickable: true,
          getLineColor: [30, 64, 175, 180],
          getFillColor: [59, 130, 246, 35],
          getLineWidth: 1.5,
          lineWidthUnits: 'pixels',
        })
      );
    }

    out.push(
      new ScatterplotLayer({
        id: 'plants',
        data: plants,
        pickable: true,
        radiusUnits: 'meters',
        getRadius: (d) => (d.id === selectedPlantId ? 8200 : 5600),
        getPosition: (d) => [Number(d.lon), Number(d.lat)],
        getFillColor: (d) => statusToColor(d._status, 230),
        getLineColor: [255, 255, 255, 220],
        getLineWidth: 1,
        stroked: true,
        onClick: (info) => {
          if (info.object) onPlantClick?.(info.object.id);
        },
      })
    );

    return out;
  }, [
    plants,
    selectedPlantId,
    onPlantClick,
    thresholds,
    layerToggles,
    aoi,
    overlay,
    overlayScore,
    ociSurface,
    ociSurfaceOpacity,
  ]);

  return (
    <DeckGL
      viewState={viewState}
      controller
      layers={layers}
      onViewStateChange={({ viewState: next }) => onViewStateChange?.(next)}
      getTooltip={({ object }) => {
        if (!object) return null;

        if (object.name && object.latest) {
          const risk = Number(object._risk ?? object.latest?.ops_risk ?? object.latest?.hab_prob);
          return {
            text: `${object.name}\nRisk: ${Number.isFinite(risk) ? (risk * 100).toFixed(1) : '—'}%`,
          };
        }

        if (object.properties?.hab_prob != null) {
          const hp = Number(object.properties.hab_prob);
          return {
            text: `Chip risk: ${Number.isFinite(hp) ? (hp * 100).toFixed(1) : '—'}%`,
          };
        }

        return null;
      }}
      style={{ position: 'absolute', inset: 0 }}
    />
  );
}
