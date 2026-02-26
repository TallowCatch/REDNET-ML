function toY(v, min, max, h, pad) {
  if (!Number.isFinite(v)) return h - pad;
  if (max <= min) return h - pad;
  const t = (v - min) / (max - min);
  return h - pad - t * (h - 2 * pad);
}

function toX(i, n, w, pad) {
  if (n <= 1) return pad;
  return pad + (i / (n - 1)) * (w - 2 * pad);
}

function buildPath(values, w, h, pad, min, max) {
  let path = '';
  let drawing = false;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (!Number.isFinite(v)) {
      drawing = false;
      continue;
    }
    const x = toX(i, values.length, w, pad);
    const y = toY(v, min, max, h, pad);
    if (!drawing) {
      path += `M ${x} ${y}`;
      drawing = true;
    } else {
      path += ` L ${x} ${y}`;
    }
  }
  return path;
}

function quantile(sortedValues, q) {
  if (!sortedValues.length) return NaN;
  const pos = (sortedValues.length - 1) * q;
  const low = Math.floor(pos);
  const high = Math.ceil(pos);
  if (low === high) return sortedValues[low];
  const w = pos - low;
  return sortedValues[low] * (1 - w) + sortedValues[high] * w;
}

function robustNormalize(values, referenceValues) {
  const refFinite = referenceValues.filter(Number.isFinite).sort((a, b) => a - b);
  if (!refFinite.length) return values.map(() => NaN);

  let lo = quantile(refFinite, 0.1);
  let hi = quantile(refFinite, 0.9);
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) {
    lo = refFinite[0];
    hi = refFinite[refFinite.length - 1];
  }

  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) {
    return values.map((v) => (Number.isFinite(v) ? 0.5 : NaN));
  }

  return values.map((v) => {
    if (!Number.isFinite(v)) return NaN;
    const t = (v - lo) / (hi - lo);
    return Math.max(0, Math.min(1, t));
  });
}

function latestValue(series, key) {
  for (let i = (series || []).length - 1; i >= 0; i -= 1) {
    const v = Number(series[i]?.[key]);
    if (Number.isFinite(v)) return v;
  }
  return null;
}

function latestFinite(values) {
  for (let i = values.length - 1; i >= 0; i -= 1) {
    if (Number.isFinite(values[i])) return values[i];
  }
  return null;
}

function fmt(v, digits) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return n.toFixed(digits);
}

const DRIVER_META = [
  { key: 'sst', normKey: 'sst_norm', label: 'SST', color: '#0284c7', digits: 2 },
  { key: 'chlor_a', normKey: 'chlor_a_norm', label: 'Chl-a', color: '#65a30d', digits: 3 },
  { key: 'kd490', normKey: 'kd490_norm', label: 'KD490', color: '#d97706', digits: 3 },
  { key: 'nflh', normKey: 'nflh_norm', label: 'NFLH', color: '#dc2626', digits: 3 },
];

export default function OceanDriversChart({ series, referenceSeries, width = 380, height = 190 }) {
  const xs = series || [];
  if (xs.length < 2) {
    return <div className="empty-chart">Not enough points for ocean driver trend</div>;
  }

  const ref = (referenceSeries && referenceSeries.length ? referenceSeries : xs);
  const normalizedByKey = {};
  DRIVER_META.forEach(({ key, normKey }) => {
    const provided = xs.map((d) => Number(d?.[normKey]));
    if (provided.some(Number.isFinite)) {
      normalizedByKey[key] = provided.map((v) => (Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : NaN));
      return;
    }
    const vals = xs.map((d) => Number(d?.[key]));
    const refVals = ref.map((d) => Number(d?.[key]));
    normalizedByKey[key] = robustNormalize(vals, refVals);
  });

  const oci = xs.map((_, i) => {
    const explicit = Number(xs[i]?.oci_proxy_adj ?? xs[i]?.oci_proxy);
    if (Number.isFinite(explicit)) return Math.max(0, Math.min(1, explicit));
    const vals = DRIVER_META.map(({ key }) => normalizedByKey[key][i]).filter(Number.isFinite);
    if (!vals.length) return NaN;
    return vals.reduce((s, v) => s + v, 0) / vals.length;
  });

  const pad = 14;
  const min = 0;
  const max = 1;

  const lines = DRIVER_META.map((d) => ({
    ...d,
    path: buildPath(normalizedByKey[d.key], width, height, pad, min, max),
  }));
  const ociPath = buildPath(oci, width, height, pad, min, max);
  const latestOCI = latestFinite(oci);

  return (
    <>
      <svg className="trend-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="ocean drivers chart">
        <rect x="0" y="0" width={width} height={height} rx="10" ry="10" fill="rgba(255,255,255,0.8)" />
        <line
          x1={toX(0, xs.length, width, pad)}
          y1={toY(0.5, min, max, height, pad)}
          x2={toX(Math.max(1, xs.length - 1), xs.length, width, pad)}
          y2={toY(0.5, min, max, height, pad)}
          stroke="#94a3b8"
          strokeWidth="1"
          strokeDasharray="4 4"
        />
        {lines.map((line) => (
          <path key={line.key} d={line.path} stroke={line.color} strokeWidth="1.7" fill="none" opacity="0.9" />
        ))}
        <path d={ociPath} stroke="#0f172a" strokeWidth="2.6" fill="none" />
      </svg>
      <div className="driver-legend">
        {lines.map((line) => (
          <div key={line.key} className="driver-item">
            <span className="driver-dot" style={{ background: line.color }} />
            <span>{line.label}</span>
            <strong className="driver-value">{fmt(latestValue(xs, line.key), line.digits)}</strong>
          </div>
        ))}
        <div className="driver-item">
          <span className="driver-dot" style={{ background: '#0f172a' }} />
          <span>OCI Proxy</span>
          <strong className="driver-value">{fmt(latestOCI, 3)}</strong>
        </div>
      </div>
    </>
  );
}
