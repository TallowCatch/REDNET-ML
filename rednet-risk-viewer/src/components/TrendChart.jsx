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

function pointsToPath(values, w, h, pad, min, max) {
  let path = '';
  for (let i = 0; i < values.length; i += 1) {
    const x = toX(i, values.length, w, pad);
    const y = toY(values[i], min, max, h, pad);
    path += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
  }
  return path;
}

function thresholdPath(level, count, w, h, pad, min, max) {
  const y = toY(level, min, max, h, pad);
  const x0 = toX(0, count, w, pad);
  const x1 = toX(Math.max(1, count - 1), count, w, pad);
  return `M ${x0} ${y} L ${x1} ${y}`;
}

function riskValue(row) {
  const ops = Number(row?.ops_risk);
  if (Number.isFinite(ops)) return ops;
  const model = Number(row?.hab_prob);
  return Number.isFinite(model) ? model : NaN;
}

export default function TrendChart({
  series,
  watch,
  action,
  width = 380,
  height = 180,
}) {
  const pad = 14;
  const vals = (series || []).map((d) => riskValue(d)).filter(Number.isFinite);

  if (vals.length < 2) {
    return <div className="empty-chart">Not enough points for trend</div>;
  }

  const min = Math.min(0, ...vals);
  const max = Math.max(1, ...vals);

  const path = pointsToPath(vals, width, height, pad, min, max);
  const watchPath = thresholdPath(watch, vals.length, width, height, pad, min, max);
  const actionPath = thresholdPath(action, vals.length, width, height, pad, min, max);

  return (
    <svg className="trend-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="risk trend chart">
      <rect x="0" y="0" width={width} height={height} rx="10" ry="10" fill="rgba(255,255,255,0.8)" />
      <path d={watchPath} stroke="#d97706" strokeWidth="1.5" strokeDasharray="4 4" fill="none" />
      <path d={actionPath} stroke="#b91c1c" strokeWidth="1.5" strokeDasharray="4 4" fill="none" />
      <path d={path} stroke="#0f766e" strokeWidth="2.2" fill="none" />
      {vals.map((v, i) => {
        const x = toX(i, vals.length, width, pad);
        const y = toY(v, min, max, height, pad);
        return <circle key={i} cx={x} cy={y} r="2.3" fill="#0f766e" />;
      })}
    </svg>
  );
}
