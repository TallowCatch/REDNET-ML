// src/forecast/trendForecast.js
export function trendForecast(series, horizonHours = 72, lookback = 12) {
    if (!series || series.length < 3) return null;
  
    const tail = series.slice(Math.max(0, series.length - lookback));
    if (tail.length < 3) return null;
  
    // Convert time to hours from start for regression stability
    const t0 = tail[0].time.getTime();
    const xs = tail.map((p) => (p.time.getTime() - t0) / (1000 * 60 * 60));
    const ys = tail.map((p) => p.risk);
  
    // Linear regression y = a + b*x
    const { a, b } = linReg(xs, ys);
  
    // Volatility estimate: std of residuals
    const residuals = ys.map((y, i) => y - (a + b * xs[i]));
    const sigma = std(residuals);
  
    // Persistence: EWMA of last values
    const ew = ewma(ys, 0.35);
    const lastT = tail[tail.length - 1].time;
    const lastX = xs[xs.length - 1];
  
    // We’ll forecast in 3 steps: +24h, +48h, +72h (or 1 step if your cadence is coarse)
    const steps = [24, 48, 72].filter((h) => h <= horizonHours);
    const preds = steps.map((dh) => {
      const x = lastX + dh;
      const mean = clamp01(0.55 * (a + b * x) + 0.45 * ew); // blend trend + persistence
      // envelope (you can tune multipliers)
      const low = clamp01(mean - 1.25 * sigma);
      const high = clamp01(mean + 1.25 * sigma);
      return { time: new Date(lastT.getTime() + dh * 3600 * 1000), mean, low, high };
    });
  
    return { model: { a, b, sigma, ew }, preds };
  }
  
  function linReg(xs, ys) {
    const n = xs.length;
    const xbar = xs.reduce((s, x) => s + x, 0) / n;
    const ybar = ys.reduce((s, y) => s + y, 0) / n;
  
    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++) {
      const dx = xs[i] - xbar;
      num += dx * (ys[i] - ybar);
      den += dx * dx;
    }
    const b = den === 0 ? 0 : num / den;
    const a = ybar - b * xbar;
    return { a, b };
  }
  
  function std(arr) {
    if (arr.length < 2) return 0;
    const m = arr.reduce((s, x) => s + x, 0) / arr.length;
    const v = arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1);
    return Math.sqrt(v);
  }
  
  function ewma(values, alpha = 0.3) {
    let s = values[0];
    for (let i = 1; i < values.length; i++) {
      s = alpha * values[i] + (1 - alpha) * s;
    }
    return s;
  }
  
  function clamp01(x) {
    return Math.max(0, Math.min(1, x));
  }
  