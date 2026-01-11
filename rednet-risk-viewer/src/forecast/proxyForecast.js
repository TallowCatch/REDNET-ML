// src/forecast/proxyForecast.js
// Fits risk ~ [sst, chlor_a, kd490, flh] on recent history (if present)
export function fitProxyModel(series, lookback = 60, lambda = 1e-2) {
    const tail = series.slice(Math.max(0, series.length - lookback));
  
    // Build X, y only where all needed proxies exist
    const X = [];
    const y = [];
    for (const p of tail) {
      const feats = [p.sst, p.chlor_a, p.kd490, p.flh];
      if (feats.some((v) => v === null || !Number.isFinite(v))) continue;
      X.push([1, ...feats]); // intercept
      y.push(p.risk);
    }
    if (X.length < 12) return null;
  
    const w = ridgeClosedForm(X, y, lambda);
    return { w, features: ["1", "sst", "chlor_a", "kd490", "flh"] };
  }
  
  // Predict risk given future proxies (or current proxies)
  export function proxyPredict(model, feats) {
    if (!model) return null;
    const x = [1, feats.sst, feats.chlor_a, feats.kd490, feats.flh];
    if (x.slice(1).some((v) => v === null || !Number.isFinite(v))) return null;
  
    let s = 0;
    for (let i = 0; i < model.w.length; i++) s += model.w[i] * x[i];
    return clamp01(s);
  }
  
  // -------- math ----------
  function ridgeClosedForm(X, y, lambda) {
    // w = (X^T X + λI)^-1 X^T y
    const XT = transpose(X);
    const XTX = matMul(XT, X);
    const p = XTX.length;
  
    for (let i = 0; i < p; i++) XTX[i][i] += lambda;
  
    const XTy = matVecMul(XT, y);
    const inv = invertSmallMatrix(XTX);
    return matVecMul(inv, XTy);
  }
  
  function transpose(A) {
    return A[0].map((_, j) => A.map((r) => r[j]));
  }
  function matMul(A, B) {
    const m = A.length, n = B[0].length, k = B.length;
    const out = Array.from({ length: m }, () => Array(n).fill(0));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        for (let t = 0; t < k; t++) out[i][j] += A[i][t] * B[t][j];
    return out;
  }
  function matVecMul(A, v) {
    return A.map((row) => row.reduce((s, x, i) => s + x * v[i], 0));
  }
  
  // Gauss-Jordan for small matrices (<= 6x6-ish). Fine here.
  function invertSmallMatrix(A) {
    const n = A.length;
    const M = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  
    for (let col = 0; col < n; col++) {
      // pivot
      let pivot = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
      [M[col], M[pivot]] = [M[pivot], M[col]];
  
      const diag = M[col][col] || 1e-12;
      for (let j = 0; j < 2 * n; j++) M[col][j] /= diag;
  
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col];
        for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
      }
    }
  
    return M.map((row) => row.slice(n));
  }
  
  function clamp01(x) {
    return Math.max(0, Math.min(1, x));
  }
  