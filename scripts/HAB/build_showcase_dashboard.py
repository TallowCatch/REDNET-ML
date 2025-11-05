#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

# ---------- helpers for gallery ----------

def best_scene_id(row):
    for c in ["scene_id", "tile", "filename", "file", "image"]:
        if c in row and pd.notna(row[c]):
            return str(row[c])
    return ""

def build_cards(predict_df, env_df, jpg_roots, score_col, ycol, k):
    # merge env context (chlor_a, kd490, nflh) by scene_id if present
    env_cols = [c for c in ["scene_id", "chlor_a", "kd490", "nflh"] if c in env_df.columns]
    if "scene_id" in env_cols:
        merged = predict_df.merge(env_df[env_cols], on="scene_id", how="left")
    else:
        merged = predict_df.copy()

    # collect JPGs
    jpg_pool = []
    for r in jpg_roots:
        jpg_pool += glob(f"{r}/**/*.jpg", recursive=True)
    have = {Path(p).stem: p for p in jpg_pool}

    cards_html = []

    merged = merged.sort_values(score_col, ascending=False).head(k)

    for _, r in merged.iterrows():
        scene_full = best_scene_id(r)
        if not scene_full:
            continue

        # match jpgs by scene base (before _0000 / _0001)
        base = Path(scene_full).stem
        base = base.split(".")[0]
        imgs = []
        for stem, path in have.items():
            if stem.startswith(base):
                imgs.append(path)
        imgs = sorted(imgs)[:2]
        if not imgs:
            continue

        # meta
        month = str(r.get("month_key", "")).strip()
        score = float(r.get(score_col, np.nan))
        pred = int(r.get("pred", 0)) if "pred" in r else 1
        ytrue = int(r.get(ycol, -1)) if ycol and ycol in r else -1

        chla = r.get("chlor_a", "nan")
        kd490 = r.get("kd490", "nan")
        nflh = r.get("nflh", "nan")

        # badges
        badges = []
        if ytrue == 1 and pred == 1:
            badges.append("<span class='badge tp'>TP</span>")
        elif ytrue == 0 and pred == 1:
            badges.append("<span class='badge fp'>FP</span>")
        elif ytrue == 1 and pred == 0:
            badges.append("<span class='badge fn'>FN</span>")
        if np.isfinite(score):
            badges.append(f"<span class='badge score'>{score_col} {score:.3f}</span>")

        # images
        img_tags = []
        dot_tags = []
        for i, p in enumerate(imgs):
            cls = "active" if i == 0 else ""
            img_tags.append(
                f"<img class='{cls}' data-dt='' src='{Path(p).name}' loading='lazy'/>"
            )
            dot_tags.append("<div class='dot'></div>")

        month_label = month if month and month != "nan" else "—"

        card = f"""
<div class='card'>
  <div class='figure'>
    <div class='badges'>{"".join(badges)}</div>
    <div class='viewport'>
      {''.join(img_tags)}
    </div>
    <div class='dtbar'><span class='dtpill'></span></div>
    <div class='controls'>
      <div class='btn prev'>&lsaquo;</div>
      <div class='btn next'>&rsaquo;</div>
    </div>
    <div class='nav'>{"".join(dot_tags)}</div>
  </div>
  <div class='meta'>
    <div class='pills'><span class='pill'>{month_label}</span></div>
    <div class='kv'>scene: {scene_full}</div>
    <div class='kv'>chlor_a: {chla}</div>
    <div class='kv'>Kd490: {kd490}</div>
    <div class='kv'>nFLH: {nflh}</div>
  </div>
</div>
"""
        cards_html.append(card)

    return "\n".join(cards_html)

def classify_strength(f1, auroc):
    if not np.isfinite(f1) or not np.isfinite(auroc):
        return "weak"
    score = 0.5 * f1 + 0.5 * auroc
    if score >= 0.80:
        return "strong"
    elif score >= 0.60:
        return "med"
    else:
        return "weak"

def build_metric_pills(df_metrics):
    pills = []
    for _, r in df_metrics.iterrows():
        name = r["model"]
        f1 = float(r["f1"])
        au = float(r["auroc"])
        strength = classify_strength(f1, au)
        cls = f"metric-pill {strength}"
        if name.startswith("FUSION"):
            cls += " active"
        pill = f"""
<button class="{cls}" data-model="{name}">
  <span class="name">{name}</span>
  <span class="vals">F1 {f1:.2f} · AUROC {au:.2f}</span>
</button>"""
        pills.append(pill)
    return "\n".join(pills)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fusion_dir",
        default="runs/fusion/fused_sets/B_mined_timecv_norm_f1",
        help="Fusion run directory to visualise (with predictions_cv2.csv)",
    )
    ap.add_argument(
        "--metrics_csv",
        default="runs/fusion/qc_model_comparison/all_model_metrics.csv",
        help="CSV with model metrics (acc,prec,rec,f1,auroc)",
    )
    ap.add_argument(
        "--env_csv",
        default="runs/datasets/hab_candidates_review.csv",
        help="CSV with scene_id, chlor_a, kd490, nflh",
    )
    ap.add_argument(
        "--jpg_roots",
        nargs="+",
        default=["qc/hab_hits_inspect"],
        help="Where to search for JPEGs (repopulated by repopulate_hab_hits_inspect.py)",
    )
    ap.add_argument(
        "--k_scenes",
        type=int,
        default=24,
        help="How many top scenes to show in gallery",
    )
    ap.add_argument(
        "--out_html",
        default="qc/index.html",
        help="Output HTML file for the dashboard",
    )
    args = ap.parse_args()

    fdir = Path(args.fusion_dir)
    preds_candidates = [
        fdir / "predictions_cv2.csv",
        fdir / "predictions.csv",
        fdir / "predictions_cv2_train.csv",
    ]
    preds = next((p for p in preds_candidates if p.exists()), None)
    if preds is None:
        raise SystemExit(f"No predictions file found in {fdir}")

    df_pred = pd.read_csv(preds)
    print(f"Loaded predictions: {preds} ({len(df_pred)} rows)")

    # pick score column
    prefer = ["p_fused", "p_tab"]
    score_col = next((c for c in prefer if c in df_pred.columns), None)
    if score_col is None:
        numeric_cols = [c for c in df_pred.select_dtypes(include=np.number).columns if c != "hab_label"]
        if not numeric_cols:
            raise SystemExit("No numeric score columns found in predictions.")
        score_col = numeric_cols[0]
    print(f"Using score column: {score_col}")

    ycol = "hab_label" if "hab_label" in df_pred.columns else None
    thr = float(np.nanmedian(df_pred[score_col]))
    df_pred["pred"] = (df_pred[score_col] >= thr).astype(int)

    # load env + metrics
    env_df = pd.read_csv(args.env_csv) if Path(args.env_csv).exists() else pd.DataFrame()
    df_metrics = pd.read_csv(args.metrics_csv)

    # filter metrics to real models (ignore DELTA rows)
    df_metrics = df_metrics[~df_metrics["model"].str.startswith("DELTA_")].reset_index(drop=True)

    # scoreboard HTML and metrics JSON for JS
    scoreboard_html = build_metric_pills(df_metrics)

    metrics_for_js = []
    for _, r in df_metrics.iterrows():
        m = {
            "model": r["model"],
            "acc": float(r["acc"]),
            "prec": float(r["prec"]),
            "rec": float(r["rec"]),
            "f1": float(r["f1"]),
            "auroc": float(r["auroc"]),
            "is_fusion": r["model"].startswith("FUSION"),
        }
        metrics_for_js.append(m)
    metrics_json = json.dumps(metrics_for_js)

    # gallery cards
    cards_html = build_cards(df_pred, env_df, args.jpg_roots, score_col, ycol, args.k_scenes)

    # ---- assemble HTML ----
    parts = []

    parts.append("""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HAB Showcase — Fusion</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="anonymous">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#050f10; --card:#0f2627; --ink:#e8f5f5; --soft:#a8d0d0;
  --pill:#2a7a5f; --pill-ink:#efe; --accent:#4b8; --accent2:#2a4;
  --tp:#1b824e; --fp:#8b2b2b; --fn:#8b732b; --score:#2a5560;
  --shadow:0 1px 0 #1c2e2e inset;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,sans-serif;
  background:var(--bg);
  color:var(--ink);
  padding:28px 20px 80px;
}
a{color:#7bffbf;text-decoration:none}
.header{max-width:1100px;margin:0 auto 22px auto;text-align:center}
h1{margin:0 0 10px 0;font-size:2.4rem;letter-spacing:.2px}
.subtitle{color:var(--soft);margin:0 0 16px 0}
.lead{max-width:900px;margin:12px auto 24px auto;color:#d7f0ee;line-height:1.6;font-size:17px;opacity:.95}
.navbtns{display:flex;justify-content:center;gap:12px;margin-bottom:6px;flex-wrap:wrap}
.navbtn{
  padding:6px 14px;border-radius:999px;border:1px solid rgba(255,255,255,.12);
  background:rgba(0,0,0,.35);color:var(--ink);font-size:13px;cursor:pointer;
}
.navbtn:hover{border-color:var(--accent)}
.hr{max-width:1100px;margin:26px auto;border:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.18),transparent)}
.section{max-width:1600px;margin:34px auto 14px auto;padding:0 2px}
.section h2{margin:0 0 10px 0;font-size:1.5rem}
.section .hint{color:var(--soft);font-size:13px;margin:2px 0 14px 0}
.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(380px,1fr)); gap:16px}
.card{
  background:var(--card); border-radius:16px; padding:12px;
  box-shadow:var(--shadow); display:flex; flex-direction:column;
  transform:translateX(0); opacity:0;
  transition:transform .6s cubic-bezier(.22,1,.36,1), opacity .5s;
}
.card.reveal-left{transform:translateX(-34px); opacity:0}
.card.reveal-right{transform:translateX(34px); opacity:0}
.card.visible{transform:translateX(0); opacity:1}
.figure{position:relative; border-radius:12px; overflow:hidden; background:#0d1e1e}
.viewport{position:relative; width:100%; aspect-ratio:4/3}
.figure img{position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block; opacity:0; transition:opacity .2s}
.figure img.active{opacity:1}
.controls{position:absolute; inset:0; display:flex; align-items:center; justify-content:space-between; pointer-events:none}
.btn{pointer-events:auto; background:rgba(0,0,0,.45); border:1px solid rgba(255,255,255,.25); color:#fff; width:40px; height:40px; border-radius:999px; display:flex; align-items:center; justify-content:center; margin:10px; user-select:none; font-size:22px; line-height:1}
.nav{position:absolute; left:0; right:0; bottom:10px; display:flex; justify-content:center; gap:8px; pointer-events:auto}
.dot{width:10px; height:10px; border-radius:50%; background:var(--accent2); opacity:.55; cursor:pointer}
.dot.active{opacity:1; background:var(--accent)}
.dtbar{position:absolute; left:0; bottom:48px; right:0; display:flex; justify-content:center}
.dtpill{background:rgba(0,0,0,.45); color:#fff; font-size:12px; padding:4px 10px; border-radius:999px; border:1px solid rgba(255,255,255,.2)}
.meta{font-size:14px; line-height:1.45; margin-top:10px; color:#cfe; word-break:break-word; overflow-wrap:anywhere}
.pills{margin-bottom:6px}
.pill{background:var(--pill); color:var(--pill-ink); padding:4px 10px; border-radius:999px; font-size:12px; margin-right:8px; display:inline-block}
.kv{margin:2px 0}
small.soft{color:var(--soft)}
.badges{position:absolute;top:6px;left:6px;display:flex;flex-wrap:wrap;gap:6px;z-index:2}
.badge{background:rgba(0,0,0,.55);border-radius:8px;padding:2px 6px;font-size:12px;line-height:1.3;color:#fff}
.badge.tp{background:var(--tp)}
.badge.fp{background:var(--fp)}
.badge.fn{background:var(--fn)}
.badge.score{background:var(--score)}
#map{width:100%;height:420px;border-radius:20px;overflow:hidden;box-shadow:var(--shadow);margin-top:12px;pointer-events:none}
.map-caption{color:var(--soft);font-size:13px;margin-top:4px}
.model-layout{display:grid;grid-template-columns:minmax(260px,320px) minmax(0,1fr);gap:24px;align-items:flex-start}
@media(max-width:900px){.model-layout{grid-template-columns:1fr}}
.ticker{margin:16px 0 24px;display:flex;flex-direction:column;gap:8px;max-width:360px}
.metric-pill{
  border:none;text-align:left;padding:8px 14px;border-radius:999px;
  font-size:13px;display:flex;justify-content:space-between;align-items:center;
  cursor:pointer;background:linear-gradient(90deg,rgba(0,0,0,.25),rgba(0,0,0,.45));
  color:var(--ink);box-shadow:var(--shadow);transition:background .2s,transform .1s,opacity .2s;
}
.metric-pill .name{font-weight:600;letter-spacing:.03em}
.metric-pill .vals{opacity:.85}
.metric-pill.strong{border:1px solid #2ecc71}
.metric-pill.med{border:1px solid #f1c40f}
.metric-pill.weak{border:1px solid #e74c3c}
.metric-pill.active{background:linear-gradient(90deg,rgba(0,0,0,.5),rgba(0,0,0,.7));transform:translateY(-1px)}
.metric-pill.inactive{opacity:.45}
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;margin-top:8px}
.chart-card{
  background:var(--card);border-radius:16px;padding:12px 14px;box-shadow:var(--shadow);
}
.chart-card h3{margin:2px 0 4px 0;font-size:1rem}
.chart-card p{margin:0 0 8px 0;font-size:13px;color:var(--soft)}
.chart-img{width:100%;border-radius:12px;display:block;background:#000;object-fit:contain}
#radar-container{height:280px}
#radarChart{width:100%;height:100%}
</style>
</head>
<body>
<div class="header">
  <h1>HAB Detection Summary</h1>
  <div class="subtitle">Oman coastline — Fusion (Tabular + Detectors)</div>
  <div class="lead">
    Visual audit of <b>fused HAB predictions</b> along the Oman coastline.
    Tabular Sentinel-3 water-quality features are fused with deep detectors over Sentinel-2 tiles
    to improve recall while controlling false alarms.
  </div>
  <div class="navbtns">
    <a class="navbtn" href="#overview">Overview</a>
    <a class="navbtn" href="#models">Model performance</a>
    <a class="navbtn" href="#top-scenes">Top scenes</a>
  </div>
</div>

<hr class="hr">

<div id="overview" class="section">
  <h2>Study area</h2>
  <div class="hint">Oman coastal region monitored for potentially harmful algal blooms (HABs).</div>
  <div id="map"></div>
  <div class="map-caption">AOI bounding box: [51.5, 26.5] – [60.8, 15.5] (approx. Oman coastline and Arabian Sea).</div>
</div>

<hr class="hr">

<div id="models" class="section">
  <h2>Model performance snapshot</h2>
  <div class="hint">
    Each pill shows test F1 and AUROC for a detector or fusion run.
    Green ≈ stronger, yellow ≈ moderate, red ≈ weaker. Click pills to toggle curves on the radar chart.
  </div>
  <div class="model-layout">
    <div>
      <div class="ticker">
""")

    parts.append(scoreboard_html)

    parts.append("""
      </div>
      <small class="soft">Thresholds for strength are based on a mix of F1 and AUROC.</small>
    </div>

    <div class="chart-grid">
      <div class="chart-card">
        <h3>Detector vs fusion — radar (interactive)</h3>
        <p>Click pills to toggle models. Metrics: accuracy, precision, recall, F1, AUROC.</p>
        <div id="radar-container">
          <canvas id="radarChart"></canvas>
        </div>
      </div>
      <div class="chart-card">
        <h3>Detector vs fusion — bar metrics</h3>
        <p>Static bar chart summarising accuracy, precision, recall, F1 and AUROC.</p>
        <img class="chart-img" src="../fusion/qc_model_comparison/all_model_bar.png" alt="Bar metrics">
      </div>
      <div class="chart-card">
        <h3>Fusion PR curve</h3>
        <p>Precision–recall curve for fusion vs baselines (from your QC script).</p>
        <img class="chart-img" src="../fusion/qc_model_comparison/comp_pr.png" alt="PR curve">
      </div>
      <div class="chart-card">
        <h3>Fusion ROC curve</h3>
        <p>ROC curve for fusion vs baselines.</p>
        <img class="chart-img" src="../fusion/qc_model_comparison/comp_roc.png" alt="ROC curve">
      </div>
    </div>
  </div>
</div>

<hr class="hr">

<div id="top-scenes" class="section">
  <h2>Top scored scenes</h2>
  <div class="hint">Scroll to reveal fused high-confidence HAB candidates (TPs & FPs annotated).</div>
  <div class="grid">
""")

    parts.append(cards_html)

    parts.append("""
  </div>
</div>

<script>
// ---- static Leaflet map (no interaction) ----
const map = L.map('map', {
  zoomControl: false,
  attributionControl: true,
  dragging: false,
  scrollWheelZoom: false,
  doubleClickZoom: false,
  boxZoom: false,
  keyboard: false,
  tap: false,
});

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 8,
  minZoom: 3
}).addTo(map);

// AOI polygon: original coords are [lon, lat]; Leaflet wants [lat, lon]
const aoi = [
  [26.5, 51.5],
  [15.5, 51.5],
  [15.5, 60.8],
  [26.5, 60.8],
  [26.5, 51.5]
];

const poly = L.polygon(aoi, {
  color: '#4bff9a',
  weight: 2,
  fillColor: '#4bff9a',
  fillOpacity: 0.15
}).addTo(map);

map.fitBounds(poly.getBounds());

// ---- feature cards: image carousels + scroll reveal ----
document.querySelectorAll('.figure').forEach(fig=>{
  const imgs=[...fig.querySelectorAll('img')];
  const dots=[...fig.querySelectorAll('.dot')];
  const dateEl=fig.querySelector('.dtpill');
  let i=0;
  function show(k){
    i=((k%imgs.length)+imgs.length)%imgs.length;
    imgs.forEach((im,j)=>im.classList.toggle('active', j===i));
    dots.forEach((d,j)=>d.classList.toggle('active', j===i));
    if(dateEl){ dateEl.textContent = imgs[i].dataset.dt || ''; }
  }
  if(imgs.length){ show(0); }
  fig.querySelector('.prev')?.addEventListener('click', ()=>show(i-1));
  fig.querySelector('.next')?.addEventListener('click', ()=>show(i+1));
  dots.forEach((d,idx)=>d.addEventListener('click', ()=>show(idx)));
});

const io = new IntersectionObserver((ents)=>{
  ents.forEach(e=>{
    if(e.isIntersecting){ e.target.classList.add('visible'); io.unobserve(e.target); }
  });
},{threshold:.15});
document.querySelectorAll('.card').forEach((el,i)=>{
  el.classList.add(i%2 ? 'reveal-right' : 'reveal-left');
  io.observe(el);
});

// ---- metrics for interactive radar ----
const METRICS = """)

    parts.append(metrics_json)
    parts.append(""";

// build radar chart
const radarLabels = ['acc','prec','rec','f1','auroc'];
const radarCtx = document.getElementById('radarChart').getContext('2d');

const datasets = METRICS.map((m, idx) => ({
  label: m.model,
  data: radarLabels.map(k => m[k] ?? 0),
  fill: true,
  tension: 0.25,
  pointRadius: 2,
  hidden: !m.is_fusion,   // show fusion models by default
}));

const radarChart = new Chart(radarCtx, {
  type: 'radar',
  data: {
    labels: radarLabels,
    datasets: datasets
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: { color: '#e8f5f5', font: { size: 11 } }
      }
    },
    scales: {
      r: {
        suggestedMin: 0,
        suggestedMax: 1,
        ticks: { stepSize: 0.2, backdropColor: 'rgba(0,0,0,0)', color:'#a8d0d0' },
        grid: { color:'rgba(255,255,255,0.15)' },
        angleLines: { color:'rgba(255,255,255,0.15)' },
        pointLabels: { color:'#e8f5f5', font:{ size:11 } }
      }
    }
  }
});

// click pills to toggle datasets
document.querySelectorAll('.metric-pill').forEach(pill=>{
  const model = pill.dataset.model;
  pill.addEventListener('click', ()=>{
    const ds = radarChart.data.datasets.find(d => d.label === model);
    if(!ds) return;
    const willShow = ds.hidden;   // toggle
    ds.hidden = !willShow;
    pill.classList.toggle('inactive', !willShow);
    pill.classList.toggle('active', willShow);
    radarChart.update();
  });
});
</script>
</body>
</html>
""")

    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")
    print(f"✓ wrote dashboard to {out_path}")

if __name__ == "__main__":
    main()
