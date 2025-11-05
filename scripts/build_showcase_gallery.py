#!/usr/bin/env python3
"""
Build a static HTML dashboard for HAB fusion results.

Outputs: qc/showcase_mined_b/index.html (or custom --outdir)

Usage:
  python scripts/HAB/build_showcase_gallery.py \
    --fusion_dir runs/fusion/fused_sets/B_mined_timecv_norm_f1 \
    --outdir qc/showcase_mined_b \
    --env_csv runs/datasets/hab_candidates_review.csv \
    --k 40
"""
import argparse, re
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np

# ============================================================
# STYLE + JS
# ============================================================
STYLE = r"""
:root{
  --bg:#0b1a1a; --card:#112a29; --ink:#e8f5f5; --soft:#a8d0d0;
  --pill:#2a7a5f; --pill-ink:#efe; --accent:#4b8; --accent2:#2a4;
  --tp:#1b824e; --fp:#8b2b2b; --fn:#8b732b; --score:#2a5560;
  --shadow:0 1px 0 #1c2e2e inset;
}
*{box-sizing:border-box}
html,body{margin:0; padding:0}
body{
  font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,sans-serif;
  background:var(--bg);color:var(--ink);padding:28px 20px 80px;
}
.header{max-width:1100px;margin:0 auto 22px auto;text-align:center}
h1{margin:0 0 6px 0;font-size:2.2rem;letter-spacing:.2px}
.subtitle{color:var(--soft);margin:0 0 16px 0}
.lead{
  max-width:900px;margin:12px auto 24px auto;color:#d7f0ee;
  line-height:1.6;font-size:17px;opacity:.95
}
.hr{
  max-width:1100px;margin:26px auto;border:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.18),transparent)
}
.section{max-width:1600px;margin:34px auto 14px auto;padding:0 2px}
.section h2{margin:0 0 10px 0}
.section .hint{color:var(--soft);font-size:13px;margin:2px 0 14px 0}
.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(380px,1fr)); gap:16px}
.card{
  background:var(--card); border-radius:16px; padding:12px; box-shadow:var(--shadow);
  display:flex; flex-direction:column; transform:translateX(0); opacity:0;
  transition:transform .6s cubic-bezier(.22,1,.36,1), opacity .5s
}
.card.reveal-left{transform:translateX(-34px); opacity:0}
.card.reveal-right{transform:translateX(34px); opacity:0}
.card.visible{transform:translateX(0); opacity:1}
.figure{position:relative; border-radius:12px; overflow:hidden; background:#0d1e1e}
.viewport{position:relative; width:100%; aspect-ratio:4/3}
.figure img{
  position:absolute; inset:0; width:100%; height:100%; object-fit:cover;
  display:block; opacity:0; transition:opacity .2s
}
.figure img.active{opacity:1}
.controls{
  position:absolute; inset:0; display:flex; align-items:center;
  justify-content:space-between; pointer-events:none
}
.btn{
  pointer-events:auto; background:rgba(0,0,0,.45);
  border:1px solid rgba(255,255,255,.25); color:#fff;
  width:40px; height:40px; border-radius:999px;
  display:flex; align-items:center; justify-content:center;
  margin:10px; user-select:none; font-size:22px; line-height:1
}
.nav{
  position:absolute; left:0; right:0; bottom:10px;
  display:flex; justify-content:center; gap:8px; pointer-events:auto
}
.dot{width:10px; height:10px; border-radius:50%; background:var(--accent2); opacity:.55; cursor:pointer}
.dot.active{opacity:1; background:var(--accent)}
.dtbar{
  position:absolute; left:0; bottom:48px; right:0;
  display:flex; justify-content:center
}
.dtpill{
  background:rgba(0,0,0,.45); color:#fff; font-size:12px;
  padding:4px 10px; border-radius:999px; border:1px solid rgba(255,255,255,.2)
}
.meta{font-size:14px; line-height:1.45; margin-top:10px; color:#cfe;
      word-break:break-word; overflow-wrap:anywhere}
.pills{margin-bottom:6px}
.pill{
  background:var(--pill); color:var(--pill-ink);
  padding:4px 10px; border-radius:999px; font-size:12px;
  margin-right:8px; display:inline-block
}
.kv{margin:2px 0}
small.soft{color:var(--soft)}
.badges{position:absolute;top:6px;left:6px;display:flex;flex-wrap:wrap;gap:6px;z-index:2}
.badge{
  background:rgba(0,0,0,.55);border-radius:8px;
  padding:2px 6px;font-size:12px;line-height:1.3;color:#fff
}
.badge.tp{background:var(--tp)}
.badge.fp{background:var(--fp)}
.badge.fn{background:var(--fn)}
.badge.score{background:var(--score)}

.navbar{
  max-width:1100px;
  margin:0 auto 20px auto;
  display:flex;justify-content:center;gap:10px;flex-wrap:wrap
}
.navbtn{
  border-radius:999px;padding:6px 14px;
  border:1px solid rgba(255,255,255,.2);
  background:rgba(0,0,0,.25);color:var(--soft);
  font-size:13px;cursor:pointer
}
.navbtn:hover{border-color:var(--accent);color:#fff}

.metrics-strip{
  max-width:1100px;margin:10px auto 0 auto;
  padding:8px 10px;border-radius:12px;
  background:linear-gradient(90deg,rgba(30,80,80,.8),rgba(8,35,35,.9));
  box-shadow:0 0 0 1px rgba(255,255,255,.05);
  font-size:13px;overflow-x:auto;white-space:nowrap;
}
.metric-pill{
  display:inline-block;padding:6px 10px;margin:4px 6px 4px 0;
  border-radius:10px;border:1px solid rgba(255,255,255,.15);
  background:rgba(0,0,0,.25);
}
.metric-pill span.label{font-weight:600;margin-right:6px}
.metric-pill.good{border-color:#32cd8a;color:#bdfad8}
.metric-pill.meh{border-color:#e6c15a;color:#ffeaa6}
.metric-pill.bad{border-color:#e06666;color:#ffc1c1}

.charts-grid{
  max-width:1100px;margin:10px auto 0 auto;
  display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
  gap:16px;
}
.chart-card{
  background:var(--card);border-radius:14px;
  padding:10px;box-shadow:var(--shadow);
}
.chart-card img{
  width:100%;border-radius:10px;display:block;
}
.chart-card h3{margin:0 0 6px 0;font-size:15px}
.chart-card p{margin:0;font-size:13px;color:var(--soft)}

#aoi-map{
  width:100%;height:330px;border-radius:18px;overflow:hidden;
  box-shadow:0 18px 40px rgba(0,0,0,.4);
}
"""

SCRIPT = r"""
// image carousels
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

// scroll-in animation
const io = new IntersectionObserver((ents)=>{
  ents.forEach(e=>{
    if(e.isIntersecting){ e.target.classList.add('visible'); io.unobserve(e.target); }
  });
},{threshold:.15});
document.querySelectorAll('.card').forEach((el,i)=>{
  el.classList.add(i%2 ? 'reveal-right' : 'reveal-left');
  io.observe(el);
});

// Leaflet AOI map (Oman box)
const aoiGeoJSON = {
  "type":"FeatureCollection",
  "features":[{"type":"Feature","properties":{},"geometry":{
    "type":"Polygon",
    "coordinates":[[
      [51.5,26.5],[51.5,15.5],[60.8,15.5],[60.8,26.5],[51.5,26.5]
    ]]
  }}]
};

function initAoiMap(){
  const mapEl = document.getElementById('aoi-map');
  if(!mapEl || !window.L) return;
  const map = L.map('aoi-map');
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 11,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);
  const layer = L.geoJSON(aoiGeoJSON, {style:{color:'#4b8',weight:2,fillOpacity:0.1}}).addTo(map);
  map.fitBounds(layer.getBounds().pad(0.15));
}
window.addEventListener('DOMContentLoaded', initAoiMap);
"""

# ============================================================
# UTILS
# ============================================================
SCENE_RE = re.compile(r'(S2[AB]_MSIL2A_[0-9T_]+)')

def to_scene_base(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    m = SCENE_RE.search(s)
    if m:
        return m.group(1)
    return Path(s).stem

def best_scene_id(row):
    for c in ["scene_id", "tile", "filename", "file", "image"]:
        if c in row and pd.notna(row[c]):
            return to_scene_base(str(row[c]))
    return None

def format_card(row, imgs, score_col, ycol):
    scene = row.get("scene_id", "")
    dt    = row.get("datetime", "")
    chla  = row.get("chlor_a", "-")
    kd    = row.get("kd490", "-")
    flh   = row.get("nflh", "-")
    score = row.get(score_col, np.nan)
    pred  = row.get("pred", -1)
    ytrue = row.get(ycol, -1)
    dots  = "".join("<div class='dot'></div>" for _ in imgs)
    ims   = "\n".join(
        f"<img class='{'active' if i==0 else ''}' data-dt='{dt}' src='{Path(j).name}' loading='lazy'/>"
        for i, j in enumerate(imgs)
    )
    badges = []
    if ytrue == 1 and pred == 1:
        badges.append("<span class='badge tp'>TP</span>")
    elif ytrue == 0 and pred == 1:
        badges.append("<span class='badge fp'>FP</span>")
    elif ytrue == 1 and pred == 0:
        badges.append("<span class='badge fn'>FN</span>")
    if np.isfinite(score):
        badges.append(f"<span class='badge score'>{score_col} {score:.3f}</span>")
    pills = "".join(
        f"<span class='pill'>{str(v)}</span>"
        for v in [row.get("month_key", ""), row.get("date_key", "")]
        if pd.notna(v)
    )
    return f"""
<div class='card'>
  <div class='figure'>
    <div class='badges'>{''.join(badges)}</div>
    <div class='viewport'>{ims}</div>
    <div class='dtbar'><span class='dtpill'></span></div>
    <div class='controls'><div class='btn prev'>&lsaquo;</div><div class='btn next'>&rsaquo;</div></div>
    <div class='nav'>{dots}</div>
  </div>
  <div class='meta'>
    <div class='pills'>{pills}</div>
    <div class='kv'>scene: {scene}</div>
    <div class='kv'>chlor_a: {chla}</div>
    <div class='kv'>Kd490: {kd}</div>
    <div class='kv'>nFLH: {flh}</div>
  </div>
</div>"""

# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_dir", default="runs/fusion/fused_sets/B_mined_timecv_norm_f1")
    ap.add_argument("--outdir", default="qc/showcase_mined_b")
    ap.add_argument("--jpg_roots", nargs="+", default=["qc/hab_hits_inspect"])
    ap.add_argument("--env_csv", default="runs/datasets/hab_candidates_review.csv")
    ap.add_argument("--k", type=int, default=40)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Load predictions CSV (auto-detect best file)
    fdir = Path(args.fusion_dir)
    cand = [
        fdir / "predictions_cv3.csv",
        fdir / "predictions.csv",
        fdir / "predictions_cv3_train.csv",
    ]
    preds = next((p for p in cand if p.exists()), None)
    if preds is None:
        raise FileNotFoundError(f"No predictions file found in {fdir}.")

    df = pd.read_csv(preds)
    print(f"Loaded predictions: {preds.name} ({len(df)} rows)")

    # Load env info (chlor_a, kd490, nflh) if available
    env_path = Path(args.env_csv)
    if env_path.exists():
        env = pd.read_csv(env_path)
        env_cols = [c for c in ["scene_id", "chlor_a", "kd490", "nflh"] if c in env.columns]
        if "scene_id" in env_cols and "scene_id" in df.columns:
            df = df.merge(env[env_cols], on="scene_id", how="left")
    else:
        print(f"[warn] env_csv not found at {env_path}; skipping env merge.")

    # Pick score column safely
    candidates = [c for c in ["p_fused", "p_tab"] if c in df.columns]
    if not candidates:
        # any frcnn score
        frcnn_cols = [x for x in df.columns if x.startswith("frcnn_")]
        if frcnn_cols:
            candidates = [frcnn_cols[0]]
    if not candidates:
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns
                        if c not in ["hab_label", "y_true"]]
        if not numeric_cols:
            raise ValueError("No numeric score columns found.")
        score_col = numeric_cols[0]
    else:
        score_col = candidates[0]

    ycol = "hab_label" if "hab_label" in df.columns else (
        "y_true" if "y_true" in df.columns else None
    )
    thr = float(np.nanmedian(df[score_col]))
    df["pred"] = (df[score_col] >= thr).astype(int)

    # Copy images recursively (we don't actually copy, we just look them up)
    jpg_pool = []
    for r in args.jpg_roots:
        jpg_pool += glob(f"{r}/**/*.jpg", recursive=True)
    have = {Path(p).stem: p for p in jpg_pool}

    # Build cards
    cards = []
    for _, r in df.sort_values(score_col, ascending=False).head(args.k).iterrows():
        scene = best_scene_id(r)
        if not scene:
            continue
        imgs = [p for k, p in have.items() if k.startswith(scene)]
        imgs = sorted(imgs)[:2]
        if not imgs:
            continue
        cards.append(format_card(r, imgs, score_col, ycol))

    # ------- MODEL METRIC "TICKER" -------
    ticker_html = ""
    metrics_path = Path("runs/fusion/qc_model_comparison/all_model_metrics.csv")
    if metrics_path.exists():
        mdf = pd.read_csv(metrics_path)
        pills = []
        for _, row in mdf.iterrows():
            name = str(row["model"])
            auroc = float(row.get("auroc", np.nan))
            f1 = float(row.get("f1", np.nan))
            if not np.isfinite(auroc):
                status = "meh"
            elif auroc >= 0.80:
                status = "good"
            elif auroc >= 0.60:
                status = "meh"
            else:
                status = "bad"
            pills.append(
                f"<span class='metric-pill {status}'>"
                f"<span class='label'>{name}</span>"
                f"F1 {f1:.2f} · AUROC {auroc:.2f}"
                "</span>"
            )
        ticker_html = (
            "<div class='metrics-strip'>"
            "<span class='soft' style='margin-right:8px;'>Model snapshot:</span>"
            + "".join(pills) +
            "</div>"
        )
    else:
        print(f"[warn] metrics file not found at {metrics_path}; ticker will be empty.")

    # ------- Assemble HTML -------
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>HAB Showcase — Fusion</title>",
        # Leaflet CSS
        "<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css' "
        "integrity='sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=' crossorigin=''/>",
        f"<style>{STYLE}</style></head><body>",

        # HERO
        "<div class='header'>"
        "<h1>HAB Detection Summary</h1>"
        "<div class='subtitle'>Oman coastline — Fusion (Tabular + Detectors)</div>"
        "</div>",

        "<div class='lead'>"
        "Visual audit of <b>fused HAB predictions</b> along the Oman coastline. "
        "Tabular Sentinel-3 water-quality features are fused with deep detectors over "
        "Sentinel-2 tiles to improve recall while controlling false alarms."
        "</div>",

        # NAV
        "<div class='navbar'>"
        "<button class='navbtn' onclick=\"document.getElementById('section-overview').scrollIntoView({behavior:'smooth'})\">Overview</button>"
        "<button class='navbtn' onclick=\"document.getElementById('section-models').scrollIntoView({behavior:'smooth'})\">Model performance</button>"
        "<button class='navbtn' onclick=\"document.getElementById('section-gallery').scrollIntoView({behavior:'smooth'})\">Top scenes</button>"
        "</div>",

        "<hr class='hr'>",

        # OVERVIEW + AOI MAP
        "<div id='section-overview' class='section'>"
        "<h2>Study area</h2>"
        "<div class='hint'>Oman coastal region monitored for potentially harmful algal blooms (HABs).</div>"
        "<div id='aoi-map'></div>"
        "</div>",

        "<hr class='hr'>",

        # MODEL METRICS SECTION
        "<div id='section-models' class='section'>"
        "<h2>Model performance snapshot</h2>"
        "<div class='hint'>Each pill shows test F1 and AUROC for a detector or fusion run. "
        "Green ≈ strong, yellow ≈ moderate, red ≈ weaker.</div>",
        ticker_html,
        "</div>",

        # CHARTS
        "<div class='section'>"
        "<div class='charts-grid'>"
        "<div class='chart-card'>"
        "<h3>Detector vs fusion — bar metrics</h3>"
        "<p>Accuracy, precision, recall, F1 and AUROC across models.</p>"
        "<img src='runs/fusion/qc_model_comparison/all_model_bar.png' alt='Bar chart of model metrics'>"
        "</div>"
        "<div class='chart-card'>"
        "<h3>Detector vs fusion — radar</h3>"
        "<p>Radar view of the same metrics to highlight trade-offs.</p>"
        "<img src='runs/fusion/qc_model_comparison/all_model_radar.png' alt='Radar chart of model metrics'>"
        "</div>"
        "<div class='chart-card'>"
        "<h3>Fusion PR curve</h3>"
        "<p>Precision–recall curve for fusion vs baselines.</p>"
        "<img src='model_comp/comp_pr.png' alt='PR curves'>"
        "</div>"
        "<div class='chart-card'>"
        "<h3>Fusion ROC curve</h3>"
        "<p>ROC curve for fusion vs baselines.</p>"
        "<img src='model_comp/comp_roc.png' alt='ROC curves'>"
        "</div>"
        "</div></div>",

        "<hr class='hr'>",

        # GALLERY
        "<div id='section-gallery' class='section'>"
        "<h2>Top scored scenes</h2>"
        "<div class='hint'>Scroll to reveal fused high-confidence HAB candidates (TPs & FPs annotated).</div>"
        "<div class='grid'>",
        "".join(cards),
        "</div></div>",

        f"<div class='section'><small class='soft'>file: {preds.name} • score: {score_col} • threshold: {thr:.3f}</small></div>",

        # Leaflet JS + custom JS
        "<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js' "
        "integrity='sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=' crossorigin=''></script>",
        f"<script>{SCRIPT}</script></body></html>",
    ]

    (out / "index.html").write_text("\n".join(html))
    print(f"✓ wrote {out / 'index.html'}")

if __name__ == "__main__":
    main()
