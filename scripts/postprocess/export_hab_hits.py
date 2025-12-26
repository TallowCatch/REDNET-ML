#!/usr/bin/env python3
import argparse, csv, glob, os, shutil, sys, html
from pathlib import Path
import pandas as pd
import numpy as np

# -------------------- helpers --------------------
def to_num(x): return pd.to_numeric(x, errors="coerce")
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, mode: str):
    if mode == "none": return
    if dst.exists() or dst.is_symlink(): dst.unlink()
    if mode == "symlink":
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)
    elif mode == "copy":
        shutil.copy2(src, dst)

def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)): return "-"
    return f"{float(v):.{d}g}"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "flh" not in df.columns and "nflh" in df.columns: df = df.rename(columns={"nflh": "flh"})
    if "kd490" not in df.columns and "Kd_490" in df.columns: df = df.rename(columns={"Kd_490": "kd490"})
    for c in ("chlor_a","flh","kd490","hab_label"):
        if c in df.columns: df[c] = to_num(df[c])
    return df

def find_tile_path(row, repo_root: Path) -> Path | None:
    tile = row.get("tile")
    if not isinstance(tile, str): return None
    csv_path = Path(row["__src_csv__"])
    tag_dir = csv_path.parent
    candidate = tag_dir / "tiles_png" / tile
    if candidate.exists(): return candidate
    matches = list(repo_root.rglob(tile))
    return matches[0] if matches else None

def scene_key(row):
    """Group within same week if values are identical."""
    tag = Path(row["__src_csv__"]).parent.name
    scene = str(row.get("scene_id",""))
    dt = pd.to_datetime(row.get("datetime"), errors="coerce", utc=True)
    week = None
    if pd.notna(dt): week = f"{dt.year}-W{int(dt.isocalendar().week)}"
    chl = round(float(row.get("chlor_a", 0)), 2)
    flh = round(float(row.get("flh", 0)), 3)
    kd  = round(float(row.get("kd490", 0)), 3)
    return (tag, week, scene, chl, flh, kd)

# -------------------- HTML --------------------
CSS = """
:root{
  --bg:#0b1a1a; --card:#132; --ink:#e8f5f5; --soft:#a8d0d0;
  --pill:#284; --pill-ink:#efe; --accent:#4b8; --accent2:#2a4;
  --shadow:0 1px 0 #1c2e2e inset;
}
*{box-sizing:border-box}
html,body{margin:0; padding:0}
body{
  font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,sans-serif;
  background:var(--bg); color:var(--ink);
  padding:28px 20px 40px;
}
.header{max-width:1200px;margin:0 auto 18px auto;text-align:center}
h1{margin:0 0 6px 0;font-size:2.1rem}
.subtitle{color:var(--soft);margin:0 0 18px 0}

/* ── QC TABS ───────────────────────────────────────────── */
.qc-wrap{max-width:1200px;margin:0 auto 28px auto}
.qc-tabs{
  display:flex; flex-wrap:wrap; gap:10px; justify-content:center;
  margin-bottom:12px;
}
.qc-tab{
  background:rgba(255,255,255,.06); color:var(--ink); border:1px solid rgba(255,255,255,.15);
  padding:6px 12px; border-radius:999px; cursor:pointer; font-size:13px;
  transition:all .15s ease-in-out;
}
.qc-tab.active{background:var(--accent); color:#063; border-color:transparent}

/* Stage with “liquid glass” morph */
.qc-stage{
  position:relative; width:100%; aspect-ratio:16/9;
  background:#0d1e1e; border-radius:14px; box-shadow:0 0 4px #041;
  overflow:hidden;
  transition:transform .36s cubic-bezier(.22,1,.36,1), box-shadow .36s ease;
}
.qc-stage.morph{
  transform:scale(.975);
  box-shadow:
    0 0 0 1px rgba(120,255,200,.25) inset,
    0 14px 42px rgba(0,0,0,.45),
    0 0 24px rgba(60,200,140,.15);
}
.qc-stage .qc-glass{
  position:absolute; inset:0; pointer-events:none; border-radius:14px;
  background:
    radial-gradient(120% 120% at 10% 0%, rgba(200,255,240,.12), transparent 35%),
    radial-gradient(100% 100% at 90% 100%, rgba(120,220,180,.10), transparent 40%);
  backdrop-filter: blur(6px) saturate(1.05);
  -webkit-backdrop-filter: blur(6px) saturate(1.05);
  opacity:0; transition:opacity .36s ease;
}
.qc-stage.morph .qc-glass{ opacity:.25; }

.qc-stage img{
  position:absolute; inset:0; width:100%; height:100%;
  object-fit:contain; background:#0d1e1e; display:block;
  opacity:0; transform:scale(1);
  transition:opacity .22s ease, transform .36s cubic-bezier(.22,1,.36,1);
}
.qc-stage img.active{ opacity:1; }
.qc-stage.morph img.active{ transform:scale(1.02); }

/* ── HAB cards (unchanged) ───────────────────────────── */
.gallery{max-width:1600px;margin:0 auto}
.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(380px,1fr)); gap:16px}
.card{background:var(--card); border-radius:16px; padding:12px; box-shadow:var(--shadow); display:flex; flex-direction:column}
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
"""

JS = """
// ── HAB card carousels
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
  show(0);
  fig.querySelector('.prev')?.addEventListener('click', ()=>show(i-1));
  fig.querySelector('.next')?.addEventListener('click', ()=>show(i+1));
  dots.forEach((d,idx)=>d.addEventListener('click', ()=>show(idx)));
});

// ── QC tabs with “liquid glass” morph
(function(){
  const stage = document.querySelector('.qc-stage');
  const tabs  = [...document.querySelectorAll('.qc-tab')];
  if(!stage || !tabs.length) return;
  const imgs  = [...stage.querySelectorAll('img')];
  // add glass overlay once
  const glass = document.createElement('div');
  glass.className = 'qc-glass';
  stage.appendChild(glass);

  let i=0, timer=null;
  function show(k){
    i=((k%imgs.length)+imgs.length)%imgs.length;
    imgs.forEach((im,j)=>im.classList.toggle('active', j===i));
    tabs.forEach((t,j)=>t.classList.toggle('active', j===i));

    // trigger morph animation
    stage.classList.add('morph');
    clearTimeout(timer);
    timer = setTimeout(()=>stage.classList.remove('morph'), 360);
  }
  show(0);
  tabs.forEach((t,idx)=>t.addEventListener('click', ()=>show(idx)));
})();
"""

# -------------------- HTML builder --------------------
def write_html(out_dir: Path, groups: dict, qc_dir: Path):
    charts = sorted(qc_dir.glob("*.png"))
    html_path = out_dir / "index.html"

    def label_from_path(p: Path) -> str:
        name = p.stem.replace("_", " ").strip()
        name = name.replace("chlor a", "chlor_a").replace("chla", "chlor_a")
        return name

    with open(html_path,"w") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>HAB Findings — Oman Coastline</title>"
                f"<style>{CSS}</style></head><body>")

        # Header
        f.write("<div class='header'>"
                "<h1>HAB Detection Summary</h1>"
                "<div class='subtitle'>Oman coastline — satellite-derived chlorophyll, FLH and turbidity anomalies (2017–2025)</div>"
                "</div>")

        # QC tabbed viewer
        if charts:
            f.write("<div class='qc-wrap'>")
            f.write("<div class='qc-tabs'>")
            for j,ch in enumerate(charts):
                lab = html.escape(label_from_path(ch))
                cls = "qc-tab active" if j==0 else "qc-tab"
                f.write(f"<button class='{cls}' type='button'>{lab}</button>")
            f.write("</div>")
            f.write("<div class='qc-stage'>")
            for j,ch in enumerate(charts):
                rel = os.path.relpath(ch, start=out_dir)
                cls = "active" if j==0 else ""
                f.write(f"<img class='{cls}' src='{html.escape(rel)}' loading='lazy'/>")
            # .qc-glass is appended by JS
            f.write("</div></div>")

        # Cards grid (unchanged)
        f.write("<div class='gallery'><div class='grid'>")
        def sort_key(k):
            tag, wk, scene, chl, flh, kd = k
            return (tag, wk or "", -chl, -flh)
        for key in sorted(groups.keys(), key=sort_key):
            imgs = groups[key]
            tag, week, scene, chl, flh, kd = key

            f.write("<div class='card'>")
            f.write("<div class='figure'><div class='viewport'>")
            for j,img in enumerate(imgs):
                src = img["dst_rel"] or img["src_rel"]
                cls = "active" if j==0 else ""
                dt  = html.escape(img.get("dt","") or "")
                f.write(f"<img class='{cls}' data-dt='{dt}' src='{html.escape(src)}' loading='lazy'/>")
            f.write("</div>")
            if len(imgs)>1:
                f.write("<div class='dtbar'><span class='dtpill'></span></div>")
                f.write("<div class='controls'><div class='btn prev'>&lsaquo;</div><div class='btn next'>&rsaquo;</div></div>")
                f.write("<div class='nav'>" + "".join("<div class='dot'></div>" for _ in imgs) + "</div>")
            f.write("</div>")  # figure

            f.write("<div class='meta'>")
            f.write("<div class='pills'>")
            f.write(f"<span class='pill'>{html.escape(tag)}</span>")
            if week: f.write(f"<span class='pill'>{html.escape(week)}</span>")
            f.write("</div>")
            if scene: f.write(f"<div class='kv'>scene: {html.escape(scene)}</div>")
            f.write(f"<div class='kv'>chlor_a: {fmt(chl)} mg m⁻³</div>")
            f.write(f"<div class='kv'>FLH/nFLH: {fmt(flh)}</div>")
            f.write(f"<div class='kv'>Kd490: {fmt(kd)} m⁻¹</div>")
            f.write("</div>")
            f.write("</div>")  # card
        f.write("</div></div>")  # grid/gallery

        f.write(f"<script>{JS}</script></body></html>")
    print(f"✓ wrote {html_path}")

# -------------------- main --------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob",default="data/aerial_*_20*/chip_indices_clean_hab.csv")
    ap.add_argument("--out_dir",default="qc/hab_hits_inspect")
    ap.add_argument("--qc_dir",default="qc/qc_all",help="folder with QC plots")
    ap.add_argument("--min_chl",type=float,default=1.0)
    ap.add_argument("--min_flh",type=float,default=0.2)
    ap.add_argument("--link_mode",choices=["copy","symlink","none"],default="copy")
    args=ap.parse_args()

    ensure_dir(Path(args.out_dir))
    repo_root=Path(".").resolve()

    groups={}
    for csv_path in glob.glob(args.glob):
        df=pd.read_csv(csv_path)
        df["__src_csv__"]=csv_path
        df=normalize_columns(df)
        if "hab_label" not in df.columns: continue

        pos=df["hab_label"].astype(float)>0.5
        chl_ok=df.get("chlor_a",0)>=args.min_chl
        flh_ok=df.get("flh",0)>=args.min_flh
        keep=pos & (chl_ok | flh_ok)

        for _,row in df[keep].iterrows():
            src=find_tile_path(row,repo_root)
            if not src: continue
            dst=Path(args.out_dir)/Path(src).name
            dst_rel=None
            try:
                link_or_copy(src,dst,args.link_mode)
                dst_rel=dst.name
            except shutil.SameFileError:
                pass

            src_rel=os.path.relpath(src,start=args.out_dir)
            key=scene_key(row)

            dt = pd.to_datetime(row.get("datetime"), errors="coerce", utc=True)
            dt_str = dt.isoformat() if pd.notna(dt) else ""

            groups.setdefault(key,[]).append({
                "src_rel": src_rel, "dst_rel": dst_rel, "dt": dt_str
            })

    write_html(Path(args.out_dir), groups, Path(args.qc_dir))

if __name__=="__main__":
    main()
