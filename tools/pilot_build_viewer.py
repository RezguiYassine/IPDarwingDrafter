"""Build a self-contained local HTML viewer: TIF | CN-fusion | phaseA | phaseB,
side by side, for every pilot figure, with prev/next keyboard navigation.

Usage: python build_viewer.py
Output: output/pilot_viewer/index.html (open directly in a browser, no server needed)
"""
import sqlite3, os, json, shutil
import cv2

RUNS = {
    "cnfusion": ("output/pilot_cnfusion", "CN-fusion"),
    "phaseA":   ("output/pilot_phaseA",   "Full-CNN phaseA (SG+D2C)"),
    "phaseB":   ("output/pilot_phaseB",   "Full-CNN phaseB (SG+D2C mix0.40)"),
}
OUT = "output/pilot_viewer"
ASSETS = f"{OUT}/assets"
os.makedirs(ASSETS, exist_ok=True)


def load_db(db_path):
    con = sqlite3.connect(db_path)
    cols = [c[1] for c in con.execute("PRAGMA table_info(results)").fetchall()]
    return {(r["patent_id"], r["sketch_id"]): r
            for r in (dict(zip(cols, row)) for row in con.execute("SELECT * FROM results"))}

runs_data = {key: load_db(f"{path}/results.db") for key, (path, _) in RUNS.items()}

# union of all (patent, sketch) keys seen across the 3 runs, sorted for stable order
all_keys = sorted(set().union(*[set(d.keys()) for d in runs_data.values()]))

figs = []
tif_cache = {}
for pat, sk in all_keys:
    entry = {"patent": pat, "sketch": sk, "configs": {}}
    tif_path = None
    for key, (root, label) in RUNS.items():
        r = runs_data[key].get((pat, sk))
        if r is None:
            entry["configs"][key] = {"status": "not_run"}
            continue
        tif_path = tif_path or r.get("input_path")
        status = r["status"]
        svg_rel = None
        if status == "ok":
            svg_src = f"{root}/{pat}/vectors/{sk}.svg"
            if os.path.exists(svg_src):
                svg_name = f"{pat}_{sk}_{key}.svg"
                shutil.copy(svg_src, f"{ASSETS}/{svg_name}")
                svg_rel = f"assets/{svg_name}"
        entry["configs"][key] = {
            "status": status,
            "svg": svg_rel,
            "n_edges": r.get("s2_n_edges"),
            "n_prims": r.get("s3_n_primitives"),
            "keypoint_src": r.get("s2_keypoint_src"),
            "s0_labels": r.get("s0_n_labels"),
        }
    # convert TIF -> PNG once per figure
    if tif_path and os.path.exists(tif_path) and tif_path not in tif_cache:
        img = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            png_name = f"{pat}_{sk}_input.png"
            h, w = img.shape
            scale = 900 / max(h, w)
            img_r = cv2.resize(img, (max(1,int(w*scale)), max(1,int(h*scale))))
            cv2.imwrite(f"{ASSETS}/{png_name}", img_r)
            tif_cache[tif_path] = f"assets/{png_name}"
    entry["input_png"] = tif_cache.get(tif_path)
    entry["patent_dir"] = os.path.basename(os.path.dirname(tif_path)) if tif_path else ""
    figs.append(entry)

json.dump(figs, open(f"{OUT}/figs.json", "w"))
print(f"{len(figs)} figures indexed -> {OUT}/figs.json")
print(f"assets written -> {ASSETS}  ({len(os.listdir(ASSETS))} files)")

CONFIG_LABELS = {k: v[1] for k, v in RUNS.items()}

HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Stage 2 Pilot — 3-way Comparison</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, Segoe UI, sans-serif; margin: 0; background:#111; color:#eee; }
  header { padding: 10px 16px; display:flex; align-items:center; gap:16px; background:#1a1a1a; border-bottom:1px solid #333; position:sticky; top:0; z-index:10;}
  header h1 { font-size:15px; margin:0; font-weight:600; flex:1; }
  .navbtn { background:#333; color:#eee; border:1px solid #555; border-radius:6px; padding:6px 14px; cursor:pointer; font-size:14px; }
  .navbtn:hover { background:#444; }
  .counter { font-variant-numeric: tabular-nums; min-width:110px; text-align:center; }
  .jump { width:60px; background:#222; color:#eee; border:1px solid #555; border-radius:4px; padding:4px; }
  main { padding:12px 16px 40px; }
  .figtitle { font-size:14px; margin-bottom:10px; color:#aaa; }
  .figtitle b { color:#fff; }
  .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; }
  .panel { background:#1c1c1c; border:1px solid #333; border-radius:8px; overflow:hidden; display:flex; flex-direction:column; }
  .panel .hdr { padding:6px 10px; font-size:12px; font-weight:600; background:#252525; border-bottom:1px solid #333; display:flex; justify-content:space-between; }
  .panel .body { flex:1; display:flex; align-items:center; justify-content:center; min-height:520px; background:#fff; }
  .panel img, .panel object { max-width:100%; max-height:640px; }
  .panel .stub { color:#900; font-size:13px; padding:20px; text-align:center; background:#1c1c1c; color:#e88; }
  .stats { font-size:11px; color:#888; padding:6px 10px; border-top:1px solid #333; }
  .stats span { margin-right:10px; }
  footer { padding:10px 16px; font-size:12px; color:#777; }
  kbd { background:#333; border:1px solid #555; border-radius:3px; padding:1px 6px; font-size:11px; }
</style></head>
<body>
<header>
  <button class="navbtn" id="prev">&larr; Prev</button>
  <span class="counter" id="counter"></span>
  <button class="navbtn" id="next">Next &rarr;</button>
  <input class="jump" id="jump" type="number" min="1"> <button class="navbtn" id="go">Go</button>
  <h1 id="figlabel"></h1>
</header>
<main>
  <div class="figtitle" id="subtitle"></div>
  <div class="grid" id="grid"></div>
</main>
<footer>Keyboard: <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> to navigate. Input TIF | """ + " | ".join(CONFIG_LABELS.values()) + """</footer>
<script>
const CONFIG_ORDER = """ + json.dumps(list(RUNS.keys())) + """;
const CONFIG_LABELS = """ + json.dumps(CONFIG_LABELS) + """;
let FIGS = __FIGS_JSON__;
let i = 0;

function statusNote(s) {
  const map = {
    "quality_gate_stage1": "stopped: Stage-1 quality gate",
    "quality_gate_stage2": "stopped: Stage-2 quality gate",
    "quality_gate_stage3": "stopped: Stage-3 quality gate",
    "quality_gate_stage4": "stopped: Stage-4 quality gate",
    "stage0": "error in Stage 0", "stage1": "error in Stage 1",
    "stage2": "error in Stage 2", "stage3": "error in Stage 3",
    "stage4": "error in Stage 4", "not_run": "not processed",
  };
  return map[s] || s;
}

function render() {
  const f = FIGS[i];
  document.getElementById("counter").textContent = (i+1) + " / " + FIGS.length;
  document.getElementById("jump").value = i+1;
  document.getElementById("figlabel").textContent = f.patent_dir + " / " + f.sketch;
  document.getElementById("subtitle").innerHTML =
    "<b>" + f.patent_dir + "</b> — sketch " + f.sketch;

  const grid = document.getElementById("grid");
  grid.innerHTML = "";

  // panel 0: input TIF
  const p0 = document.createElement("div"); p0.className = "panel";
  p0.innerHTML = `<div class="hdr"><span>Input (TIF)</span></div>
    <div class="body">${f.input_png ? `<img src="${f.input_png}">` : '<div class="stub">no input</div>'}</div>
    <div class="stats">&nbsp;</div>`;
  grid.appendChild(p0);

  for (const key of CONFIG_ORDER) {
    const c = f.configs[key] || {status: "not_run"};
    const p = document.createElement("div"); p.className = "panel";
    let body;
    if (c.svg) {
      body = `<object type="image/svg+xml" data="${c.svg}"></object>`;
    } else {
      body = `<div class="stub">${statusNote(c.status)}</div>`;
    }
    let stats = "";
    if (c.status === "ok") {
      stats = `<span>edges: ${c.n_edges ?? "-"}</span><span>prims: ${c.n_prims ?? "-"}</span>` +
              (c.keypoint_src ? `<span>kp: ${c.keypoint_src}</span>` : "");
    } else {
      stats = `<span style="color:#e88">${statusNote(c.status)}</span>`;
    }
    p.innerHTML = `<div class="hdr"><span>${CONFIG_LABELS[key]}</span></div>
      <div class="body">${body}</div>
      <div class="stats">${stats}</div>`;
    grid.appendChild(p);
  }
}

function go(delta) { i = Math.max(0, Math.min(FIGS.length-1, i+delta)); render(); }
document.getElementById("prev").onclick = () => go(-1);
document.getElementById("next").onclick = () => go(1);
document.getElementById("go").onclick = () => { i = Math.max(0, Math.min(FIGS.length-1, (+document.getElementById("jump").value)-1)); render(); };
window.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") go(-1);
  if (e.key === "ArrowRight") go(1);
});

render();
</script>
</body></html>
"""
# Inline the figure data directly (fetch() of local JSON is blocked by Chrome
# under file://, so double-clicking index.html must work with no server).
HTML = HTML.replace("__FIGS_JSON__", json.dumps(figs))
open(f"{OUT}/index.html", "w").write(HTML)
print(f"viewer -> {OUT}/index.html  ({os.path.getsize(f'{OUT}/index.html')/1e6:.1f} MB)")
